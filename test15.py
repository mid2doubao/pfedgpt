import os
from typing import List
from tqdm import tqdm
import fire
import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    PeftModel,
    PromptEncoderConfig,
    TaskType,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from fed_utils import (
    FedAvg2,
    client_selection,
    global_evaluation,
    GeneralClient15,
)#NewTrainer,
import datasets
from utils.prompter import Prompter
import copy
from collections import OrderedDict
import sys  # 导入sys模块
sys.setrecursionlimit(300000)  # 将默认的递归深度修改为3000
import json

datasets.utils.logging.set_verbosity_error()

from transformers import  TrainingArguments,DataCollatorForSeq2Seq
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"





def fl_finetune(
    # model/data params
    global_model: str = "",
    data_path: str = "./data",
    output_dir: str = "./lora-shepherd/",
    adapter_dir: str = "./lora-adapter6/",
    # FL hyperparamas
    client_selection_strategy: str = "random",
    client_selection_frac: float = 1,
    num_communication_rounds: int = 50,
    num_clients: int = 10,
    # Local training hyperparams
    local_batch_size: int = 32,  # 128,
    local_micro_batch_size: int = 8,  # 32
    local_num_epochs: int = 1,
    local_learning_rate: float = 5e-5,
    local_val_set_size: float = 0.2,
    local_save_steps: int = 3,
    cutoff_len: int = 512,
    # LoRA hyperparams
    lora_r: int = 8,  #16
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,  #0.05
    lora_target_modules: List[str] = [
        "q_proj","v_proj"
    ],
    # llm hyperparams
    train_on_inputs: bool = False,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca_short",  # The prompt template to use, will default to alpaca.
    # p tuning hyperparams
    adapter_package: str = "peft",
    adapter_method: str = "p-tuning",
    encoder_reparameterization_type: str = "MLP",
    encoder_hidden_size: int = 128,
    num_virtual_tokens: int = 20,
    # lora2 tuning hyperparams
    lora2_r: int = 16,
    lora2_alpha: int = 16,
    lora2_dropout: float = 0.05,
    lora2_target_modules: List[str] = [
        "v_proj",
    ],
    lora2: bool = True,
    best_perplexity = float('inf'),
    all_trained_clients_set = set()
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"adapter_package: {adapter_package}\n"
            f"adapter_method: {adapter_method}\n"
            f"encoder_reparameterization_type: {encoder_reparameterization_type}\n"
            f"encoder_hidden_size: {encoder_hidden_size}\n"
            f"num_virtual_tokens: {num_virtual_tokens}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"
    combined_params_dict = {}
    heterogeneity_weights_dict = {}
    prediction_counts_dict = {}
    flag = False
    last_fed_par=None
    flag1=False

    data_path = os.path.join(data_path, str(num_clients))
    assert (os.path.exists(data_path), "Please generate the data files for each client")

    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        global_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(global_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        '''full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )'''
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            None,
            data_point["response"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        tokenized_full_prompt['text_field'] = full_prompt

        if not train_on_inputs:
            '''user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )'''
            user_prompt = prompter.generate_prompt(
                data_point["instruction"],None
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably #生成任务 
        #print(tokenized_full_prompt)              #分类任务
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    print(model)
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))
    params_dict = copy.deepcopy(
        OrderedDict(
            (name, param.detach())
            for name, param in model.named_parameters()
            if "default" in name
        )
    )
    adapter_weight = get_peft_model_state_dict(model, params_dict, "default")
    '''att = AttentionWeight(
        input_dim=len(adapter_weight.keys()), output_dim=len(adapter_weight.keys())
    )'''
    '''global_client=GlobalClient(model=model,
            data_path=data_path,
            output_dir=output_dir,
            )
    global_client.preprare_local_dataset(generate_and_tokenize_prompt)'''
    all_clients = {
        client_id: GeneralClient15(
            client_id=client_id,
            model=model,
            data_path=data_path,
            output_dir=output_dir,
            init_adapter_weight=adapter_weight,
            phase1_train_weight=0.8,
            phase2_train_weight=0,
            test_data_weight=0.2,
        )
        for client_id in range(num_clients)
    }
    config.save_pretrained(output_dir)

    for client_id, client in all_clients.items():
        client.preprare_local_dataset(generate_and_tokenize_prompt)

    for epoch in tqdm(range(num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_id_set = client_selection(
            num_clients,
            client_selection_frac,
            client_selection_strategy,
            other_info=epoch,
        )
        selected_clients_set = [
            all_clients[client_id] for client_id in selected_clients_id_set
        ]
        
        all_trained_clients_set.update(selected_clients_id_set)
        collected_training_loss_per_params = []

        for client in selected_clients_set:
            client_id = client.client_id
            client.epoch=epoch

            print(
                "\nPreparing the local dataset and trainer for Client_{}".format(
                    client_id
                )
            )

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training(flag)

            

            client.build_local_trainer(
                tokenizer,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_num_epochs,
                local_learning_rate,
                group_by_length,
                ddp,
            )

            client.optim_agg()

            print("Local training starts ... ")
            client.train()


            # client.save_predictions(tokenizer)

            print("\nTerminating the local training of Client_{}".format(client_id))
            (
                model,
                local_dataset_len_dict,
                previously_selected_clients_set,
                last_client_id,
                training_loss_per_param,
            ) = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set
            )
        
            collected_training_loss_per_params.append(training_loss_per_param)

        print("Collecting the weights of clients and performing aggregation")
        model, global_training_loss_per_param = FedAvg2(
            model,
            selected_clients_id_set,
            output_dir,
            local_dataset_len_dict,
            epoch,
            all_clients,
            collected_training_loss_per_params,
        )
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, str(epoch), "adapter_model.bin"),
        )
    
        print(all_trained_clients_set)
        for client_id, client in all_clients.items():

            # 假设这里的 client 对象有一个方法来更新其模型
            all_clients[client_id].update_model(model,global_training_loss_per_param)
            all_clients[client_id].update_weight()
            print('2222')

        if epoch == num_communication_rounds - 1:
            '''torch.save(
                model.state_dict(), os.path.join(adapter_dir, "adapter_model.bin")
            )
            config.save_pretrained(adapter_dir)'''

            '''perplexity_path = os.path.join(output_dir, "best_perplexity.json")
            with open(perplexity_path, "w") as f:
                json.dump({"best_perplexity": best_perplexity}, f)'''

            final_output_dir = os.path.join(output_dir, "final_outputs")
            os.makedirs(final_output_dir, exist_ok=True)
            print('3333')
            for client_id in all_trained_clients_set:
                client = all_clients[client_id]
                client.initiate_local_training(flag)
                client.optimize_model()
                output=client.end_output()
                print('4444')
                combined_params_dict[client_id] = output["combined_params"]
                #heterogeneity_weights_dict[client_id] = output["heterogeneity_weight"]
                #prediction_counts_dict[client_id] = output["prediction_count"]

                output_data = output["combined_params"]
                output_path = os.path.join(final_output_dir, f"client_{client_id}_output")
                os.makedirs(output_path, exist_ok=True)
                torch.save(output_data, output_path + "/pytorch_model.bin")

                '''with open(os.path.join(output_path, "metadata.json"), 'w') as f:
                    json.dump({
                        "heterogeneity_weight": output["heterogeneity_weight"],
                        "prediction_count": output["prediction_count"]
                    }, f, indent=4)'''

                print(f'Data for client {client_id} saved successfully.')

        # Please design the evaluation method based on your specific requirements in the fed_utils/evaluation.py file.
   
if __name__ == "__main__":
    fire.Fire(fl_finetune)
