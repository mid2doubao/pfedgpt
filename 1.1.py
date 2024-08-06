import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from fed_utils import Lora_FedAvg, client_selection, DualPersonalizingClient,evaluation
from fed_utils import client_participation_scheduling
import datasets
from utils.prompter import Prompter
from transformers import BitsAndBytesConfig
from datasets import load_dataset

# 设置datasets库的日志级别为ERROR，以减少不必要的输出
datasets.utils.logging.set_verbosity_error()

def fl_finetune(
        # 模型和数据相关参数
        global_model: str = '',
        data_path: str = './data',
        output_dir: str = './lora-shepherd/',
        # 联邦学习超参数
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 1,
        num_communication_rounds: int = 20,
        num_clients: int = 8,
        client_epoch: int = 1,
        # 本地训练超参数
        local_batch_size: int = 32,
        local_micro_batch_size: int = 8,
        local_num_epochs: int = 1,
        local_learning_rate: float = 1.5e-4,
        cutoff_len: int = 512,
        # LoRA超参数
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj","v_proj"
        ],
        # LLM超参数
        train_on_inputs: bool = False,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,
        prompt_template_name: str = "alpaca_short",
        # debug选项
        debug_mode: bool = False,
        debug_samples_per_client: int = 100,  
        evaluation_output: str = "./results"      
):

    """
    联邦学习微调LLM的主函数
    """
    evaluation_output=output_dir
    print(evaluation_output)
    # 检查必要参数
    assert global_model, "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    # 设置数据路径
    data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(data_path), "Please generate the data files for each client"

    # 设置设备和分布式训练参数
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    config = LlamaConfig.from_pretrained(global_model)
    config.pad_token_id = 0

    # 加载全局模型和分词器
    pretrained_model = LlamaForCausalLM.from_pretrained(
        global_model,
        config = config,
        quantization_config=quantization_config,
        
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(global_model,legacy=False)

    def tokenize(prompt, add_eos_token=True):
        """
        对prompt进行分词
        """
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
        """
        生成prompt并进行分词
        """
        if "context" not in data_point:
            data_point["context"] = ""
    
    # 检查并处理response字段
        if "output" in data_point:
            data_point["response"] = data_point.pop("output")
        
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]
        return tokenized_full_prompt

    # 准备模型进行8位量化训练
    pretrained_model = prepare_model_for_kbit_training(pretrained_model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(pretrained_model, lora_config)

    # 设置模型并行
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    previously_selected_clients_set = set()
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    pdar = tqdm(range(num_communication_rounds))

    # 开始联邦学习的主循环
    for epoch in pdar:

        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=client_epoch)
        
        # 对每个选中的客户端进行训练
        for client_id in selected_clients_set:
            client = DualPersonalizingClient(client_id, model, data_path, output_dir)
            client.preprare_local_dataset(generate_and_tokenize_prompt, debug_mode, debug_samples_per_client)
            client.build_local_trainer(tokenizer,
                                       local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       local_num_epochs,
                                       local_learning_rate,
                                       group_by_length,
                                       ddp)

            client.initiate_local_training()
            client.train()
            model, local_dataset_len_dict, previously_selected_clients_set = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set
            )
            
            del client

        # 聚合客户端模型
        print("Collecting the weights of clients and performing aggregation")
        global_adapter = Lora_FedAvg(
            selected_clients_set,
            output_dir,
            local_dataset_len_dict,
            epoch,
        )

        save_dir = os.path.join(output_dir, str(epoch))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "global_lora_weights.bin")

        # 只保存 LoRA 相关的权重
        lora_weights = {k: v for k, v in global_adapter.items() if 'lora' in k.lower()}

        # 保存权重
        torch.save(lora_weights, save_path)

    ########################################################  创建一个新的全局模型  ########################################################
        # 创建一个临时的 PeftModel 并加载聚合的 LoRA 权重
        model = get_peft_model(pretrained_model, lora_config)
        model.load_state_dict(global_adapter, strict=False)
        # 合并 LoRA 权重到基础模型
        model = model.merge_and_unload()
        # 基于合并后的模型创建一个新的 LoRA 模型
        model = get_peft_model(model, lora_config)
        model.train()

    ##################################################### evaluation ###############################################################

    selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                            other_info=client_epoch)

    for client_id in selected_clients_set:
        global_eval_model = get_peft_model(pretrained_model, lora_config)
        global_adapter = torch.load(os.path.join(output_dir,f"{client_epoch-1}/global_lora_weights.bin"))
        global_eval_model.load_state_dict(global_adapter, strict=False)
        local_eval_model = get_peft_model(pretrained_model, lora_config)
        local_adapter = torch.load(os.path.join(output_dir,f"{client_epoch-1}/local_output_{client_id}", "adapter_model.bin"))
        local_eval_model.load_state_dict(local_adapter, strict=False)

        local_data = load_dataset("json", data_files=os.path.join(data_path,  f"local_val_{client_id}.json"))
        if debug_mode:
            local_data["train"] = local_data["train"].select(range(min(len(local_data["train"]), debug_samples_per_client)))
        local_val_dataset = local_data["train"].shuffle().map(generate_and_tokenize_prompt)

        name = f"global_results_{client_id}.json"
        client_participation_scheduling.evaluation(global_eval_model,local_val_dataset,tokenizer,prompter,client_id, evaluation_output, name) 
        name = f"local_results_{client_id}.json"
        client_participation_scheduling.evaluation(local_eval_model,local_val_dataset,tokenizer,prompter,client_id, evaluation_output, name)

if __name__ == "__main__":
    fire.Fire(fl_finetune)