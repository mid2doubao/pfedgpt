import os
import json
import fire
import torch
import transformers
from tqdm import tqdm
import sys
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
import random

from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
)
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

#from huggingface_hub import login
import os


def load_test_dataset(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_predictions_to_test_dataset(
    model, tokenizer, test_dataset, device, prompter,max_length=512
):
    predictions = []
    for item in tqdm(test_dataset, desc="Generating Predictions"):
        predicted_response = predict_response(
            model, tokenizer, item, device, prompter,max_length
        )
        item["predicted_response"] = predicted_response  # 添加预测结果到测试数据中
        predictions.append(item)

    # 返回包含预测结果的测试数据集
    return predictions


def predict_response(model, tokenizer, item, device,prompter, max_length=512):
    instruction = item["instruction"]
    #context = item["context"]
    #input_text = f"{instruction} {context}"
    prompt = prompter.generate_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    '''inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)'''
    '''generation_config = GenerationConfig(
            temperature=0.6,
            top_p=0.9,
            top_k=40,
            num_beams=4,
            do_sample=True, 
            no_repeat_ngram_size=6, 
            repetition_penalty=1.8
            #**kwargs,
        )'''
    '''generation_config = GenerationConfig(
            temperature=0.5, #0.01
            top_p=0.7,  #0.75
            top_k=40,   #40
            num_beams=4,  #4
            do_sample=True, 
            no_repeat_ngram_size=6, 
            repetition_penalty=1.8
            #**kwargs,
        )'''
    generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,  #0.75
            top_k=40,   #40
            num_beams=8,  #4
            do_sample=False, 
            no_repeat_ngram_size=6, 
            repetition_penalty=1.8
            #**kwargs,
        )
    input_length = input_ids.size(1)
    '''if input_length >= max_length:
        max_length = input_length + 128'''

    with torch.no_grad():
        '''generation_output = model.generate(
            input_ids=input_ids, attention_mask = inputs['attention_mask'].to(device),generation_config=generation_config, return_dict_in_generate=True, output_scores=True, max_new_tokens=120,
        )'''  # 50 is not enough
        generation_output = model.generate(
            input_ids=input_ids,generation_config=generation_config, return_dict_in_generate=True, output_scores=True, max_new_tokens=120,
        )
    #predicted_response = tokenizer.decode(output[0], skip_special_tokens=True)
    #s = gen_output.sequences[0]
    #output = tokenizer.decode(s)
    #yield prompter.get_response(output)
    #return predicted_response
    outputs = []
    '''for j in range(len(generation_output.sequences)):
            s = generation_output.sequences[j]
            outputs.append(tokenizer.decode(s))'''
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    outputs=prompter.get_response(output)
    print(outputs)
    return outputs

'''def predict_response(model, tokenizer, item, device, max_length=50):
    instruction = item["instruction"]
    context = item["context"]
    input_text = f"{instruction} {context}"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    input_length = input_ids.size(1)
    #if input_length >= max_length:
    max_length = input_length + 10

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids, max_length=max_length
        )  # 50 is not enough
    predicted_response = tokenizer.decode(output[0], skip_special_tokens=True)
    return predicted_response'''


def evaluate_model_on_test_dataset(
    model, tokenizer, test_dataset, device, weight_function=None
):
    correct_predictions = 0
    total_predictions = len(test_dataset)

    if weight_function:
        selected_indices = weight_function(test_dataset)
        test_dataset = [test_dataset[i] for i in selected_indices]
        total_predictions = len(test_dataset)

    predictions = []

    for item in tqdm(test_dataset, desc="Evaluating"):
        predicted_response = predict_response(model, tokenizer, item, device)
        true_response = item["response"]
        if predicted_response == true_response:
            correct_predictions += 1

        # 打印预测和实际结果
        print(f"Predicted: {predicted_response}\nTrue: {true_response}\n")

        # 保存预测和实际结果
        predictions.append({"predicted": predicted_response, "true": true_response})

    accuracy = correct_predictions / total_predictions
    return accuracy, predictions  # 返回accuracy和predictions列表


def random_weight_function(dataset, percentage=0.05):
    total_samples = len(dataset)
    selected_samples = int(total_samples * percentage)
    selected_indices = random.sample(range(total_samples), selected_samples)
    return selected_indices


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights_path: str = "",
    lora_config_path: str = "",  # provide only the file path, excluding the file name 'adapter_config.json'
    prompt_template: str = "alpaca_short",  # The prompt template to use, will default to alpaca.
    server_name: str = "127.0.0.1",
    share_gradio: bool = False,
    test_dataset_path: str = "./data/8/global_test.json",
    result_save_path: str = "./Global_evaluation_client.json",
    test_sample_percentage: float = 0.1,
):
    base_model = base_model  # or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    print(prompt_template)
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if not lora_weights_path.endswith(".bin"):
        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                torch_dtype=torch.float16,
            )
        elif device == "mps":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                device_map={"": device},
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = prepare_model_for_int8_training(model)
        print(lora_config_path)
        config = LoraConfig.from_pretrained(lora_config_path)
        lora_weights = torch.load(lora_weights_path)
        model = PeftModel(model, config)
        set_peft_model_state_dict(model, lora_weights, "default")
        del lora_weights

    '''model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )'''

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    test_dataset = load_test_dataset(test_dataset_path)

    # 生成预测并保存
    test_dataset_with_predictions = save_predictions_to_test_dataset(
        model, tokenizer, test_dataset, device,prompter
    )

    # 将包含预测的测试数据集保存到文件中
    with open(result_save_path, "w") as f:
        json.dump(test_dataset_with_predictions, f, ensure_ascii=False, indent=4)

    print(f"Predictions saved to {result_save_path}")


if __name__ == "__main__":
    fire.Fire(main)
