from zhipuai import ZhipuAI
import json
from tqdm import tqdm
import re
import fire
def evaluation(file_path: str= "./lora1-dolly-shepherd-7b/global_results_0.json",
results_file_path:str= "./lora1-dolly-shepherd-7b/global_results_rou_0.json",
):

    client = ZhipuAI(
        api_key="1dfeb51999ee15d2147ae2d3e9b131c6.rBlaCFI1esnyECvr"
    )  # 请填写您自己的APIKey

    """file_path = (
        "./lora-shepherd-7b/8/predictions_client_6_0.json"  # 请修改为您文件的实际路径)
    )"""
    #file_path = "./global_eva20/client1.json"
    #file_path = "./base_client5.json"
    #file_path = "./private_eva20/client1.json"
    #file_path = "./base_evaluation_client.json"
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    scores = []
    pattern = r"^(?:[0-9]|[1-9][0-9]|100)$"

    for item in tqdm(data):
        instruction = item["instruction"]
        #context = item["context"]
        context=" "
        response = item["Reference"]
        # generated_response = item["generated_response"]
        generated_response = item["Prediction"]
        # 构造包含提示词的请求消息
        prompt = f"根据下面的指令和上下文，根据期望答案与你的认知，评估生成答案的合理性。请给出0到100的分数，其中100分表示完全合理，0分表示完全不合理，输出给出一个数字即可。\n\n指令：{instruction}\n上下文：{context}\n期望答案：{response}\n生成的回答：{generated_response}\n\n评分："

        messages = [
            {
                "role": "system",
                "content": "请评估以下回答的准确性和相关性，并给出分数。在回复中只给出分数即可，不用解释原因",
            },
            {"role": "user", "content": prompt},
        ]

        # 发起请求
        response = client.chat.completions.create(
            model="glm-4",  # 使用的模型名称
            messages=messages,
            stream=True,
        )
        for chunk in response:
            score = chunk.choices[0].delta.content  # 获取评分结果
            print(score)
            if re.match(pattern, score):
                scores.append(int(score))
            print(scores)

    average_score = sum(scores) / len(scores) if scores else 0

    # 将平均分和所有评分结果保存到文件中
    '''results_file_path = (
        "./global_eva20/acc_client1.json"  # 请修改为您希望保存结果的实际路径
    )'''
    '''results_file_path = (
        "./eva3/client0_7_7.json"  # 请修改为您希望保存结果的实际路径
    )'''
    '''results_file_path = (
        "./private_eva20/acc_client1.json"  # 请修改为您希望保存结果的实际路径
    )'''
    with open(results_file_path, "w", encoding="utf-8") as results_file:
        json.dump(
            {"average_score": average_score, "scores": scores},
            results_file,
            ensure_ascii=False,
            indent=4,
        )

    print(f"评分完成，平均分为：{average_score}，详细结果已保存到：{results_file_path}")

if __name__ == "__main__":
    fire.Fire(evaluation)
