# QWEN LOCAL (works on the cluster!)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen():
    def __init__(self, model_name:str="Qwen/Qwen2.5-7B-Instruct"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prompt(self, prompt:str):
        # up to 32,768 tokens
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response





# QWEN API VERSION - DOES NOT WORK (GIVES ERROR FOR API_KEY)
# Code from https://modelstudio.console.alibabacloud.com/?tab=doc#/doc/?type=model&url=2840915
# API_KEY => the environment variable in the terminal like this: export API_KEY="your_api_key_here"

import os
from openai import OpenAI

try:
    client = OpenAI(
        # The API keys for the Singapore and China (Beijing) regions are different. To obtain an API key, see https://modelstudio.console.alibabacloud.com/?tab=model#/api-key
        # If you have not configured an environment variable, replace the following line with your Model Studio API key: api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # The following URL is for the Singapore region. If you use a model in the China (Beijing) region, replace the URL with: https://dashscope.aliyuncs.com/compatible-mode/v1
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    print(client.api_key)
    completion = client.chat.completions.create(
        model="Qwen2.5-7B-Instruct",  
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Who are you?'}
            ]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Error message: {e}")
    print("For more information, see https://www.alibabacloud.com/help/en/model-studio/developer-reference/error-code")

