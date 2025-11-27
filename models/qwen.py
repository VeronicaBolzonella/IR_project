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





# QWEN API VERSION 
# code from https://openrouter.ai/docs/docs/overview/models
# API_KEY => the environment variable in the terminal like this: export API_KEY="your_api_key_here"

from openai import OpenAI
import os

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("API_KEY"),
)
completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "IR_project", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "IR_project", # Optional. Site title for rankings on openrouter.ai.
  },
  model="qwen/qwen-2.5-7b-instruct",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)
