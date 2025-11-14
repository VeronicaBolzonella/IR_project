import TruthTorchLM as ttlm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.qwen import Qwen

def generate_with_ue(prompt, model, api=True):
    sum_of_eigen = ttlm.truth_methods.SumEigenUncertainty()
    # sv to implement

    truth_methods = [sum_of_eigen]

    #tokenizer = AutoTokenizer.from_pretrained(model)
    
    tokenizer=model.tokenizer
    
    messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
    #api = False
    #model = "distilgpt2"  

    if api:
        # Generate text with truth values (api model)
        output = ttlm.generate_with_truth_value(
            model=model,
            messages=messages,
            truth_methods=truth_methods
        )
    else:
        output = ttlm.generate_with_truth_value(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            truth_methods=truth_methods,
            max_new_tokens=100,
            temperature=0.7
        )
    return output


model = Qwen()
ue_values = generate_with_ue("What is the capital of France?", model=model, api=False)
print(ue_values)
