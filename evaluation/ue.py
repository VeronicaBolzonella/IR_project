import TruthTorchLM as ttlm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.qwen import Qwen

def generate_with_ue(prompt, api=False):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # This is required for TruthTorch
    model.config.output_hidden_states = True
    model.config.output_attentions = True
    model.config.use_cache = False
    model.config.return_dict = True
    
    
    sum_of_eigen = ttlm.truth_methods.SumEigenUncertainty()
    # sv to implement

    truth_methods = [sum_of_eigen]

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
            temperature=0.7,
            model_kwargs={
                "output_attentions": True,
                "output_hidden_states": True,
                "use_cache": False,
                "return_dict": True
            }   
        )
    return output



ue_values = generate_with_ue("What is the capital of France?", api=False)
print(ue_values)
