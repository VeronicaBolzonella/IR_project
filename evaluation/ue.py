import TruthTorchLM as ttlm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.qwen import Qwen

# def generate_with_ue(prompt, model=None, api=False, seed=42):
#     tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
#     if model is None:
#         model = AutoModelForCausalLM.from_pretrained(
#             "Qwen/Qwen2.5-0.5B-Instruct",
#             device_map="auto",
#             torch_dtype=torch.float16
#         )
    
#     sum_of_eigen = ttlm.truth_methods.SumEigenUncertainty()
#     semantic_entropy = ttlm.truth_methods.SemanticEntropy()
    
#     truth_methods = [sum_of_eigen, semantic_entropy]

#     messages = [
#                 {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ]

#     if api:
#         # Generate text with truth values (api model)
#         output = ttlm.generate_with_truth_value(
#             model=model,
#             messages=messages,
#             truth_methods=truth_methods,
#             generation_seed=seed
#         )
#     else:
#         output = ttlm.generate_with_truth_value(
#             model=model,
#             tokenizer=tokenizer,
#             messages=messages,
#             truth_methods=truth_methods,
#             max_new_tokens=100,
#             generation_seed=seed
#         )
#     return output


#qwen_model = Qwen()
#ue_values = generate_with_ue("What is the capital of France?", model = qwen_model, api=False)

# ue_values = generate_with_ue("What is the capital of France?", model=None, api=False) # none means its defined inside the function
# print(ue_values['normalized_truth_values'])


def generate_with_ue(prompt, model=None, api=False, seed=42):
    '''
    Output looks like this: 
    
    truth_dict = {
        "generated_text": generated_text,
        "claims": claims,
        "normalized_truth_values": normalized_truth_values,
        "unnormalized_truth_values": unnormalized_truth_values,
        "claim_check_method_details": method_spec_outputs,
    }
    
    '''
    
    # Decomposition method splits the generated text into claims
    decomp_method= ttlm.decomposiition_methods.StrucuredDecompositionAPI(model=model, decomposition_depth=1)
    
    sum_of_eigen = ttlm.truth_methods.SumEigenUncertainty()
    semantic_entropy = ttlm.truth_methods.SemanticEntropy()
    
    truth_methods = [sum_of_eigen, semantic_entropy]

    messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

    
    output = ttlm.long_form_generation_with_truth_value(
        model=model,
        messages=messages,
        decomp_method=decomp_method,
        claim_check_methods=truth_methods, # maybe the truth methods need to be wrapped somehow
        generation_seed=seed,  ## here can also add context if we want to add documents here instead of in the prompt
    )
    return output