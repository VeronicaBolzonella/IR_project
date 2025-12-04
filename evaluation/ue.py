import TruthTorchLM as ttlm
from TruthTorchLM.long_form_generation import StructuredDecompositionAPI
from TruthTorchLM.long_form_generation.generation import long_form_generation_with_truth_value
from TruthTorchLM.long_form_generation.claim_check_methods.question_answer_generation import QuestionAnswerGeneration
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def generate_with_ue(prompt, model=None, seed=42):
    '''
    Output looks like this: 
    
    For claim check version:
    truth_dict = {
        "generated_text": generated_text,
        "claims": claims,
        "normalized_truth_values": normalized_truth_values,
        "unnormalized_truth_values": unnormalized_truth_values,
        "claim_check_method_details": method_spec_outputs,
        ...
    }
    
    For non-claim check version:
    truth_dict = {
        "generated_text": generated_text,
        "normalized_truth_values": normalized_truth_values,
        "unnormalized_truth_values": unnormalized_truth_values,
    }
    
    
    '''
    
    # Decomposition method splits the generated text into claims
    decomp_method= StructuredDecompositionAPI(model='openrouter/openai/gpt-4o-mini', decomposition_depth=1)
    
    # print(decomp_method("This is a simple test sentence."))
    
    sum_of_eigen = ttlm.truth_methods.SumEigenUncertainty(entailment_model_device='cuda' if torch.cuda.is_available() else 'cpu')
    #p_true = ttlm.truth_methods.PTrue()
    
    truth_methods = [sum_of_eigen]

    claim_check_methods = [QuestionAnswerGeneration(model=model, num_questions=2, truth_methods=truth_methods, entailment_model_device='cuda' if torch.cuda.is_available() else 'cpu', seed=seed )]
    
    messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

    
    output = long_form_generation_with_truth_value(
        model=model,
        messages=messages,
        decomp_method=decomp_method,
        claim_check_methods=claim_check_methods, # maybe the truth methods need to be wrapped somehow
        generation_seed=seed,  ## here can also add context if we want to add documents here instead of in the prompt
    )
    
    # output = ttlm.generate_with_truth_value(
    #     model=model,
    #     messages=messages,
    #     truth_methods=truth_methods,
    #     generation_seed=seed
    # )
    
    return output