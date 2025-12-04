import TruthTorchLM as ttlm
from TruthTorchLM.long_form_generation import StructuredDecompositionAPI
from TruthTorchLM.long_form_generation.generation import long_form_generation_with_truth_value
from TruthTorchLM.long_form_generation.claim_check_methods.question_answer_generation import QuestionAnswerGeneration
import torch

def generate_with_ue(prompt:str, model:str, seed=42)->dict:
    """
    Generate asnwer to the promp using Qwen and returns a dictionary including generated answer,
    claims in which the anwer is broken down, truth values (normalized and not), and other method specs.

    Args:
        prompt (str): Question to prompt the model, in RAG includes prompt and docs
        model (str): API of the model to prompt.
        seed (int, optional): Seed for the model's generation reproducibility. Defaults to 42.

    Returns:
        output (dict): dictionary with answer and truth values

    Notes:
        results looks like this: 
    
        truth_dict = {
            "generated_text": generated_text,
            "claims": claims,
            "normalized_truth_values": normalized_truth_values,
            "unnormalized_truth_values": unnormalized_truth_values,
            "claim_check_method_details": method_spec_outputs,
            ...
        }
    """
    
    # Decomposition method splits the generated text into claims
    decomp_method= StructuredDecompositionAPI(model='openrouter/openai/gpt-4o-mini', decomposition_depth=1)
        
    sum_of_eigen = ttlm.truth_methods.SumEigenUncertainty(
        entailment_model_device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    #p_true = ttlm.truth_methods.PTrue()
    
    truth_methods = [sum_of_eigen]

    claim_check_methods = [QuestionAnswerGeneration(
        model=model, 
        num_questions=2, 
        truth_methods=truth_methods, 
        entailment_model_device='cuda' if torch.cuda.is_available() else 'cpu', seed=seed
    )]

    messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

    output = long_form_generation_with_truth_value(
        model=model,
        messages=messages,
        decomp_method=decomp_method,
        claim_check_methods=claim_check_methods, 
        generation_seed=seed,  
    )
    
    return output