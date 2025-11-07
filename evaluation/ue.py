import TruthTorchLM as ttlm

def generate_with_ue(prompt, api_model):
    sum_of_eigen = ttlm.truth_methods.SumEigenUncertainty()
    # sv to implement

    truth_methods = [sum_of_eigen]
    
    messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

    # Generate text with truth values (Huggingface model)
    output_api_model = ttlm.generate_with_truth_value(
        model=api_model,
        messages=messages,
        truth_methods=truth_methods
    )

    return output_api_model



