import os
import litellm
from litellm import completion

# Testing whether litellm api call works

# Set the base path
# In your environment, set the api key as "OPENROUTER_API_KEY"
os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"

print("Testing LiteLLM + Qwen + OpenRouter...")

response1 = completion(
    model="openrouter/qwen/qwen-2.5-7b-instruct",    # <-- adjust name if needed
    messages=[{"role": "user", "content": "Give me 4 rhyming words"}],
    max_tokens=40,
    seed = 42,
)

print("Model 1 returned:")
print(response1.choices[0].message["content"])


response2 = completion(
    model="openrouter/qwen/qwen-2.5-7b-instruct",    # <-- adjust name if needed
    messages=[{"role": "user", "content": "Give me 4 rhyming words"}],
    max_tokens= 40,
    seed = 42,
)

print("Model 2 returned:")
print(response2.choices[0].message["content"])

"""
First test without seeds:
Model 1 returned:
Sure! Here are four rhyming words: cat, hat, mat, and fat.
Model 2 returned:
Sure! Here are four rhyming words: 

1. Cat
2. Hat
3. Rat
4. Flat
(IR_project) 
"""



""" 
Second Test with seed=42 ==> gives same result!!

Testing LiteLLM + Qwen + OpenRouter...
Model 1 returned:
Sure! Here are four words that rhyme with "tree":

1. be
2. key
3. free
4. thee
Model 2 returned:
Sure! Here are four words that rhyme with "tree":

1. be
2. key
3. free
4. thee
(IR_project) 
"""