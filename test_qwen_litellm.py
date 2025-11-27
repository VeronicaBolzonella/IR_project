import os
from litellm import completion

# Testing whether litellm api call works

# Set the base path
# In your environment, set the api key as "OPENROUTER_API_KEY"
os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"

print("Testing LiteLLM + Qwen + OpenRouter...")

response = completion(
    model="openrouter/qwen/qwen-2.5-7b-instruct",    # <-- adjust name if needed
    messages=[{"role": "user", "content": "Hello, what is 2+2?"}],
    max_tokens=20,
)

print("Model returned:")
print(response.choices[0].message["content"])