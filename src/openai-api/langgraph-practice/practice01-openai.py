import openai 
import os

# practice code using openai.OpenAi.chat.completions

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

client = openai.OpenAI(
    base_url=openai_base_url,
    api_key=api_key
)

completion = client.chat.completions.create(
    # model="ai/llama3.1:latest", 
    model="mlx-community/gemma-4-12B-it-6bit", 
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Create a short story about a chicken and a cat. Just create the story. No need to provide the analysis or explanation."}
        ],
)

print(completion.choices[0].message.content)
