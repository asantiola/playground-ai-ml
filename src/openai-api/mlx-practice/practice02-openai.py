import openai 
import os

# practice code using openai.OpenAi.chat.completions

workspaces = os.environ.get(
    "WORKSPACES",
    "/workspaces"
)

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
    api_key=api_key,
)

completion = client.chat.completions.create(
    # model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    model = "mlx-community/gpt-oss-20b-MXFP4-Q4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant, Mario from the Super Mario Brothers."},
        {"role": "user", "content": "Tell me a joke."}
    ],
)

print(completion.choices[0].message.content)
