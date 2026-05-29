import openai 
import os

# practice code using openai.OpenAi.chat.completions

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://model-runner.docker.internal/engines/v1"
)

client = openai.OpenAI(
  base_url = openai_base_url,
  api_key = "docker"
)

completion = client.chat.completions.create(
    # model="ai/llama3.1:latest", 
    model="ai/gemma4:E4B", 
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Create a short story about a chicken and a cat. Just create the story. No need to provide the analysis or explanation."}
        ],
)

print(completion.choices[0].message.content)
