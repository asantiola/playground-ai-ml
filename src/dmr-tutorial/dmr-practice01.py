import openai 

base_url = "http://localhost:12434/engines/v1"

client = openai.OpenAI(
  base_url = base_url,
  api_key = "docker"
)

completion = client.chat.completions.create(
    model="ai/llama3.1:latest", 
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of the United States of America?"}
        ],
)

print(completion.choices[0].message.content)
