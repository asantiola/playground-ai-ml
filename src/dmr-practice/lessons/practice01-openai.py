import openai 

# practice code using openai.OpenAi.chat.completions

base_url = "http://localhost:12434/engines/v1"

client = openai.OpenAI(
  base_url = base_url,
  api_key = "docker"
)

completion = client.chat.completions.create(
    model="ai/llama3.1:latest", 
    # model="ai/gpt-oss:latest", 
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Create a short story about a chicken and a cat. Just create the story. No need to provide the analysis or explanation."}
        ],
)

print(completion.choices[0].message.content)
