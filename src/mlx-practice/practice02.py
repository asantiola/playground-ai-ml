import openai 

# practice code using openai.OpenAi.chat.completions

base_url = "http://localhost:12434/v1"

client = openai.OpenAI(
  base_url = base_url,
  api_key = "docker"
)

completion = client.chat.completions.create(
    # model = "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    model = "mlx-community/gpt-oss-20b-MXFP4-Q4",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of the United States of America?"}
        ],
)

print(completion.choices[0].message.content)
