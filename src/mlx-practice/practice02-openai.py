import openai 

# practice code using openai.OpenAi.chat.completions

base_url = "http://localhost:12434/v1"

client = openai.OpenAI(
  base_url = base_url,
  api_key = "mlx-lm"
)

completion = client.chat.completions.create(
    model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    # model = "mlx-community/gpt-oss-20b-MXFP4-Q4",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Create a short story about a chicken and a cat. Just create the story. No need to provide the analysis or explanation."}
        ],
)

print(completion.choices[0].message.content)
