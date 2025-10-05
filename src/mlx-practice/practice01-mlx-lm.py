from mlx_lm import load, generate

# model_path = "mlx-community/gpt-oss-20b-MXFP4-Q4"
model_path = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
model, tokenizer = load(model_path)

chat_history = [
    {
        "role": "system",
        "context": "You are Mario from Super Mario Brothers. Answer as Mario."
    },
    {
        "role": "user",
        "content": "What are ducks?",
    }
]

prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)

response = generate(model, tokenizer, prompt, verbose=False)
print(f"response: {response}")
