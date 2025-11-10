from mlx_lm import load, generate

model_path = "mlx-community/gpt-oss-20b-MXFP4-Q4"
# model_path = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
model, tokenizer = load(model_path)

print(f"tokenizer: {tokenizer}\n")

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant, Mario from the Super Mario Brothers.",
    },
    {
        "role": "user",
        "content": "Tell me a joke.",
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

print(f"prompt: {prompt}\n")

response = generate(model, tokenizer, prompt, verbose=False)

print(f"response: {response}")
