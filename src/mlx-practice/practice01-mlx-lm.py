from mlx_lm import load, generate

# model_name = "mlx-community/gpt-oss-20b-MXFP4-Q4"
model_name = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
model, tokenizer = load(model_name)

prompt = "Why do birds fly south? Keep your answer brief. No need for explanations."

if tokenizer.chat_template is not None:
    messages = [{
        "role": "user",
        "content": prompt,
    }]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

response = generate(model, tokenizer, prompt, verbose=True)
print(f"response: {response}")
