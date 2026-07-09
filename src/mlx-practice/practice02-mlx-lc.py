from mlx_vlm import load, generate

model_path="mlx-community/gemma-4-12B-it-qat-6bit"
model, processor = load(model_path)

print(f"processor: {processor}\n")

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

prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f"prompt: {prompt}\n")

response = generate(
    model, 
    processor, 
    prompt, 
    verbose=False,
    max_tokens=8192,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)

print(f"response: {response}")
