from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "mlx-community/gemma-4-12B-it-qat-6bit"
model, processor = load(model_path)

image_path = "./data/images/handwriting.jpg"

messages = [
    {
        "role": "user",
        "content": "Describe what you see in the image file."
    }
]

formatted_prompt = apply_chat_template(
    processor, 
    model.config, 
    messages, 
    num_images=1
)

output = generate(
    model, 
    processor, 
    formatted_prompt,
    image=image_path,
    temperature=0.0      
)

print(output.text)
