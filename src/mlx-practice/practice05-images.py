from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
import sys

model_path = "mlx-community/gemma-4-12B-it-qat-4bit"
model, processor = load(model_path)

image_path = "./data/images/handwriting.jpg"

prompt = """Give a brief description of what you see, then do the following:
- If you see text, print what you see. If it is not in English, translate it.
- If it is a puzzle, approach problems step-by-step, verify boundary conditions, 
and rigorously check your assumptions before calculating the final answer.
"""

messages = [
    {
        "role": "user",
        "content": prompt,
    }
]

formatted_prompt = apply_chat_template(
    processor, 
    model.config, 
    messages, 
    num_images=1
)

for chunk in stream_generate(
    model, 
    processor, 
    formatted_prompt,
    image=image_path,
    temperature=0.0,
):
    sys.stdout.write(chunk.text)
    sys.stdout.flush()
print("\n\n")
