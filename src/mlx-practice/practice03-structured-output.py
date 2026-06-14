import json
from mlx_vlm import generate, load
from pydantic import BaseModel, Field

model_path = "mlx-community/gemma-4-12B-it-6bit"


# 1. Define your Response class
class Response(BaseModel):
    answer: str = Field(description="The answer to the question")

    confidence_score: float = Field(
        description="Score on how confident you are on the answer, from 0 to 1."
    )

# 2. Inject the JSON Schema string into the System Prompt text
# This tells the model exactly how to format its text output.
schema_json = json.dumps(Response.model_json_schema(), indent=2)

system_prompt = f"""You are a helpful assistant. 
You MUST respond exclusively with a single JSON object that strictly conforms to this JSON Schema:

{schema_json}

Do not include any conversational text, markdown blocks like ```json, or explanations. Output only the raw valid JSON object.
"""

human_prompt = "What is the capital of France?"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": human_prompt},
]

# Load model and processor components
model, tokenizer = load(model_path)
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print("Generating structured text...")
# 3. Generate raw text from the model
text_output = generate(model, tokenizer, prompt, verbose=False)
print(f"\nDEBUG:\n{text_output}\n")

try:
    # 4. Parse the output back into your complete Response object
    response = Response.model_validate_json(text_output.text)

    print("\n--- Final Structured Output ---")
    print(f"Object type: {type(response)}")
    print(f"Complete Object: {response}")
    print(f"Answer: {response.answer}")
    print(f"Confidence Score: {response.confidence_score}")

except Exception as e:
    print(f"\nParsing Failed: {e}")
    print(f"Raw Model Output was:\n{text_output}")
