import json
from mlx_vlm import generate, load
from pydantic import BaseModel, Field

model_path = "mlx-community/gemma-4-12B-it-qat-6bit"


# 1. Define your Response class
class Response(BaseModel):
    reasoning: str = Field(
        description="Step-by-step logical deduction analyzing each hint to answer the question."
    )

    answer: str = Field(
        description="The answer to the question"
    )

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

puzzle_einstein = """There are five houses of different colors adjacent to one another on a road. 
In each house lives a man of different nationality. 
Each man has a favorite drink, a favorite brand of cigarettes, and keeps a different kind of pet.

The Englishman lives in the red house.
The Swede keeps dogs.
The Dane drinks tea.
The green house is just to the left of the white house.
The owner of the green house drinks coffee.
The Pall Mall smoker keeps birds.
The owner of the yellow house smokes Dunhills.
The man in the center house drinks milk.
The Norwegian lives in the first house.
The Blend Smoker has a neighbor who keeps cats.
The man who smokes Blue Masters drinks bier.
The man who keeps horses lives next to the Dunhill smoker.
The German smokes Prince.
The Norwegian lives next to the blue house.
The Blend smoker has a neighbor who drinks water.

Who owns the fish?
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": puzzle_einstein},
]

# Load model and processor components
model, processor = load(model_path)
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print("Generating structured text...")
# 3. Generate raw text from the model
text_output = generate(
    model,
    processor,
    prompt,
    verbose=False,
    temperature=0.1,
    top_p=0.95,
    top_k=64,
)

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
