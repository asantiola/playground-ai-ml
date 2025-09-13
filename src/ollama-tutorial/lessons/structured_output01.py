from ollama import Client
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import json

ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
ollama_url = f"http://{ollama_host}:11434"

ollama_config_file = "/workspace/data/ollama_conf.json"
with open(ollama_config_file) as file:
    ollama_config = json.load(file)

llm_model = ollama_config.get("llm_model", "llama3.1")
llm_temp = ollama_config.get("llm_temp", 0.0)

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")
print("\n", "-------------------------\n")

class Pet(BaseModel):
    score: float = Field(
        description="Confidence score between 0 to 1 that input is a pet."
    )
    name: str = Field(
        description="Name of the pet."
    )
    animal: str = Field(
        description="Animal type of the pet."
    )
    food: Optional[List[str]] = Field(
        description="Food for the pet."
    )

class Pets(BaseModel):
    pets: List[Pet] = Field(
        description="List of pets."
    )

client = Client(
    host=ollama_url
)

def parse_pets(input: str) -> Pets:
    prompt = f"Validate input is a pet. Input={input}"

    response = client.chat(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=llm_model,
        format=Pets.model_json_schema(),
        options={
            "temperature": llm_temp
        }
    )

    return Pets.model_validate_json(response.message.content)

inputs = [
    "We were given Smirnoff and Chicklet, 2 lovebirds that both love seeds.",
    "I have parrot named Ray, and he has a sister named May. She loves nuts.",
    "I have a recurring dream about a woman that transforms into a parrot.",
    "My first dog is Whitey loves bones and treats.",
    "Roger is a cat that I have found, and she loves milk.",
    "I bought a duckling and named it Bibe."
    "I had 2 dogs named Jinggoy and Jinky. Jinggoy loves bones."
]

def validate_pets(pets: Pets) -> List[Pet]:
    valid_pets = []
    for i, pet in enumerate(pets):
        print(f"validating pet {i}: {pet}")
        if pet.score >= 0.7:
            valid_pets.append(pet)
    return valid_pets

my_pets = []
for i, input in enumerate(inputs):
    all_pets = parse_pets(input)
    print(f"validating all_pets {i}: {all_pets.pets}")
    valid_pets = validate_pets(all_pets.pets)
    for pet in valid_pets:
        my_pets.append(pet)

for pet in my_pets:
    print(f"my pet: {pet}")
