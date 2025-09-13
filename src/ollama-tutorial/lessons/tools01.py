from ollama import Client, ChatResponse
from pydantic import BaseModel, Field
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

client = Client(
    host=ollama_url
)

def add_two_numbers(a: int, b: int) -> int:
    return int(a) + int(b)

def subtract_two_numbers(a: int, b: int) -> int:
    return int(a) - int(b)

subtract_two_numbers_tool = {
    "type": "function",
    "function": {
        "name": "subtract_two_numbers",
        "description": "Subtract two numbers",
        "parameters": {
            "type": "object",
            "required": ["a", "b"],
            "properties": {
                "a": {"type": "integer", "description": "The first number"},
                "b": {"type": "integer", "description": "The second number"},
            },
        },
    },
}

class Animal(BaseModel):
    name: str = Field(
        description="Name of the animal"
    )
    confidence_score: float = Field(
        description="Confidence score between 0 and 1, that this is a valid animal."
    )
    info: str = Field(
        description="Information about the animal"
    )

def get_animal_info(animal: str) -> Animal:
    response: ChatResponse = client.chat(
        model=llm_model,
        messages=[{ "role": "user", "content": f"Describe this animal: {animal}"}],
        options={ "temperature": 0.75 },
        format=Animal.model_json_schema()
    )
    return Animal.model_validate_json(response.message.content)

get_animal_info_tool = {
    "type": "function",
    "function": {
        "name": "get_animal_info",
        "description": "Get more information only about this animal",
        "parameters": {
            "type": "object",
            "required": ["animal"],
            "properties": {
                "animal": {"type": "string", "description": "The animal"}
            },
        },
    },
}

catch_all_tool = {
    "type": "function",
    "function": {
        "name": "catch_all_function",
        "description": "Fallback function for messages not about addition, subtraction or animals."
    }
}

available_functions = {
    "add_two_numbers": add_two_numbers,
    "subtract_two_numbers": subtract_two_numbers,
    "get_animal_info": get_animal_info
}

tools = [
    add_two_numbers,
    subtract_two_numbers_tool,
    get_animal_info_tool,
    catch_all_tool
]

options = {
    "temperature": llm_temp
}

inputs = [
    "What is twenty minus 3?",
    "What is the weather in Sapporo today?",
    "What is an aardvark?",
    "Give me the sum of two and forty.",
    "Give me yesterday's run logs.",
]

for input in inputs:
    print(f"processing input for tool calling: {input}")
    response: ChatResponse = client.chat(
        model=llm_model,
        messages=[{ "role": "user", "content": input }],
        tools=tools,
        options=options
    )

    print(f"response.message: {response.message}")

    if response.message.tool_calls == None:
        print("response.message.tool_calls is None")
        continue

    for tool in response.message.tool_calls:
        if function_to_call := available_functions.get(tool.function.name):
            print(f"calling {tool.function.name}")
            output = function_to_call(**tool.function.arguments)
            print(f"output: {output}")
        else:
            print(f"unhandled function {tool.function.name}")
        print("\n", "-------------------------\n")
