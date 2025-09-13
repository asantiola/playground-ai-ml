from ollama import Client
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

client = Client(
    host=ollama_url
)

response = client.chat(
    messages = [
        {
            "role": "user",
            "content": "Today is 28th of February 2024. What is the date 2 days from today?",
        },
        {
            "role": "user",
            "content": "Write in in YYYY-MM-DD format.",
        },
        {
            "role": "user",
            "content": "Explain how you got your answer. How confident are you on your answer, between 0 and 1?",
        },
    ],
    model = llm_model,
    options = {
        "temperature": llm_temp
    }
)

print(response.message.content)

response = client.chat(
    messages = [
        {
            "role": "user",
            "content": "How many dimensions are there? How confident are you on your answer, between 1 to 10?",
        },
    ],
    model = llm_model,
    options = {
        "temperature": llm_temp
    }
)

print(response.message.content) 
