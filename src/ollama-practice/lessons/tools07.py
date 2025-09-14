from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
import datetime
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

ollama_embeddings_model = ollama_config.get("ollama_embeddings_model", "mxbai-embed-large")
print(f"Embedding model '{ollama_embeddings_model}'")
print("\n", "-------------------------\n")

oe_embeddings = OllamaEmbeddings(
    base_url=ollama_url,
    model=ollama_embeddings_model
)

@tool
def days_from_today(days: int = 0) -> datetime.datetime:
    """
    Computes a new datetime based on number of days from today.

    Args:
        delta (int): The number of days difference from today. 
    """
    return datetime.datetime.now() + datetime.timedelta(days=days)

available_functions = {
    "days_from_today": days_from_today,
}

tools = [
    days_from_today,
]

chat = ChatOllama(
    base_url=ollama_url,
    model=llm_model,
    temperature=llm_temp
).bind_tools(tools)

def invoke(input: str):
    result = chat.invoke(input)
    print(f"result: {result}\n")

    for tool in result.tool_calls:
        if function_to_call := available_functions.get(tool["name"]):
            print(f"calling {tool["name"]}")
            print(f"args {tool["args"]}")
            output = function_to_call.invoke(tool["args"])
            print(f"output: {output}")
        else:
            print(f"unhandled function {tool["name"]}")
        print("\n", "-------------------------\n")

invoke("Get yesterday's date. Get today's date.")
invoke("What is the date 2 days from now?")
