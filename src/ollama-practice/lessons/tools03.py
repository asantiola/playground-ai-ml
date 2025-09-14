from ollama import Client, ChatResponse, EmbedResponse
from pydantic import BaseModel, Field
from pymongo import MongoClient
import os
import json


ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
ollama_url = f"http://{ollama_host}:11434"

mongodb_host = os.environ.get("MONGODB_HOST", "localhost")
mongodb_user = os.environ.get("MONGODB_USER", "user")
mongodb_pass = os.environ.get("MONGODB_PASS", "pass")
mongodb_conn = f"mongodb://{mongodb_user}:{mongodb_pass}@{mongodb_host}:27017/?directConnection=true"
print(f"MongoDB host string: {mongodb_host}")

mongo_client = MongoClient(mongodb_conn)
print("client.server_info():", mongo_client.server_info(), "\n")

rag_db = "rag_oe_db"
rag_collection = "rag_oe_collection"
collection = mongo_client[rag_db][rag_collection]
print("collection.estimated_document_count(): ", collection.estimated_document_count(), "\n")

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

client = Client(
    host=ollama_url
)

def get_oe_embedding(data):
    response : EmbedResponse = client.embed(
        model=ollama_embeddings_model,
        input=data
    )
    return response.embeddings[0]

# Define a function to run vector search queries
def get_query_results(query):
    """Gets results from a vector search query."""

    query_embedding = get_oe_embedding(query)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": 5
            }
        },
        { 
            "$project": {
                "_id": 0,
                "text": 1
            }
        }
    ]

    results = collection.aggregate(pipeline)

    array_of_results = []
    for doc in results:
        array_of_results.append(doc)
    return array_of_results

class Veagles(BaseModel):
    name: str = Field(
        description="Topic about Veagles."
    )
    confidence_score: float = Field(
        description="Confidence score on the scale of 0 to 1 on your answer."
    )
    info: str = Field(
        description="Information about the Veagles."
    )

def get_veagles_info(query: str) -> Veagles:
    documents = get_query_results(query)
    prompt_fmt = "Query: {query}. Documents={documents}. Provide a confidence score on the scale of 0 to 1 on your answer."
    response: ChatResponse = client.chat(
        model=llm_model,
        messages=[{ "role": "user", "content": prompt_fmt.format(query=query, documents=documents)}],
        options={ "temperature": 0.75 },
        format=Veagles.model_json_schema()
    )
    return Veagles.model_validate_json(response.message.content)

get_veagles_info_tool = {
    "type": "function",
    "function": {
        "name": "get_veagles_info",
        "description": "Get more information on Veagles related queries.",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "The query related to Veagles."}
            },
        },
    },
}

class Place(BaseModel):
    name: str = Field(
        description="Name of the city or country."
    )
    confidence_score: float = Field(
        description="Confidence score on the scale of 0 to 1 on your answer."
    )
    info: str = Field(
        description="Information about the city or country"
    )

def get_place_info(place: str) -> Place:
    prompt_fmt = "Describe this city or country: {place}. Provide a confidence score on the scale of 0 to 1 on your answer."
    response: ChatResponse = client.chat(
        model=llm_model,
        messages=[{ "role": "user", "content": prompt_fmt.format(place=place)}],
        options={ "temperature": 0.75 },
        format=Place.model_json_schema()
    )
    return Place.model_validate_json(response.message.content)

get_place_info_tool = {
    "type": "function",
    "function": {
        "name": "get_place_info",
        "description": "Get more information about a city or country",
        "parameters": {
            "type": "object",
            "required": ["place"],
            "properties": {
                "place": {"type": "string", "description": "The city or country"}
            },
        },
    },
}

class Animal(BaseModel):
    name: str = Field(
        description="Name of the animal"
    )
    confidence_score: float = Field(
        description="Confidence score on the scale of 0 to 1 on your answer."
    )
    info: str = Field(
        description="Information about the animal"
    )

def get_animal_info(animal: str) -> Animal:
    prompt_fmt = "Describe this animal: {animal}. Provide a confidence score on the scale of 0 to 1 on your answer."
    response: ChatResponse = client.chat(
        model=llm_model,
        messages=[{ "role": "user", "content": prompt_fmt.format(animal=animal)}],
        options={ "temperature": 0.75 },
        format=Animal.model_json_schema()
    )
    return Animal.model_validate_json(response.message.content)

get_animal_info_tool = {
    "type": "function",
    "function": {
        "name": "get_animal_info",
        "description": "Get more information about this animal",
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
        "description": "Fallback function for messages not about animals, Veagles, or places."
    }
}

available_functions = {
    "get_animal_info": get_animal_info,
    "get_place_info": get_place_info,
    "get_veagles_info": get_veagles_info,
}

tools = [
    get_animal_info_tool,
    get_veagles_info_tool,
    get_place_info_tool,
    catch_all_tool,
]

options = {
    "temperature": llm_temp
}

inputs = [
    "Give me more information about aardvarks.",
    "Who is Alex from the Veagles?",
    "Tell me something about Manila.",
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
