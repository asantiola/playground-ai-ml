from pymongo import MongoClient
from ollama import Client, ChatResponse, EmbedResponse
import os
import json

ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
ollama_url = f"http://{ollama_host}:11434"

print(f"Ollama URL: {ollama_url}")

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

client = Client(
    host=ollama_url
)

options = {
    "temperature": llm_temp
}

template="""You are an assistant for question-answering tasks. 
    Use the following documents to answer the question. 

    If you don't know the answer, just say that you don't know. 

    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Documents: {documents} 
    Answer: 
"""

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

def query(question):
    print(f"Question: {question}\n")
    documents = get_query_results(question)
    response : ChatResponse = client.chat(
        model=llm_model,
        messages=[
            {
                "role": "user",
                "content": template.format(question=question, documents=documents)
            }
        ],
        options=options
    )
    print(f"Answer: {response.message.content}\n\n")

questions = "/workspace/data/questions.txt"
with open(questions) as file:
    for line in file:
        query(line.rstrip())
