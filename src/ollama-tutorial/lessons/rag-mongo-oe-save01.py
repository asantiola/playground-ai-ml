from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from ollama import Client, EmbedResponse
from langchain_community.document_loaders import TextLoader
import os
import json
import time

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

ollama_embeddings_model = ollama_config.get("ollama_embeddings_model", "mxbai-embed-large")
ollama_embeddings_dimensions = ollama_config.get("ollama_embeddings_dimensions", 512)

print(f"Embedding model '{ollama_embeddings_model}'")

client = Client(
    host=ollama_url
)

doc_path = "/workspace/data/documents"
files = [os.path.join(doc_path, file) for file in os.listdir(doc_path)]

# Load and prepare documents
docs_lists = [TextLoader(file).load_and_split() for file in files]

def get_oe_embedding(data):
    response : EmbedResponse = client.embed(
        model=ollama_embeddings_model,
        input=data
    )
    return response.embeddings[0]

docs_to_insert = [{
    "text": doc.page_content,
    "embedding": get_oe_embedding(doc.page_content)
} for doc_list in docs_lists for doc in doc_list]

mongo_client[rag_db].drop_collection(rag_collection)
collection = mongo_client[rag_db][rag_collection]

result = collection.insert_many(docs_to_insert)

print("collection.estimated_document_count(): ", collection.estimated_document_count(), "\n")

# Create your index model, then create the search index
index_name="vector_index"
search_index_model = SearchIndexModel(
  definition = {
    "fields": [
      {
        "type": "vector",
        "numDimensions": ollama_embeddings_dimensions,
        "path": "embedding",
        "similarity": "cosine"
      }
    ]
  },
  name = index_name,
  type = "vectorSearch"
)
collection.create_search_index(model=search_index_model)

# Wait for initial sync to complete, maybe slow when creating indices...
print("Polling to check if the index is ready...")
predicate=None
if predicate is None:
   predicate = lambda index: index.get("queryable") is True

while True:
   indices = list(collection.list_search_indexes(index_name))
   if len(indices) and predicate(indices[0]):
      break
   time.sleep(5)
print(index_name + " is ready for querying.")
