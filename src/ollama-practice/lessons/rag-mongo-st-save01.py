# TextLoader
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import os
import json
import time

mongodb_host = os.environ.get("MONGODB_HOST", "localhost")
mongodb_user = os.environ.get("MONGODB_USER", "user")
mongodb_pass = os.environ.get("MONGODB_PASS", "pass")
mongodb_conn = f"mongodb://{mongodb_user}:{mongodb_pass}@{mongodb_host}:27017/?directConnection=true"
print(f"MongoDB host string: {mongodb_host}")

ollama_config_file = "/workspace/data/ollama_conf.json"
with open(ollama_config_file) as file:
    ollama_config = json.load(file)

# Initialize the LLM
# e.g. llama3, mistral, llama3.2, phi3
llm_model = ollama_config.get("llm_model", "llama3.2")
llm_temp = ollama_config.get("llm_temp", 0.0)

# Initialize Embeddings model
# e.g. sentence-transformers/all-mpnet-base-v2, thenlper/gte-small
sentencetransformer_embeddings_model = ollama_config.get("sentencetransformer_embeddings_model", "thenlper/gte-small")
sentence_transformer = SentenceTransformer(sentencetransformer_embeddings_model)
embedding_dimension = sentence_transformer.get_sentence_embedding_dimension()

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")
print(f"Embedding model '{sentencetransformer_embeddings_model}' maximum sequence length: {sentence_transformer.max_seq_length}")
print(f"Embedding dimension: {embedding_dimension}")

doc_path = "/workspace/data/documents"
files = [os.path.join(doc_path, file) for file in os.listdir(doc_path)]

# Load and prepare documents
docs_lists = [TextLoader(file).load_and_split() for file in files]

def get_st_embedding(data):
    embedding = sentence_transformer.encode(data)
    return embedding.tolist()

docs_to_insert = [{
    "text": doc.page_content,
    "embedding": get_st_embedding(doc.page_content)
} for doc_list in docs_lists for doc in doc_list]

client = MongoClient(mongodb_conn)
print("client.server_info():", client.server_info(), "\n")

rag_db = "rag_st_db"
rag_collection = "rag_st_collection"

client[rag_db].drop_collection(rag_collection)
collection = client[rag_db][rag_collection]

result = collection.insert_many(docs_to_insert)

print("collection.estimated_document_count(): ", collection.estimated_document_count(), "\n")
# print(f"result.inserted_ids: {result.inserted_ids}\n")
# print("collection.find_one(): ", collection.find_one(), "\n")


# Create your index model, then create the search index
index_name="vector_index"
search_index_model = SearchIndexModel(
  definition = {
    "fields": [
      {
        "type": "vector",
        "numDimensions": embedding_dimension,
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
