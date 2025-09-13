# TextLoader
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
import redis
import os
import json


redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_user = os.environ.get("REDIS_USER", "default")
redis_pass = os.environ.get("REDIS_PASS", "redis_pass")
redis_url = f"redis://{redis_user}:{redis_pass}@{redis_host}:6379"
print(f"Redis host: {redis_host}")

ollama_config_file = "/workspace/data/ollama_conf.json"
with open(ollama_config_file) as file:
    ollama_config = json.load(file)

# Initialize the LLM
# e.g. llama3, mistral, llama3.2, phi3
llm_model = ollama_config.get("llm_model", "llama3.2")
llm_temp = ollama_config.get("llm_temp", 0.0)

# Initialize Embeddings model
# e.g. sentence-transformers/all-mpnet-base-v2, thenlper/gte-small
embeddings_model = ollama_config.get("embeddings_model", "sentence-transformers/all-mpnet-base-v2")
hf_embeddings_device = ollama_config.get("hf_embeddings_device", "cpu")
sentence_transformer = SentenceTransformer(embeddings_model)
embedding_dimension = sentence_transformer.get_sentence_embedding_dimension()

hf_embeddings = HuggingFaceEmbeddings(
    model_name=embeddings_model,
    model_kwargs={"device": hf_embeddings_device},
    encode_kwargs={"normalize_embeddings": False},
)

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")
print(f"Embedding model '{embeddings_model}' maximum sequence length: {sentence_transformer.max_seq_length}")
print(f"Embedding dimension: {embedding_dimension}")
print(f"hf_embeddings_device: {hf_embeddings_device}")

doc_path = "/workspace/data/documents"
files = [os.path.join(doc_path, file) for file in os.listdir(doc_path)]

# Load and prepare documents
docs_lists = [TextLoader(file).load_and_split() for file in files]

index_name = "vector_index"
config = RedisConfig(
    index_name=index_name,
    redis_url=redis_url,
)

client = redis.from_url(redis_url)
print("client.ping(): ", client.ping())

client.flushall()

vector_store = RedisVectorStore(
    embeddings=hf_embeddings, 
    config=config
)

for docs_list in docs_lists:
    vector_store.add_documents(docs_list)

client.save()
client.quit()
