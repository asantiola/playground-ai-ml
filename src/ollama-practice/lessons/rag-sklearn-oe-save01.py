from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings
import os
import json


ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
ollama_url = f"http://{ollama_host}:11434"

print(f"Ollama URL: {ollama_url}")

ollama_config_file = "/workspace/data/ollama_conf.json"
with open(ollama_config_file) as file:
    ollama_config = json.load(file)

ollama_embeddings_model = ollama_config.get("ollama_embeddings_model", "mxbai-embed-large")

# Initialize the LLM
# e.g. llama3, mistral, llama3.2, phi3
llm_model = ollama_config.get("llm_model", "llama3.2")
llm_temp = ollama_config.get("llm_temp", 0.0)

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")
print(f"Embedding model '{ollama_embeddings_model}'")

doc_path = "/workspace/data/documents"
files = [os.path.join(doc_path, file) for file in os.listdir(doc_path)]

# Load and prepare documents
docs_lists = [TextLoader(file).load_and_split() for file in files]

oe_embeddings = OllamaEmbeddings(
    base_url=ollama_url,
    model=ollama_embeddings_model
)

# SKLearn persist path
persist_path = "/workspace/data/sklearn-oe"

vector_store = SKLearnVectorStore(
    persist_path=persist_path,
    embedding=oe_embeddings
)
for docs_list in docs_lists:
    vector_store.add_documents(docs_list)
vector_store.persist()

print(f"VectorStore at {persist_path} is ready for querying")
