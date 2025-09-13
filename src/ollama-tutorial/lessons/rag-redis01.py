from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_redis import RedisConfig, RedisVectorStore
import os
import json


ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
ollama_url = f"http://{ollama_host}:11434"

print(f"Ollama URL: {ollama_url}")

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


template="""You are an assistant for question-answering tasks. 
    Use the following documents to answer the question. 

    If you don't know the answer, just say that you don't know. 

    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Documents: {documents} 
    Answer: 
"""

index_name = "vector_index"
config = RedisConfig(
    index_name=index_name,
    redis_url=redis_url,
)
vector_store = RedisVectorStore(
    embeddings=hf_embeddings, 
    config=config
)
retriever = vector_store.as_retriever(
    search_type="similarity",
    searcg_kwargs={"k": 2}
)

llm = ChatOllama(
    model=llm_model,
    temperature=llm_temp,
    base_url=ollama_url
)


# Define prompt
prompt = PromptTemplate(template=template, input_variables=["question", "documents"],)

# Create chain combining the prompt template and LLM
rag_chain = prompt | llm

# Define the RAG Application
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        
        # Get the answer from LLM
        answer = self.rag_chain.invoke({ 
            "documents": documents, 
            "question": question,
        })
        
        return answer

# Initialize the RAG Application
rag_application = RAGApplication(retriever, rag_chain)

def query(question):
    print(f"Question: {question}\n")
    answer = rag_application.run(question)
    print(f"Answer: {answer.content}\n\n")

questions = "/workspace/data/questions.txt"
with open(questions) as file:
    for line in file:
        query(line.rstrip())
