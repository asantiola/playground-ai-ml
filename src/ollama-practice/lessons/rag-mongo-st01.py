from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
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
hf_embeddings_device = ollama_config.get("hf_embeddings_device", "cpu")

print(f"Embedding model '{sentencetransformer_embeddings_model}'")
print(f"hf_embeddings_device: {hf_embeddings_device}")

hf_embeddings = HuggingFaceEmbeddings(
    model_name=sentencetransformer_embeddings_model,
    model_kwargs={"device": hf_embeddings_device},
    encode_kwargs={"normalize_embeddings": False},
)

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")

doc_path = "/workspace/data/documents"
files = [os.path.join(doc_path, file) for file in os.listdir(doc_path)]

client = MongoClient(mongodb_conn)
print("client.server_info():", client.server_info(), "\n")

rag_db = "rag_st_db"
rag_collection = "rag_st_collection"
collection = client[rag_db][rag_collection]

print("collection.estimated_document_count(): ", collection.estimated_document_count(), "\n")



template="""You are an assistant for question-answering tasks. 
    Use the following documents to answer the question. 

    If you don't know the answer, just say that you don't know. 

    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Documents: {documents} 
    Answer: 
"""

index_name = "vector_index"

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=hf_embeddings,
    index_name=index_name,
    relevance_score_fn="cosine",
)
retriever = vector_store.as_retriever(k=4)

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
