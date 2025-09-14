from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# Initialize Embeddings model
# e.g. sentence-transformers/all-mpnet-base-v2, thenlper/gte-small
embeddings_model = ollama_config.get("embeddings_model", "sentence-transformers/all-mpnet-base-v2")

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")
print(f"Embedding model '{ollama_embeddings_model}'")

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
retriever = vector_store.as_retriever(k=4)

# Define prompt
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 
        Use the following documents to answer the question. 

        If you don't know the answer, just say that you don't know. 

        Use three sentences maximum and keep the answer concise:
        Question: {question} 
        Documents: {documents} 
        Answer: 
        """,
    input_variables=["question", "documents"],
)

llm = ChatOllama(
    model=llm_model,
    temperature=llm_temp,
    base_url=ollama_url
)

# Create chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()

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
    print(f"Answer: {answer}\n\n")

questions = "/workspace/data/questions.txt"
with open(questions) as file:
    for line in file:
        query(line.rstrip())
