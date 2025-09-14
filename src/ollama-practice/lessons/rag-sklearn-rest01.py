from flask import Flask, request, jsonify

# TextLoader
# from langchain_community.document_loaders import TextLoader

# Word Documents
# from langchain_community.document_loaders import UnstructuredWordDocumentLoader
# import nltk
# nltk.download("punkt_tab")
# nltk.download("averaged_perceptron_tagger_eng")

# PDF
from langchain_community.document_loaders import PyPDFLoader

# Web Pages
# from langchain_community.document_loaders import WebBaseLoader

from langchain_community.vectorstores import SKLearnVectorStore
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_ollama import ChatOllama
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

# Initialize the LLM
# e.g. llama3, mistral, llama3.2, phi3
llm_model = ollama_config.get("llm_model", "llama3.2")
llm_temp = ollama_config.get("llm_temp", 0.0)

# Initialize Embeddings model
# e.g. sentence-transformers/all-mpnet-base-v2, thenlper/gte-small
embeddings_model = ollama_config.get("embeddings_model", "sentence-transformers/all-mpnet-base-v2")
hf_embeddings_device = ollama_config.get("hf_embeddings_device", "cpu")

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")
print(f"Embedding model '{embeddings_model}' maximum sequence length: {SentenceTransformer(embeddings_model).max_seq_length}")
print(f"hf_embeddings_device: {hf_embeddings_device}")

doc_path = "/workspace/data/documents"
files = [os.path.join(doc_path, file) for file in os.listdir(doc_path)]

# urls = [
#     "<https://mywebpage/posts/2025-01-15-data/>",
#     "<https://mywebpage/posts/2025-01-15-details/>",
# ]

# Load and prepare documents
# docs_lists = [TextLoader(file).load() for file in files]
# docs_lists = [UnstructuredWordDocumentLoader(file, mode="elements", strategy="fast").load() for file in files]
docs_lists = [PyPDFLoader(file).load() for file in files]
# docs_lists = [WebBaseLoader(url).load() for url in urls]

hf_embeddings = HuggingFaceEmbeddings(
    model_name=embeddings_model,
    model_kwargs={"device": hf_embeddings_device},
    encode_kwargs={"normalize_embeddings": False},
)

vector_store = SKLearnVectorStore(
    embedding=hf_embeddings
)
for docs_list in docs_lists:
    vector_store.add_documents(docs_list)
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
    return rag_application.run(question)


app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def index():
    return 'Hello Lex!'

@app.route('/question', methods=['POST'])
def ask_question():
    data = request.get_json()
    answer = query(data['question'])
    return jsonify(answer), 201

app.run()
