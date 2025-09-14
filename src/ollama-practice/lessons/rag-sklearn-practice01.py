# TextLoader
from langchain_community.document_loaders import TextLoader

# # UnstructuredWordDocumentLoader
# from langchain_community.document_loaders import UnstructuredWordDocumentLoader
# import nltk
# nltk.download("punkt_tab")
# nltk.download("averaged_perceptron_tagger_eng")

# # PyPDFLoader
# from langchain_community.document_loaders import PyPDFLoader

# # Web Pages
# from langchain_community.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
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

ollama_config_file = "/home/dclxvi/repo/playground-ai/data/ollama_conf.json"
with open(ollama_config_file) as file:
    ollama_config = json.load(file)

# Initialize the LLM
# e.g. llama3, mistral, llama3.2, phi3
llm_model = ollama_config.get("llm_model", "llama3.2")
llm_temp = ollama_config.get("llm_temp", 0.0)

# Initialize Embeddings model
# e.g. sentence-transformers/all-mpnet-base-v2, thenlper/gte-small
embeddings_model = ollama_config.get("sentencetransformer_embeddings_model", "sentence-transformers/all-mpnet-base-v2")
hf_embeddings_device = ollama_config.get("hf_embeddings_device", "cpu")

# Initialize Splitter config
splitter_chunk_size = ollama_config.get("splitter_chunk_size", 200)
splitter_chunk_overlap = ollama_config.get("splitter_chunk_overlap", 0)

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")
print(f"Embedding model '{embeddings_model}' maximum sequence length: {SentenceTransformer(embeddings_model).max_seq_length}")
print(f"hf_embeddings_device: {hf_embeddings_device}")
print(f"Splitter chunk size: {splitter_chunk_size}")
print(f"Splitter chunk overlap: {splitter_chunk_overlap}")

doc_path = "/home/dclxvi/repo/playground-ai/data/documents-txt"
files = [os.path.join(doc_path, file) for file in os.listdir(doc_path)]

# urls = [
#     "http://apache:80/",
#     "http://apache:80/Billiards.html",
#     "http://apache:80/Guitars.html",
#     "http://apache:80/SoftwareEngineering.html",
# ]

# Load and prepare documents
docs = [TextLoader(file).load() for file in files]
# docs = [UnstructuredWordDocumentLoader(file, mode="elements", strategy="fast").load() for file in files]
# docs = [PyPDFLoader(file).load() for file in files]
# docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=splitter_chunk_size,
    chunk_overlap=splitter_chunk_overlap
)
doc_splits = text_splitter.split_documents(docs_list)

hf_embeddings = HuggingFaceEmbeddings(
    model_name=embeddings_model,
    model_kwargs={"device": hf_embeddings_device},
    encode_kwargs={"normalize_embeddings": False},
)

vector_store = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=hf_embeddings
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

questions = "/home/dclxvi/repo/playground-ai/data/questions.txt"
with open(questions) as file:
    for line in file:
        query(line.rstrip())
