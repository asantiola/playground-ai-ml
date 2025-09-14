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

from langchain_community.vectorstores import SKLearnVectorStore
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain.prompts import ChatPromptTemplate
# from langchain.prompts import PromptTemplate

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import json


ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
ollama_url = f"http://{ollama_host}:11434"

print(f"Ollama URL: {ollama_url}")

# apache_host = os.environ.get("APACHE_HOST", "localhost")
# apache_url = f"http://{apache_host}:80"

# print(f"Apache URL: {apache_url}")

ollama_config_file = "/workspace/data/ollama_conf.json"
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

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")
print(f"Embedding model '{embeddings_model}' maximum sequence length: {SentenceTransformer(embeddings_model).max_seq_length}")
print(f"hf_embeddings_device: {hf_embeddings_device}")

doc_path = "/workspace/data/documents"
# doc_path = "/workspace/data/documents-doc"
# doc_path = "/workspace/data/documents-pdf"
files = [os.path.join(doc_path, file) for file in os.listdir(doc_path)]
# urls = [ f"{apache_url}/", f"{apache_url}/Billiards.html", f"{apache_url}/Guitars.html", f"{apache_url}/SoftwareEngineering.html"]

# Load and prepare documents
docs_lists = [TextLoader(file).load_and_split() for file in files]
# docs_lists = [UnstructuredWordDocumentLoader(file, mode="elements", strategy="fast").load_and_split() for file in files]
# docs_lists = [PyPDFLoader(file).load_and_split() for file in files]
# docs_lists = [WebBaseLoader(url).load_and_split() for url in urls]

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

system_prompt = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise:

    {context} 
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# prompt = PromptTemplate(
#     template="""
#         You are an assistant for question-answering tasks. 
#         Use the following pieces of retrieved context to answer the question.
#         If you don't know the answer, just say that you don't know. 
#         Use three sentences maximum and keep the answer concise:

#         Context: {context}
#         Question: {input}
#     """,
#     input_variables=["context"]
# )

llm = ChatOllama(
    model=llm_model,
    temperature=llm_temp,
    base_url=ollama_url
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def query(question):
    print(f"Question: {question}\n")
    answer = rag_chain.invoke({ "input" : question })
    print(f"Answer: {answer["answer"]}\n\n")

questions = "/workspace/data/questions.txt"
with open(questions) as file:
    for line in file:
        query(line.rstrip())
