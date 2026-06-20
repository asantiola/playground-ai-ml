from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from lesson10_common import selection_embeddings
import os
import shutil

workspaces = os.environ.get(
    "WORKSPACES",
    "/workspaces"
)

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

embeddings_model_name, embeddings = selection_embeddings()

# # Docker Model Runner:
# embeddings = OpenAIEmbeddings(
#     model="ai/embeddinggemma:300M-Q8_0",
#     base_url=openai_base_url,
#     api_key=api_key,
#     # disable check_embedding_ctx_length if your local model has different constraints
#     check_embedding_ctx_length=False,
# )

pdf_path = workspaces + "/playground-ai-ml/data/documents-pdf/Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splits = text_splitter.split_documents(pages)

persist_directory = workspaces + "/playground-ai-ml/data/chromadb/stocks24"
collection_name = "stock_market"

if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
    shutil.rmtree(persist_directory)

os.mkdir(persist_directory)

try:
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata={
            "embedding_class": embeddings_model_name
        }
    )
except Exception as e:
    print(f"Error setting up ChromaDB: {e}")
    raise
