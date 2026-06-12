from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from typing import List
from langchain_core.embeddings import Embeddings
from mlx_embeddings.utils import load as load_mlx_embedding
import os

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

class MLXGemmaEmbeddings(Embeddings):
    def __init__(self, model_id: str = "mlx-community/embeddinggemma-300m-4bit"):
        # This handles the custom encoder layers natively on your Apple Silicon GPU
        self.model, self.tokenizer = load_mlx_embedding(model_id)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Process the list of chunks coming from your text splitter
        embeddings_list = []
        for text in texts:
            input_ids = self.tokenizer.encode(text, return_tensors="mlx")
            outputs = self.model(input_ids)
            
            # Extract the mean-pooled, normalized embedding vectors
            text_embeds = outputs.text_embeds.tolist()
            embeddings_list.extend(text_embeds)
            
        return embeddings_list

    def embed_query(self, text: str) -> List[float]:
        # Process individual user search queries
        return self.embed_documents([text])[0]

# Replace your commented out block with this instantiation:
embeddings = MLXGemmaEmbeddings()

# # Docker Model Runner:
# embeddings = OpenAIEmbeddings(
#     model="ai/embeddinggemma:300M-Q8_0",
#     base_url=openai_base_url,
#     api_key=api_key,
#     # disable check_embedding_ctx_length if your local model has different constraints
#     check_embedding_ctx_length=False,
# )

pdf_path = workspaces + "/playground-ai-ml/data/Stock_Market_Performance_2024.pdf"

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

persist_directory = workspaces + "/playground-ai-ml/.chromadb"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.mkdir(persist_directory)

try:
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
except Exception as e:
    print(f"Error setting up ChromaDB: {e}")
    raise
