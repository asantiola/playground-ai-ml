from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from typing import List
from langchain_core.embeddings import Embeddings
from mlx_embeddings import load, generate
import os
import shutil

def selection(what, choices_names, choices):
    print(f"Select a {what}:")
    for index, option in enumerate(choices_names, start=1):
        print(f"[{index}] {option}")

    while True:
        try:
            choice = int(input("\nEnter the number of your choice: "))

            if 1 <= choice <= len(choices):
                return choices_names[choice - 1], choices[choice - 1]
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

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
        self.model, self.tokenizer = load(model_id)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        formatted_texts = [f"task: retrieval-document | text: {text}" for text in texts]
        
        encoded = self.tokenizer.batch_encode_plus(
            formatted_texts, 
            return_tensors="mlx", 
            padding=True,
            truncation=True
        )
        
        outputs = self.model(
            encoded["input_ids"], 
            attention_mask=encoded.get("attention_mask")
        )
        
        return outputs.text_embeds.tolist()

    def embed_query(self, text: str) -> List[float]:
        formatted_query = f"task: retrieval-query | query: {text}"
        
        encoded = self.tokenizer.batch_encode_plus(
            [formatted_query], 
            return_tensors="mlx", 
            padding=True,
            truncation=True
        )
        
        outputs = self.model(
            encoded["input_ids"], 
            attention_mask=encoded.get("attention_mask")
        )
        
        return outputs.text_embeds.tolist()[0]

class MLXCompatibleEmbeddings(Embeddings):
    def __init__(self, model_id: str = "mlx-community/mxbai-embed-large-v1"):
        self.model, self.tokenizer = load(model_id)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        output = generate(self.model, self.tokenizer, texts=texts)
        return output.text_embeds.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def createMLXGemmaEmbeddings(model_id: str):
    return MLXGemmaEmbeddings(model_id)

def createMLXCompatibleEmbeddings(model_id: str):
    return MLXCompatibleEmbeddings(model_id)

embeddings_model_names = [
    "mlx-community/embeddinggemma-300m-4bit",
    "mlx-community/mxbai-embed-large-v1",
]
embeddings_creators = [
    createMLXGemmaEmbeddings,
    createMLXCompatibleEmbeddings,
]

embeddings_model_name, embeddings_creator = selection("embeddings", embeddings_model_names, embeddings_creators)
embeddings = embeddings_creator(embeddings_model_name)

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
