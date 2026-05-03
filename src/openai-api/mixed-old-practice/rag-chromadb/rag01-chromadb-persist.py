from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

HOME=os.environ["HOME"]
embeddings_model = "thenlper/gte-small"

# Warning seen: huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
# - Avoid using `tokenizers` before the fork if possible
# - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
os.environ["TOKENIZERS_PARALLELISM"] = ""

hf_embeddings = HuggingFaceEmbeddings(
    cache_folder=HOME + "/repo/playground-ai-ml/.cache",
    model_name=embeddings_model,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

doc_path = HOME + "/repo/playground-ai-ml/data/documents-txt"
documents = []
for filename in os.listdir(doc_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(doc_path, filename)
        loader = TextLoader(file_path)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
splits = text_splitter.split_documents(documents=documents)

chromadb_path=HOME + "/repo/playground-ai-ml/.chromadb"
vector_store = Chroma.from_documents(
    documents=splits,
    embedding=hf_embeddings,
    persist_directory=chromadb_path,
)
