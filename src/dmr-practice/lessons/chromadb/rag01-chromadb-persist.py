import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

client = chromadb.PersistentClient(
    path="/Users/asantiola/repo/playground-ai-ml/.chromadb",
)
hb_time = client.heartbeat()
print(f"heartbeat: {hb_time}")

os.environ["HUGGINGFACE_HUB_CACHE"] = "/Users/asantiola/repo/playground-ai-ml/.cache"
embeddings_model = "thenlper/gte-small"
hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=embeddings_model,
)

collection_name="persisted_documents"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=hf_ef,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

doc_path = "/Users/asantiola/repo/playground-ai-ml/data/documents-txt"
files = [os.path.join(doc_path, file) for file in os.listdir(doc_path)]

documents = []
metadatas = []
ids = []
current_id = 0

for filename in os.listdir(doc_path):
    if filename.endswith(".txt"):
        filepath = os.path.join(doc_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Chunk the text
        chunks = text_splitter.split_text(raw_text)

        # Prepare data for ChromaDB
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})
            ids.append(f"{filename.replace('.txt', '')}_chunk_{current_id}")
            current_id += 1

if documents:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Added {len(documents)} chunks to ChromaDB collection '{collection_name}'.")
else:
    print("No text files found or no chunks generated.")
