import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

client = chromadb.EphemeralClient()
hb_time = client.heartbeat()
print(f"heartbeat: {hb_time}")

os.environ["HUGGINGFACE_HUB_CACHE"] = "/Users/asantiola/repo/playground-ai-ml/.cache"
embeddings_model = "thenlper/gte-small"
hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=embeddings_model,
)

collection_name="my_documents"
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

def query(question):
    question = "Tell me the interests of Lex."
    results = collection.query(
        query_texts=[question],
        n_results=5 # Retrieve top 5 most relevant chunks
    )

    print("\n--- Query Results ---")
    for _, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"Result:")
        print(f"  Source: {meta['source']}")
        print(f"  Chunk Index: {meta['chunk_index']}")
        print(f"  Content: {doc[:200]}...") # Print first 200 characters of the chunk
        print("-" * 30)

questions = "/Users/asantiola/repo/playground-ai-ml/data/questions.txt"
with open(questions) as file:
    for line in file:
        query(line.rstrip())
