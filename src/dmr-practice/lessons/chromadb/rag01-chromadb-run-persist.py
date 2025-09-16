import chromadb
from chromadb.utils import embedding_functions
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
