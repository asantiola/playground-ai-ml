from langchain_core.documents import Document
from langchain_chroma import Chroma
from practice13_common import selection_embeddings
import os
import json
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


book_mapping = {
    # Old Testament (1-39)
    1: "Genesis", 2: "Exodus", 3: "Leviticus", 4: "Numbers", 5: "Deuteronomy",
    6: "Joshua", 7: "Judges", 8: "Ruth", 9: "1_Samuel", 10: "2_Samuel",
    11: "1_Kings", 12: "2_Kings", 13: "1_Chronicles", 14: "2_Chronicles",
    15: "Ezra", 16: "Nehemiah", 17: "Esther", 18: "Job", 19: "Psalms",
    20: "Proverbs", 21: "Ecclesiastes", 22: "Song_of_Solomon", 23: "Isaiah",
    24: "Jeremiah", 25: "Lamentations", 26: "Ezekiel", 27: "Daniel",
    28: "Hosea", 29: "Joel", 30: "Amos", 31: "Obadiah", 32: "Jonah",
    33: "Micah", 34: "Nahum", 35: "Habakkuk", 36: "Zephaniah", 37: "Haggai",
    38: "Zechariah", 39: "Malachi",

    # New Testament (40-66)
    40: "Matthew", 41: "Mark", 42: "Luke", 43: "John", 44: "Acts",
    45: "Romans", 46: "1_Corinthians", 47: "2_Corinthians", 48: "Galatians",
    49: "Ephesians", 50: "Philippians", 51: "Colossians", 52: "1_Thessalonians",
    53: "2_Thessalonians", 54: "1_Timothy", 55: "2_Timothy", 56: "Titus",
    57: "Philemon", 58: "Hebrews", 59: "James", 60: "1_Peter", 61: "2_Peter",
    62: "1_John", 63: "2_John", 64: "3_John", 65: "Jude", 66: "Revelation"
}

json_file = os.path.join(workspaces, "playground-ai-ml/data/kjv.json")
persist_directory = os.path.join(workspaces, "playground-ai-ml/data/chromadb/bibles")
collection_name = "kjv"

if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
    shutil.rmtree(persist_directory)
os.makedirs(persist_directory, exist_ok=True)

chroma_metadata = {
    "embedding_class": embeddings_model_name,
}

vector_store = None

print("Parsing JSON and initializing vector store in batches...")

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = data["verses"]["row"]

BATCH_SIZE = 500
current_batch = []

for i, row in enumerate(rows):
    fields = row["field"]
    book_num = int(fields[1])
    book_name = book_mapping.get(book_num, f"Unknown_{book_num}")
    
    page_content = f"{book_name} {fields[2]}:{fields[3]} - {fields[4]}"
    metadata = {
        "verse_id": int(fields[0]),
        "book_num": book_num,
        "book_name": str(book_name),
        "chapter": int(fields[2]),
        "verse": int(fields[3])
    }
    
    current_batch.append(Document(page_content=page_content, metadata=metadata))
    
    if len(current_batch) >= BATCH_SIZE:
        if vector_store is None:
            # Initialize store with the first batch
            vector_store = Chroma.from_documents(
                documents=current_batch,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name,
                collection_metadata=chroma_metadata
            )
        else:
            # Append subsequent batches
            vector_store.add_documents(documents=current_batch)
            
        print(f"Indexed up to verse {i + 1}...")
        current_batch = []

if current_batch:
    if vector_store is None:
        vector_store = Chroma.from_documents(
            documents=current_batch,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
            collection_metadata=chroma_metadata
        )
    else:
        vector_store.add_documents(documents=current_batch)

print("Vector store creation complete.")
