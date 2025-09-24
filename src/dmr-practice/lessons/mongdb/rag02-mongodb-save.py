import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_mongodb import MongoDBAtlasVectorSearch

HOME=os.environ["HOME"]

embeddings_model = "ai/mxbai-embed-large"
embeddings_model_dimensions = 1024

oa_embeddings = OpenAIEmbeddings(
    model=embeddings_model,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
    # disable check_embedding_ctx_length if your local model has different constraints
    check_embedding_ctx_length=False,
)

doc_path = "/Users/asantiola/repo/playground-ai-ml/data/documents-txt"
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

client = MongoClient("mongodb://mongo_user:mongo_pass@localhost:27017/?directConnection=true")
print("client.server_info():", client.server_info(), "\n")

rag_db = "rag_dmr_db"
rag_collection = "rag_dmr_collection"
collection = client[rag_db][rag_collection]
index_name = "vector_index"

# Create your index model, then create the search index
index_name="vector_index"
search_index_model = SearchIndexModel(
  definition = {
    "fields": [
      {
        "type": "vector",
        "numDimensions": embeddings_model_dimensions,
        "path": "embedding",
        "similarity": "cosine"
      }
    ]
  },
  name = index_name,
  type = "vectorSearch"
)
collection.create_search_index(model=search_index_model)

vector_store = MongoDBAtlasVectorSearch.from_documents(
    documents=splits,
    collection=collection,
    embedding=oa_embeddings,
    index_name=index_name,
    relevance_score_fn="cosine",
)
