import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import redis

embeddings_model = "thenlper/gte-small"

doc_path = "/Users/asantiola/repo/playground-ai-ml/data/documents-txt"
files = [os.path.join(doc_path, file) for file in os.listdir(doc_path)]
docs_lists = [TextLoader(file).load_and_split() for file in files]

hf_embeddings = HuggingFaceEmbeddings(
    cache_folder="/Users/asantiola/repo/playground-ai-ml/.cache",
    model_name=embeddings_model,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

redis_url = "redis://default:password@localhost:6379"
index_name = "vector_index"
config = RedisConfig(
    index_name=index_name,
    redis_url=redis_url,
)

client = redis.from_url(redis_url)
print("client.ping(): ", client.ping())

client.flushall()

vector_store = RedisVectorStore(
    embeddings=hf_embeddings, 
    config=config
)

for docs_list in docs_lists:
    vector_store.add_documents(docs_list)

client.save()
client.quit()
