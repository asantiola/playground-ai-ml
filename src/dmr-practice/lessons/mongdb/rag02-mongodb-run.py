from langchain_openai.embeddings import OpenAIEmbeddings
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

embeddings_model = "ai/mxbai-embed-large"
embeddings_model_dimensions = 1024

oa_embeddings = OpenAIEmbeddings(
    model=embeddings_model,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
    # disable check_embedding_ctx_length if your local model has different constraints
    check_embedding_ctx_length=False,
)

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

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=oa_embeddings,
    index_name=index_name,
    relevance_score_fn="cosine",
)

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
        Basing on your training data, augmented by these documents, please answer.
        Use three sentences maximum and keep the answer concise:
        Question: {question} 
        Documents: {documents} 
        Answer: 
        """,
    input_variables=["question", "documents"],
)

llm = ChatOpenAI(
    model="ai/llama3.1",
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

rag_chain = prompt | llm
retriever = vector_store.as_retriever()

def query(question):
    print(f"Question: \"{question}\"\n")
    documents = retriever.invoke(question)
    answer = rag_chain.invoke({
        "documents": documents,
        "question": question,
    })
    print(f"Answer: {answer.content}\n\n")

questions = "/Users/asantiola/repo/playground-ai-ml/data/questions.txt"
with open(questions) as file:
    for line in file:
        query(line.rstrip())
