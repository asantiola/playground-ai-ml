from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
import os
import chromadb

workspaces = os.environ.get(
    "WORKSPACES",
    "/workspaces"
)

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

openai_base_url_embedding = os.environ.get(
    "OPENAI_BASE_URL_EMBEDDING", 
    "http://localhost:12435/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

embeddings = OpenAIEmbeddings(
    model="unsloth/embeddinggemma-300m-GGUF:Q8_0",
    base_url=openai_base_url_embedding,
    api_key=api_key,
    # disable check_embedding_ctx_length if your local model has different constraints
    check_embedding_ctx_length=False,
)

persist_directory = os.path.join(workspaces, "playground-ai-ml/shared-data/python/chromadb/bibles")
collection_name = "kjv"

def auto_load_vector_store(persist_directory, collection_name):
    native_client = chromadb.PersistentClient(path=persist_directory)
    collection = native_client.get_collection(name=collection_name)
    model_class_name = collection.metadata.get("embedding_class")

    print(f"auto-detected embeddings model: {model_class_name}")
    # embeddings = get_embeddings(model_class_name)

    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

try:
    vector_store = auto_load_vector_store(persist_directory, collection_name)
except Exception as e:
    print(f"Error setting up ChromaDB: {e}")
    raise

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 16}, # K is the amount of chunks to return
)

llm = ChatOpenAI(
    model="unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL",
    base_url=openai_base_url,
    api_key=api_key,
    temperature=1.0,
    extra_body={
        "top_p": 0.95,
        "top_k": 64,
    },
)

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

system_message_template = SystemMessagePromptTemplate.from_template(
    "You are an intelligent AI assistant with access to Bible information.\n"
    "Answer the user's question using ONLY the provided Bible context below. "
    "If the answer cannot be found in the context, say that you don't know.\n\n"
    "--- CONTEXT ---\n{context}"
)

human_message_template = HumanMessagePromptTemplate.from_template("{question}")

prompt = ChatPromptTemplate.from_messages([
    system_message_template,
    human_message_template
])

rag_chain = (
    {
        "context": retriever | format_docs, 
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is in Leviticus 18?"):
    print(chunk, end="", flush=True)
print("\n")
