from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from practice13_common import get_embeddings
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

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

persist_directory = os.path.join(workspaces, "playground-ai-ml/data/chromadb/bibles")
collection_name = "kjv"

def auto_load_vector_store(persist_directory, collection_name):
    native_client = chromadb.PersistentClient(path=persist_directory)
    collection = native_client.get_collection(name=collection_name)
    model_class_name = collection.metadata.get("embedding_class")

    print(f"auto-detected embeddings model: {model_class_name}")
    embeddings = get_embeddings(model_class_name)

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
    search_kwargs={"k": 5}, # K is the amount of chunks to return
)

llm = ChatOpenAI(
    model="mlx-community/gemma-4-12B-it-6bit",
    base_url=openai_base_url,
    api_key=api_key,
    temperature=0,
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

rewrite_prompt = ChatPromptTemplate.from_template(
    "Extract only the core search keywords or question from this user request. "
    "Strip out formatting requests or commands like 'read me', 'print', or 'bullet points'.\n\n"
    "Request: {raw_question}\n"
    "Search Query:"
)

query_rewriter = rewrite_prompt | llm | StrOutputParser()

rag_chain = (
    {
        "context": (lambda x: query_rewriter.invoke({"raw_question": x})) | retriever | format_docs, 
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("Did Satan and God talk to each other?"):
    print(chunk, end="", flush=True)
print("\n")
