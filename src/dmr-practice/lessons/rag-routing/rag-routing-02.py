from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

# practice rag selection using a selector agent

embeddings_model = "ai/mxbai-embed-large"
HOME=os.environ["HOME"]

oa_embeddings = OpenAIEmbeddings(
    model=embeddings_model,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
    # disable check_embedding_ctx_length if your local model has different constraints
    check_embedding_ctx_length=False,
)

def create_retriever(doc_path):
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

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=oa_embeddings,
    )

    return vector_store.as_retriever()

retriever_billiards = create_retriever(HOME + "/repo/playground-ai-ml/data/routing-txt/billiards")
retriever_guitars = create_retriever(HOME + "/repo/playground-ai-ml/data/routing-txt/guitars")
retriever_technologies = create_retriever(HOME + "/repo/playground-ai-ml/data/routing-txt/technologies")
retrievers = (None, retriever_billiards, retriever_guitars, retriever_technologies)

llm = ChatOpenAI(
    model="ai/gpt-oss",
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

selection_prompt = PromptTemplate(
    template="""You are helpful assistant that analyzes questions.
        You only return integers: 0, 1, 2, 3.
        If the question is not related to Lex, return 0.
        Else if the question is related to billiards, return 1.
        Else if the question is related to guitars, return 2.
        Else if the question is related to software engineering or programming, return 3.
        Otherwise, if it doesn't match any of these topics, return 0.
        Question: {question}
        """,
    input_variables=["question"]
)

chain = (selection_prompt | llm)

def agent_selector(question: str):
    response=chain.invoke({ "question": question})
    return retrievers[int(response.content)]

rag_prompt = PromptTemplate(
    template="""You are an assistant helping answer questions.
        Basing on your training data, augmented by the context, please answer.
        Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context}
        """,
    input_variables=["question", "context"],
)

def agent_expert(question: str):
    rag_chain = (
        { "context": agent_selector, "question": RunnablePassthrough() } | 
        rag_prompt | 
        llm
    )
    response = rag_chain.invoke(question)
    print(f"\nquestion: {question}\nanswer: {response.content}\n")

agent_expert("What are Lex's break cue?")
agent_expert("Where is Cebu City?")
agent_expert("How many guitars does Lex have?")
agent_expert("Is Lex a programmer?")
agent_expert("What is the largest bone in the human body?")
