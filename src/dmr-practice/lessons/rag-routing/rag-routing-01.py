from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

# practice rag selection using a function checking for keywords

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

def select_retriever(question: str):
    """
    Selects a retriever based on keywords in the question.
    
    Args:
        question: The question that will be used to determine retriever.
    """

    if "lex" not in question.lower():
        return None
    elif "billiard" in question.lower() or "cue" in question.lower():
        return retriever_billiards
    elif "guitar" in  question.lower():
        return retriever_guitars

    return retriever_technologies

llm = ChatOpenAI(
    model="ai/gpt-oss",
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

rag_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
        Basing on your training data, augmented by the context, please answer.
        Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context}
        """,
    input_variables=["question", "context"],
)

def ask_expert(question: str):
    rag_chain = (
        { "context": select_retriever, "question": RunnablePassthrough() } | 
        rag_prompt | 
        llm
    )
    response = rag_chain.invoke(question)
    print(f"\nquestion: {question}\nanswer: {response.content}\n")

ask_expert("What are Lex's break cues?")
ask_expert("Where is Cebu City?")
ask_expert("How many guitars does Lex have?")
ask_expert("Is Lex a programmer?")
ask_expert("What is the largest bone in the human body?")
