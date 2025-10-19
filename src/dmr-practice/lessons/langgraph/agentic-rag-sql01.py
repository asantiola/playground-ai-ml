import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Literal

HOME=os.environ["HOME"]

api_url = "http://localhost:12434/engines/v1"
api_key = "docker"
llm_model = "ai/gpt-oss:latest"
embeddings_model = "ai/mxbai-embed-large"

db_uri = "sqlite:///" + HOME + "/repo/playground-ai-ml/data/sql-agentic-ai.db"
db = SQLDatabase.from_uri(db_uri)

llm = ChatOpenAI(
    model=llm_model,
    temperature=0,
    base_url=api_url,
    api_key=api_key,
)

oa_embeddings = OpenAIEmbeddings(
    model=embeddings_model,
    base_url=api_url,
    api_key=api_key,
    # disable check_embedding_ctx_length if your local model has different constraints
    check_embedding_ctx_length=False,
)

# # Nodes:
# classify the input
# - query billiards vector store
# - query guitars vector store
# - query technologies vector store
# - query company sql db

db_billiards = HOME + "/repo/playground-ai-ml/data/billiards.db"
db_guitars = HOME + "/repo/playground-ai-ml/data/guitars.db"
db_technologies = HOME + "/repo/playground-ai-ml/data/technologies.db"

def get_retriever(oa_embeddings: OpenAIEmbeddings, store_path: str):
    vector_store = Chroma(
        embedding_function=oa_embeddings,
        persist_directory=store_path
    )
    return vector_store.as_retriever()

retriever_billiards = get_retriever(oa_embeddings, db_billiards)
retriever_guitars = get_retriever(oa_embeddings, db_guitars)
retriever_technologies = get_retriever(oa_embeddings, db_technologies)

class InputClassification(TypedDict):
    subject: Literal["lex_billiards", "lex_guitars", "lex_technologies", "albatross_company", "unknown"]

class State(TypedDict):
    input: str
    classification: InputClassification
    output: str

def classify_input(state: State):
    # print(f"query_rag state: {state}\n")
    prompt = PromptTemplate.from_template(
        """
        Analyze the input and classify it.
        Determines if input's subject is about:
        - 'lex_billiards': About Lex's billiards.
        - 'lex_guitars': About Lex's guitars.
        - 'lex_technologies': Lex's technologies
        - 'albatross_company': questions about Albatross company
        - 'unknown': neither about Lex or Albatross company
        Input: {input}
        """
    )
    structured_llm = llm.with_structured_output(schema=InputClassification)
    chain = prompt | structured_llm
    classification = chain.invoke({ "input": state["input"] })
    if classification["subject"] in ["lex_billiards", "lex_guitars", "lex_technologies"]:
        goto="query_rag"
    elif classification["subject"] == "albatross_company":
        goto="query_sql"
    else:
        goto="query_general"
    
    return Command(
        update={ "classification": classification },
        goto=goto
    )

def query_rag(state: State):
    # print(f"query_rag state: {state}\n")
    if state["classification"]["subject"] == "lex_billiards":
        retriever = retriever_billiards
    elif state["classification"]["subject"] == "lex_guitars":
        retriever = retriever_guitars
    elif state["classification"]["subject"] == "lex_technologies":
        retriever = retriever_technologies
    
    prompt = PromptTemplate.from_template(
        """
        You are a helpful AI assistant that answers questions based on the the documents provided.
        Documents: {documents}.
        Input: {input}
        """
    )
    chain = (
        { "documents": retriever, "input": RunnablePassthrough() } |
        prompt | 
        llm | 
        StrOutputParser()
    )
    response = chain.invoke(state["input"])
    return { "output": response }

sql_agent = create_sql_agent(llm=llm, db=db, agent_type="tool-calling", verbose=False)

def query_sql(state: State):
    # print(f"query_sql state: {state}\n")
    response = sql_agent.invoke({ "input": state["input"] })
    return { "output": response["output"] }

def query_general(state: State):
    # print(f"query_general state: {state}\n")
    prompt = PromptTemplate.from_template("You are a helpful AI assistant that answers questions. Input: {input}")
    chain = prompt | llm
    response = chain.invoke({ "input": state["input"] })
    return { "output": response.content }

graph_builder = StateGraph(State)
graph_builder.add_node("classify_input", classify_input)
graph_builder.add_node("query_rag", query_rag)
graph_builder.add_node("query_sql", query_sql)
graph_builder.add_node("query_general", query_general)
graph_builder.add_edge(START, "classify_input")
graph_builder.add_edge("query_rag", END)
graph_builder.add_edge("query_sql", END)
graph_builder.add_edge("query_general", END)

memory = MemorySaver()

config = {"configurable": {"thread_id": "my_conversation_1"}}
app = graph_builder.compile(checkpointer=memory)

question = "What break cues does Lex use?"
answer = app.invoke({ "input": question }, config=config)
print(f"question: {question}\nanswer:\n{answer["output"]}\n")

question = "Does Lex play the guitar?"
answer = app.invoke({ "input": question }, config=config)
print(f"question: {question}\nanswer:\n{answer["output"]}\n")

question = "What are the departments in Albatross company?"
answer = app.invoke({ "input": question }, config=config)
print(f"question: {question}\nanswer:\n{answer["output"]}\n")

question = "Is Lex an application programmer?"
answer = app.invoke({ "input": question }, config=config)
print(f"question: {question}\nanswer:\n{answer["output"]}\n")

question = "Among the employees of Albatross that have contact information, who has the highest salary and where does he lives?"
answer = app.invoke({ "input": question }, config=config)
print(f"question: {question}\nanswer:\n{answer["output"]}\n")

question = "What is the largest bone in the human body?"
answer = app.invoke({ "input": question }, config=config)
print(f"question: {question}\nanswer:\n{answer["output"]}\n")


# # sample output:
# question: What break cues does Lex use?
# answer:
# Lex uses an **Action ACT 56 Break and Jump cue**.

# question: Does Lex play the guitar?
# answer:
# Yes. The documents state that Lex studied classical guitar and used several guitars—including a Yamaha junior nylon guitar and an acoustic guitar—so he does play the guitar.

# question: What are the departments in Albatross company?
# answer:
# The departments in Albatross company are:
# - IT  
# - HR  
# - Marketing  
# - Finance

# question: Is Lex an application programmer?
# answer:
# Yes.  
# All of the documents list a wide range of programming languages (C++, Python, Golang, Java, C, C#, Perl, SQL, Visual Basic, JavaScript, Angular) and application‑level frameworks/architectures (Spring, Spring Boot, REST API, Web services, Microservices, CORBA). These are the typical tools used by an **application programmer**—someone who writes, designs, and maintains software that runs on end‑user machines or servers. Therefore, based on the information provided, Lex is indeed an application programmer.

# question: Among the employees of Albatross that have contact information, who has the highest salary and where does he lives?
# answer:
# The employee with the highest salary who also has contact information is **Janice**, and she lives at **Ayala, Makati**.

# question: What is the largest bone in the human body?
# answer:
# The largest bone in the human body is the **femur** (thigh bone). It extends from the hip joint to the knee and is the strongest and longest bone, supporting the weight of the body and enabling movement of the lower limbs.
