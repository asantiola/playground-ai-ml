import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Literal

# use OOP for agentic-rag-sql02.py, create vector store on the fly

HOME=os.environ["HOME"]

# # Nodes:
# classify the input
# - query billiards vector store
# - query guitars vector store
# - query technologies vector store
# - query company sql db

class InputClassification(TypedDict):
    subject: Literal["lex_billiards", "lex_guitars", "lex_technologies", "albatross_company", "unknown"]

class State(TypedDict):
    input: str
    classification: InputClassification
    output: str

class QueryAgents:
    def __init__(self, config: dict):
        self.is_valid = False

        db_uri = config["db_uri"]

        if not db_uri:
            return
        
        db = SQLDatabase.from_uri(db_uri)

        llm_model = config["llm_model"]
        api_url = config["api_url"]
        api_key = config["api_key"]

        if not llm_model or not api_key:
            return

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            base_url=api_url,
            api_key=api_key,
        )

        embeddings_model = config["embeddings_model"]

        if not embeddings_model:
            return

        self.oa_embeddings = OpenAIEmbeddings(
            model=embeddings_model,
            base_url=api_url,
            api_key=api_key,
            # disable check_embedding_ctx_length if your local model has different constraints
            check_embedding_ctx_length=False,
        )

        doc_billiards = config["doc_billiards"]
        doc_guitars = config["doc_guitars"]
        doc_technologies = config["doc_technologies"]

        if not doc_billiards or not doc_guitars or not doc_technologies:
            return

        self.retriever_billiards = self.get_retriever(doc_billiards)
        self.retriever_guitars = self.get_retriever(doc_guitars)
        self.retriever_technologies = self.get_retriever(doc_technologies)

        self.sql_agent = create_sql_agent(llm=self.llm, db=db, agent_type="tool-calling", verbose=False)

        self.is_valid = True

    def get_retriever(self, doc_path: str):
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
            embedding=self.oa_embeddings,
        )
        return vector_store.as_retriever()

    def classify_input(self, state: State):
        if not self.is_valid:
            return
        
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
        structured_llm = self.llm.with_structured_output(schema=InputClassification)
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

    def query_rag(self, state: State):
        if not self.is_valid:
            return
        
        # print(f"query_rag state: {state}\n")
        if state["classification"]["subject"] == "lex_billiards":
            retriever = self.retriever_billiards
        elif state["classification"]["subject"] == "lex_guitars":
            retriever = self.retriever_guitars
        elif state["classification"]["subject"] == "lex_technologies":
            retriever = self.retriever_technologies
        
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
            self.llm | 
            StrOutputParser()
        )
        response = chain.invoke(state["input"])
        return { "output": response }

    def query_sql(self, state: State):
        if not self.is_valid:
            return
        
        # print(f"query_sql state: {state}\n")
        response = self.sql_agent.invoke({ "input": state["input"] })
        return { "output": response["output"] }

    def query_general(self, state: State):
        if not self.is_valid:
            return
        
        # print(f"query_general state: {state}\n")
        prompt = PromptTemplate.from_template("You are a helpful AI assistant that answers questions. Input: {input}")
        chain = prompt | self.llm
        response = chain.invoke({ "input": state["input"] })
        return { "output": response.content }

query_agents = QueryAgents(
    {
        "api_url": "http://localhost:12434/engines/v1",
        "api_key": "docker",
        "llm_model": "ai/gpt-oss:latest",
        "embeddings_model": "ai/mxbai-embed-large",
        "db_uri": "sqlite:///" + HOME + "/repo/playground-ai-ml/data/sql-agentic-ai.db",
        "doc_billiards": HOME + "/repo/playground-ai-ml/data/routing-txt/billiards/",
        "doc_guitars": HOME + "/repo/playground-ai-ml/data/routing-txt/guitars/",
        "doc_technologies": HOME + "/repo/playground-ai-ml/data/routing-txt/technologies/",
    }
)

graph_builder = StateGraph(State)
graph_builder.add_node("classify_input", query_agents.classify_input)
graph_builder.add_node("query_rag", query_agents.query_rag)
graph_builder.add_node("query_sql", query_agents.query_sql)
graph_builder.add_node("query_general", query_agents.query_general)
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

question = "Can Lex do application programming?"
answer = app.invoke({ "input": question }, config=config)
print(f"question: {question}\nanswer:\n{answer["output"]}\n")

question = "Among Albatross employees with contact information, who has the highest salary and tell me all the details?"
answer = app.invoke({ "input": question }, config=config)
print(f"question: {question}\nanswer:\n{answer["output"]}\n")

question = "What is the largest bone in the human body?"
answer = app.invoke({ "input": question }, config=config)
print(f"question: {question}\nanswer:\n{answer["output"]}\n")
