from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from datetime import datetime
import os
import operator

llm = ChatOpenAI(
    model="ai/gemma4:4B-128k",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

embeddings = OpenAIEmbeddings(
    model="ai/embeddinggemma:300M-Q8_0",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
    # disable check_embedding_ctx_length if your local model has different constraints
    check_embedding_ctx_length=False,
)

@before_model
def pre_model_hook(state: dict, config: any) -> dict:
    print(f"\n===== BEFORE LLM CALL =====\n{state}\n")
    return state

@after_model
def post_model_hook(state: dict, config: dict) -> dict:
    print(f"\n===== AFTER LLM CALL =====\n{state}\n")
    return state

HOME = os.environ["HOME"]
persist_directory = HOME + "/repo/playground-ai-ml/.chromadb"
collection_name = "stock_market"

try:
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
except Exception as e:
    print(f"Error setting up ChromaDB: {e}")
    raise

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}, # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from the Stock Market Performance 2024 document."""

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    
    results = []

    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

rag_tools = [retriever_tool]
llm_rag = llm.bind_tools(rag_tools)

rag_system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

retriever_agent = create_agent(
    model=llm_rag,
    tools=rag_tools,
    system_prompt=rag_system_prompt,
    # middleware=[pre_model_hook, post_model_hook],
)
# response = retriever_agent.invoke({"messages": [HumanMessage(content="How did Meta perform in 2024?")]})
# print(f"===== AI RESPONSE =====\n{response["messages"][-1].content}\n")

@tool
def get_current_date_time() -> str:
    """Returns the current date time"""

    return datetime.now()

date_time_tools = [get_current_date_time]
llm_date_time = llm.bind_tools(date_time_tools)
date_time_system_prompt="""
You are a helpful AI assistant that gets date and time. 
Use only the provided tools.
Always respond with both date and time information.
"""

date_time_agent = create_agent(
    model=llm_date_time,
    tools=date_time_tools,
    system_prompt=date_time_system_prompt,
    # middleware=[pre_model_hook, post_model_hook],
)

class DateTimeResponse(BaseModel):
    year: int = Field(description="The year")
    month: int = Field(description="The month")
    day: int = Field(description="The day")
    hour: int = Field(description="The hour")
    min: int = Field(description="The minute")
    sec: int = Field(description="The second")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    date_time: DateTimeResponse

llm_formatter = llm.with_structured_output(schema=DateTimeResponse)
def formatter_agent(state: AgentState) -> AgentState:
    human_message = HumanMessage(content=state["messages"][-1].content)
    messages = [SystemMessage(content="Extract the DateTimeResponse"), human_message]
    state["date_time"] = llm_formatter.invoke(messages)
    return state

# unformatted_state = date_time_agent.invoke({"messages": [HumanMessage(content="What date is it today?")]})
# print(f"===== AI RESPONSE =====\n{unformatted_state["messages"][-1].content}\n")
# formatted_state = formatter_agent(unformatted_state)
# print(f"===== AI RESPONSE =====\n{formatted_state["date_time"]}\n")

def general_agent(state: AgentState) -> AgentState:
    system_prompt="You are a helpful assistant. Answer the question to the best of your ability."
    messages = [SystemMessage(content=system_prompt), state["messages"][-1]]
    response = llm.invoke(messages)
    return {"messages": [response]}

def print_last_node(state: AgentState) -> AgentState:
    if "date_time" in state and state["date_time"] is not None:
        print(f"===== AI response =====\n{state["date_time"]}\n")
    else:
        print(f"===== AI response =====\n{state["messages"][-1].content}")
    return state

def asker_node(state: AgentState) -> Command:
    system_prompt="""
    Analyze the input and classify it.
    If input is about Stock Market Performance 2024, return 'stock2024'
    If input is about retrieving date/time, return 'datetime'
    if input is one of the following: 'exit', 'quit', or 'bye', return 'exit'
    Otherwise, return 'general'
    """
    prompt = input("\nWhat is your question: ")
    human_message = HumanMessage(content=prompt)
    messages = [SystemMessage(content=system_prompt), human_message]
    response = llm.invoke(messages)
    classification = response.content
    # print(f"===== CLASSIFICATION =====\n{classification}\n")

    if classification == "stock2024":
        return Command(
            update={ "messages": state["messages"] + [human_message], "date_time": None},
            goto="retriever_agent",
        )
    elif classification == "datetime":
        return Command(
            update={ "messages": state["messages"] + [human_message], "date_time": None},
            goto="date_time_agent",
        )
    elif classification == "exit":
        return Command(
            goto=END
        )
    else:
        return Command(
            update={ "messages": state["messages"] + [human_message], "date_time": None},
            goto="general_agent",
        )

graph = StateGraph(AgentState)

graph.add_node("asker_node", asker_node)
graph.add_node("retriever_agent", retriever_agent)
graph.add_node("date_time_agent", date_time_agent)
graph.add_node("formatter_agent", formatter_agent)
graph.add_node("general_agent", general_agent)
graph.add_node("print_last_node", print_last_node)

graph.add_edge(START, "asker_node")

graph.add_edge("date_time_agent", "formatter_agent")
graph.add_edge("formatter_agent", "print_last_node")

graph.add_edge("retriever_agent", "print_last_node")

graph.add_edge("general_agent", "print_last_node")

graph.add_edge("print_last_node", "asker_node")

memory = MemorySaver()
app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "session01"}}
app.invoke({}, config=config)
