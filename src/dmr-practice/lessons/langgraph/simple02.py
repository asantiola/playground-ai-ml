from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
import operator

# variations for simple01, with memory!

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

llm = ChatOpenAI(
    model="ai/gpt-oss:latest", 
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

def chatbot(state: State):
    messages = state["messages"]
    response = llm.invoke(messages)
    return { "messages": [response] }


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
# graph_builder.set_entry_point("chatbot") #alternative below
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

memory = MemorySaver()
app = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "my_conversation_1"}}

initial_messages = [
    SystemMessage(content="You are Mario from Super Mario Brothers that helps answer questions."),
    HumanMessage(content="What is the capital of France?"),
]

result = app.invoke({ "messages": initial_messages }, config=config)
print(f"result:\n{result}\n")
print(f"answer:\n{result["messages"][-1].content}\n")

# # Get user input
# user_input = input("User: ")
# input_message = [HumanMessage(content=user_input)]

input_message = [
    HumanMessage(content="From which continent is it?")
]
result = app.invoke({ "messages": input_message }, config=config)
print(f"result:\n{result}\n")
print(f"answer:\n{result["messages"][-1].content}\n")

input_message = [
    HumanMessage(content="What are the countries around it?")
]
result = app.invoke({ "messages": input_message }, config=config)
print(f"result:\n{result}\n")
print(f"answer:\n{result["messages"][-1].content}\n")
