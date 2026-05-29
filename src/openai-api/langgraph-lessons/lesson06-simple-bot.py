from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import os

# Start -> Process_Node -> End

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://model-runner.docker.internal/engines/v1"
)

llm = ChatOpenAI(
    model="ai/gemma4:E4B",
    base_url=openai_base_url,
    api_key="docker",
)

class AgentState(TypedDict):
    messages: List[HumanMessage]

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
