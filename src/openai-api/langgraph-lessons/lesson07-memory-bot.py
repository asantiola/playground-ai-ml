from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import os

# Start -> Process_Node -> End

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/engines/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

llm = ChatOpenAI(
    model="ai/gemma4:E4B",
    base_url=openai_base_url,
    api_key=api_key,
)

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    # print(f"CURRENT STATE: {state["messages"]}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []
user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")
