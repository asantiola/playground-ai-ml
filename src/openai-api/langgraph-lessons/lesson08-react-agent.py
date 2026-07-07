from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage     # the foundation class for all message types
from langchain_core.messages import ToolMessage     # passes data back to LLM after it calls a tool
from langchain_core.messages import SystemMessage   # message for providing instruction to LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import os

# # Annotated - provides additional context without affecting the type itself
# email = Annotated[str, "This has to be a valid email format"]
# print(email.__metadata__)

# # Sequence - to automatically handle the state updates for sequence such as adding new message to chat history

# # add_messages - a reducer function / aggregates
# # reducer - rule that controls how update from nodes are combined with the existing state
# # without reducer, updates would replace existing values entirely
# # Without a reducer:
# state = {"messages": ["Hi!"]}
# update = {"messages": ["Nice to meet you!"]}
# new_state = {"messages": ["Nice to meet you!"]}
# # With a reducer
# state = {"messages": ["Hi!"]}
# update = {"messages": ["Nice to meet you!"]}
# new_state = {"messages": ["Hi!", "Nice to meet you!"]}

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""

    return a + b

@tool
def substract(a: int, b: int):
    """This is a substraction function that substracts the second number from the first"""

    return a - b

@tool
def multiply(a: int, b: int):
    """This is a multipication function that multiplies 2 numbers together"""

    return a * b

tools = [add, substract, multiply]

llm = ChatOpenAI(
    model="mlx-community/gemma-4-12B-it-qat-6bit",
    base_url=openai_base_url,
    api_key=api_key,
)

llm_with_tools = llm.bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability"
    )
    response = llm_with_tools.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please")]}
print_stream(app.stream(inputs, stream_mode="values"))

# inputs = {"messages": [("user", "Add 3 + 4. Add 12 + 12.")]}
# print_stream(app.stream(inputs, stream_mode="values"))
