from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    number1: int
    number2: int
    operation: str
    finalNumber: int

def adder(state: AgentState) -> AgentState:
    """This node adds the 2 numbers"""

    state["finalNumber"] = state["number1"] + state["number2"]
    return state

def substracter(state: AgentState) -> AgentState:
    """This node substracts the 2 numbers"""

    state["finalNumber"] = state["number1"] - state["number2"]
    return state

def decide_next_node(state: AgentState) -> AgentState:
    """This node will select the next node of the graph"""

    if state["operation"] == "+":
        return "addition_operation"
    elif state["operation"] == "-":
        return "subtraction_operation"

graph = StateGraph(AgentState)
graph.add_node("add_node", adder)
graph.add_node("subtract_node", substracter)
graph.add_node("router", lambda state: state) # pass through function

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router", 
    decide_next_node, 
    {
        # edge: node
        "addition_operation": "add_node",
        "subtraction_operation": "subtract_node"
    }
)
graph.add_edge("add_node", END)
graph.add_edge("subtract_node", END)

app = graph.compile()

result = app.invoke({ "number1": 10, "number2": 4, "operation": "-"})
print(result)

result = app.invoke({ "number1": 2, "number2": 3, "operation": "+"})
print(result)
