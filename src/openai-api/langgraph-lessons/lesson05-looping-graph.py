from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
import random

class AgentState(TypedDict):
    name: str
    number: List[int]
    counter: int

def greeting_node(state: AgentState) -> AgentState:
    """Greeting Node which says hi to the person"""

    state["name"] = f"Hi there, {state["name"]}"
    state["counter"] = 0
    
    return state

def random_node(state: AgentState) -> AgentState:
    """Generate a random number from 0 to 10"""

    state["number"].append(random.randint(0, 10))
    state["counter"] += 1

    return state

# like any software dev, there are multiple ways do this!
def should_continue(state: AgentState) -> AgentState:
    """Function to decide what to do next"""

    if state["counter"] < 5:
        print(f"Entering loop {state["counter"]}")
        return "loop" # continue loop
    else:
        return "exit"

graph = StateGraph(AgentState)

graph.add_node("greeting", greeting_node)
graph.add_node("random", random_node)

graph.add_edge(START, "greeting")
graph.add_edge("greeting", "random")
graph.add_conditional_edges(
    "random",               # source
    should_continue,        # action
    {                       
        "loop": "random",   # self-loop
        "exit": END         # end the graph
    }
)

app = graph.compile()
result = app.invoke({ "name": "Lex", "number": [], "counter": -6})
print(result)
