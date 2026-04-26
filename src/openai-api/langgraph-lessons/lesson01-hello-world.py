from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict): # our state schema
    message: str

def greeting_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting message to the state"""

    state["message"] = "Hey " + state["message"] + ", how are you?"
    return state

graph = StateGraph(AgentState)
graph.add_node("greeter", greeting_node)
graph.set_entry_point("greeter")
graph.set_finish_point("greeter")

app = graph.compile()
result = app.invoke({"message": "Lex"})
print(result["message"])
