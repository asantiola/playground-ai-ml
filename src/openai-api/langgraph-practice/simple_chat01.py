from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

#  https://www.datacamp.com/tutorial/langgraph-tutorial

class State(TypedDict):
    # messages have the type "list".
    # the add_messages function appends messages to the list,, rather than overwriting
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatOpenAI(
    model="ai/gpt-oss:20B", 
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

def chatbot(state: State):
    return { "messages": [llm.invoke(state["messages"])] }

# 1st argument is unique node name
# 2nd argument is the function or object to be called whenever the node is used
graph_builder.add_node("chatbot", chatbot)

graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

memory = MemorySaver()
app = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "my_conversation_1"}}

while True:
    user_input = input("User: ")
    if (user_input.lower() in ["quit", "exit", "q"]):
        print("Bye")
        break

    for event in app.stream({ "messages": [("user", user_input)]}, config=config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
