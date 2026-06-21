from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_message
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.tools import tool
from practice13_common import get_embeddings
import os
import chromadb

workspaces = os.environ.get(
    "WORKSPACES",
    "/workspaces"
)

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

persist_directory = os.path.join(workspaces, "playground-ai-ml/data/chromadb/bibles")
collection_name = "kjv"

def auto_load_vector_store(persist_directory, collection_name):
    native_client = chromadb.PersistentClient(path=persist_directory)
    collection = native_client.get_collection(name=collection_name)
    model_class_name = collection.metadata.get("embedding_class")

    print(f"auto-detected embeddings model: {model_class_name}")
    embeddings = get_embeddings(model_class_name)

    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

try:
    vector_store = auto_load_vector_store(persist_directory, collection_name)
except Exception as e:
    print(f"Error setting up ChromaDB: {e}")
    raise

@tool
def retriever_tool(query: str, book: str = None, chapter: int = None):
    """Searches the KJV Bible. Use optional book and chapter filters for exact verse lookups.

    Args:
        query (str): The search query or keywords to look for in the Bible.
        book (str): Optional Book in the Bible.
        chapter (int): Optional Chapter in the Book in the Bible.
    
    Returns:
        str: The info obtained from RAG.
    """
    
    search_kwargs = {"k": 8}
    
    # If the LLM specifies a book/chapter, use absolute metadata filtering
    if book or chapter:
        filter_dict = {}
        if book:
            filter_dict["book"] = book.strip().capitalize()
        if chapter:
            filter_dict["chapter"] = int(chapter)
            
        search_kwargs["where"] = filter_dict

    local_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    docs = local_retriever.invoke(query)

    if not docs:
        return "No relevant verses found matching those criteria."
        
    return "\n".join(doc.page_content for doc in docs)

llm = ChatOpenAI(
    model="mlx-community/gemma-4-12B-it-6bit",
    base_url=openai_base_url,
    api_key=api_key,
    temperature=0,
)

tools = [retriever_tool]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_message]

def should_continue(state: AgentState) -> str:
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about King James Version of the Bible.
Use the retriever tool available to answer questions about the bible. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} # dictionary of our tools

# LLM Agent
def call_llm(state: AgentState):
    """Function to call the LLM with the current state."""

    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm_with_tools.invoke(messages)
    return {"messages": [message]}

# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            # print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            # print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    # print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END,
    }
)
graph.add_edge("retriever_agent", "llm")

memory = MemorySaver()
rag_agent = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "session_1"}}

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['/exit', '/quit', '/bye']:
            break

        if user_input.lower() == '/clear':
            active_id = config["configurable"]["thread_id"]
            memory.storage.pop(active_id, None)
            print("\n[System: Memory safely wiped from RAM! Starting fresh.]")
            continue

        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages}, config=config)
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

running_agent()
