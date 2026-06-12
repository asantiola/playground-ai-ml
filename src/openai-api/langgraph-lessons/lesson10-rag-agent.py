from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_message
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.tools import tool
from typing import List
from langchain_core.embeddings import Embeddings
from mlx_embeddings.utils import load as load_mlx_embedding
import os

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

class MLXGemmaEmbeddings(Embeddings):
    def __init__(self, model_id: str = "mlx-community/embeddinggemma-300m-4bit"):
        # This handles the custom encoder layers natively on your Apple Silicon GPU
        self.model, self.tokenizer = load_mlx_embedding(model_id)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Process the list of chunks coming from your text splitter
        embeddings_list = []
        for text in texts:
            input_ids = self.tokenizer.encode(text, return_tensors="mlx")
            outputs = self.model(input_ids)
            
            # Extract the mean-pooled, normalized embedding vectors
            text_embeds = outputs.text_embeds.tolist()
            embeddings_list.extend(text_embeds)
            
        return embeddings_list

    def embed_query(self, text: str) -> List[float]:
        # Process individual user search queries
        return self.embed_documents([text])[0]

# Replace your commented out block with this instantiation:
embeddings = MLXGemmaEmbeddings()

persist_directory = workspaces + "/playground-ai-ml/.chromadb"
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
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
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
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
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

rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()
