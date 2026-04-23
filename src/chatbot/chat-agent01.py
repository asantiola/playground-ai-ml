from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 1. Define Dummy Tools
@tool
def get_weather(city: str):
    """Returns the current weather for a specific city."""
    return f"The weather in {city} is sunny, 25C."

@tool
def get_stock_price(ticker: str):
    """Returns the current stock price for a given ticker symbol."""
    return f"The stock price for {ticker} is $150.00."

tools = [get_weather, get_stock_price]

# 2. Initialize LLM (Local Endpoint)
# Point to Ollama's OpenAI-compatible port (default: 11434)
llm = ChatOpenAI(
    model="ai/gpt-oss:20B",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

# 3. Setup Memory and Orchestration
# MemorySaver allows the agent to track thread history
memory = MemorySaver()
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# 4. Chat Execution with Memory Persistence
def chat(query: str, thread_id: str):
    print(f"\nUser: {query}")
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the agent
    response = agent_executor.invoke(
        {"messages": [("user", query)]},
        config=config
    )
    
    # Extract the final AI message
    print(f"Agent: {response['messages'][-1].content}")

if __name__ == "__main__":
    # Unique ID to maintain conversation state
    THREAD = "session_001"

    # Query 1: Demonstrating multi-tool use in one go
    chat("What is the weather in Tokyo and the stock price for AAPL?", THREAD)
    
    # Query 2: Demonstrating memory (referring to Tokyo)
    chat("Which city did I just ask about?", THREAD)
