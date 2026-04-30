from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from pydantic import Field
from typing import Union, TypedDict, Annotated, Sequence, Callable
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
import operator

@tool
def get_current_date_time() -> str:
    """Returns the current date time in """

    return datetime.now().strftime("%Y-%B-%d %H:%M:%S")

@tool
def get_weather(city: str):
    """Returns the current weather for a specific city."""
    return f"The weather in {city} is sunny, 25C."

@tool
def lookup_stock_symbol(company_name: str) -> str:
    """
    Returns the stock symbol for a given company name.

    Args:
        company_name (str): The company name (e.g. 'Visa')
    
    Returns:
        str: The stock symbol (e.g. 'V') or an error message.
    """

    # Hard-coded mapping of company names to their NYSE symbols
    nyse_top_20 = {
        "Berkshire Hathaway": "BRK.B",
        "Walmart": "WMT",
        "JPMorgan Chase": "JPM",
        "Eli Lilly": "LLY",
        "Visa": "V",
        "Exxon Mobil": "XOM",
        "Johnson & Johnson": "JNJ",
        "Mastercard": "MA",
        "Chevron": "CVX",
        "Bank of America": "BAC",
        "UnitedHealth Group": "UNH",
        "The Home Depot": "HD",
        "Oracle": "ORCL",
        "Caterpillar": "CAT",
        "AbbVie": "ABBV",
        "Procter & Gamble": "PG",
        "Coca-Cola": "KO",
        "Morgan Stanley": "MS",
        "Goldman Sachs": "GS",
        "Salesforce": "CRM"
    }
    
    if company_name not in nyse_top_20:
        return f"Symbol not found for {company_name}"
    
    return nyse_top_20[company_name]

class StockQuote(TypedDict):
    sell_bid: float = Field(description="The Sell Bid price of a particular stock")
    buy_ask: float = Field(description="The Buy Ask price of a particular stock")

@tool
def get_stock_quotes(symbol: str) -> Union[StockQuote, str]:
    """
    Returns the quote for a given symbol.

    Args:
        symbol (str): The company symbol (e.g. 'V')
    
    Returns:
        a StockQuote on success or an error message.
    """

    nyse_top_19_less_oracle = {
        "BRK.B": {"sell_bid": 475.15, "buy_ask": 475.61},
        "WMT": {"sell_bid": 127.50, "buy_ask": 128.37},
        "JPM": {"sell_bid": 309.20, "buy_ask": 309.40},
        "LLY": {"sell_bid": 851.87, "buy_ask": 858.05},
        "V": {"sell_bid": 336.92, "buy_ask": 338.50},
        "XOM": {"sell_bid": 152.03, "buy_ask": 154.00},
        "JNJ": {"sell_bid": 224.90, "buy_ask": 227.60},
        "MA": {"sell_bid": 526.35, "buy_ask": 529.52},
        "CVX": {"sell_bid": 190.07, "buy_ask": 191.36},
        "BAC": {"sell_bid": 52.51, "buy_ask": 52.73},
        "UNH": {"sell_bid": 368.44, "buy_ask": 369.10},
        "HD": {"sell_bid": 320.83, "buy_ask": 321.45},
        "CAT": {"sell_bid": 812.07, "buy_ask": 818.68},
        "ABBV": {"sell_bid": 198.05, "buy_ask": 204.50},
        "PG": {"sell_bid": 146.28, "buy_ask": 146.38},
        "KO": {"sell_bid": 78.86, "buy_ask": 79.12},
        "MS": {"sell_bid": 186.09, "buy_ask": 187.25},
        "GS": {"sell_bid": 903.25, "buy_ask": 905.50},
        "CRM": {"sell_bid": 148.30, "buy_ask": 149.10}
    }

    if symbol not in nyse_top_19_less_oracle:
        return f"Quotes not found for symbol {symbol}"
    
    return nyse_top_19_less_oracle[symbol]

tools_financial = [lookup_stock_symbol, get_stock_quotes]
tools_general = [get_current_date_time, get_weather]
tools_all = [get_current_date_time, get_weather, lookup_stock_symbol, get_stock_quotes]

llm_general = ChatOpenAI(
    model="ai/gemma4:4B-128k",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
).bind_tools(tools_general)

llm_financial = ChatOpenAI(
    model="ai/gemma4:4B-128k",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
).bind_tools(tools_financial)

llm_default = ChatOpenAI(
    model="ai/gemma4:4B-128k",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
).bind_tools(tools_all)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

memory = MemorySaver()

@wrap_model_call
def model_middleware(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    user_msg = request.messages[-1].content.lower()
    if any(word in user_msg for word in ["date", "time", "weather"]):
        print("Select model dedicated for getting time and weather")
        return handler(request.override(model=llm_general))
    elif any(word in user_msg for word in ["stock", "bid", "ask", "financial", "price"]):
        print("Select model dedicated for financials")
        return handler(request.override(model=llm_financial))
    else:
        print("Select default model")
        return handler(request.override(model=llm_default))

@wrap_tool_call
def tool_middleWare(request: ToolCallRequest, handler):
    print(f"\n===== TOOL CALL =====\n{request}\n")
    response = handler(request)
    print(f"\n===== TOOL RESPONSE =====\n{response}\n")
    return response

agent = create_agent(
    model=None,
    tools=tools_all,
    system_prompt="You are a helpful assistant expert in Stocks. Use only the provided tools.",
    middleware=[model_middleware, tool_middleWare]
)

graph = StateGraph(AgentState)
graph.add_node("financial_agent", agent)
graph.add_edge(START, "financial_agent")
graph.add_edge("financial_agent", END)

app = graph.compile(checkpointer=memory)

thread_id = "session_001"

while True:
    user_input = input("\nWhat is your question: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    config = {"configurable": {"thread_id": thread_id}}
    messages = [HumanMessage(content=user_input)] 
    response = app.invoke({"messages": messages}, config=config)
    print(response["messages"][-1].content)
