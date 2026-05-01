from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from datetime import datetime
from typing import TypedDict, Union

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
        Union[StockQuote, str] on success or an error message.
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

tools = [get_current_date_time, lookup_stock_symbol, get_stock_quotes]

class FinancialInfo(BaseModel):
    symbol: str = Field(
        description="The symbol of the stock"
    )
    sell_bid: float = Field(
        None,
        description="The Sell Bid price of a particular stock if present, otherwise return null",
    )
    buy_ask: float = Field(
        None,
        description="The Buy Ask price of a particular stock if present, other wise return null",
    )
    confidence_score: float = Field(
        description="A number between 0 and 1, inclusive, describing the confidence on the response"
    )
    confidence_reason: str = Field(
        description="Explanation for the confidence score"
    )

llm_with_tools = ChatOpenAI(
    model="ai/gemma4:4B-128k",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
).bind_tools(tools)

agent = create_agent(
    model=llm_with_tools,
    tools=tools,
    system_prompt="You are a helpful assistant expert in Stocks. Use only the provided tools.",
    response_format=FinancialInfo
)

response = agent.invoke({"messages": [HumanMessage(content="Give me the financial info for Visa.")]})
structured_response : FinancialInfo = response["structured_response"]
print(structured_response)

response = agent.invoke({"messages": [HumanMessage(content="Give me the financial info for Oracle.")]})
structured_response : FinancialInfo = response["structured_response"]
print(structured_response)

