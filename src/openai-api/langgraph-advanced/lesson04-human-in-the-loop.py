from langchain.tools import tool
from typing import Union, TypedDict
from pydantic import Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import after_model
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
import os

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

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

@tool
def place_order(symbol: str, action: str, shares: int, limit_price: float, order_type: str = "limit") -> dict:
    """
    Execute a stock order.

    Args:
        symbol (str): The company symbol
        action (str): 'buy' or 'sell'
        shares (int): Number of shares to trade (pre-computed by the agent)
        limit_price (float): Limit price per share
        order_type (str): Order type, default "limit"
    
    Returns:
        status: Execution status (simulated)
        symbol
        shares
        limit_price
        total_spent
        type: Order type used
        action
    """

    total_spent = round(shares * limit_price, 2)
    return {
        "status": "filled",
        "symbol": symbol,
        "shares": shares,
        "limit_price": limit_price,
        "total_spent": total_spent,
        "type": order_type,
        "action": action
    }

llm = ChatOpenAI(
    model="mlx-community/gemma-4-12B-it-qat-6bit",
    base_url=openai_base_url,
    api_key=api_key,
    temperature=1.0,
    extra_body={
        "top_p": 0.95,
        "top_k": 64,
    },
)

RISKY_TOOLS = {"place_order"}

@after_model
def post_model_hook(state, _):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        for tc in last.tool_calls:
            if tc.get("name") in RISKY_TOOLS:
                decision = interrupt({"awaiting": tc["name"], "args": tc.get("args", {})})
                print(f"\n===== DECISION RECEIVED =====\n{decision}\n")

                if isinstance(decision, dict) and decision.get("approved"):
                    return {}
                
                tool_msg = ToolMessage(
                    content=f"Cancelled by human. Continue without executing the tool and provide the next steps.",
                    tool_call_id=tc["id"],
                    name=tc["name"] 
                )
                return {"messages": [tool_msg]}
    return {}

system_message = """
You are a financial advisor assistant. Use the provided tools to ground your answers in an up-to-date market data.
Be concise, factual, and risk-aware.

Be decisive: when you have sufficient information to act, proceed with tool calls without asking for confirmation.
Only if information is mising or uncertain, ask a concise clarifying question.

When preparing or describing actions, include appropriate parameters 
(e.g., symbol, shares, limit price, budgets) based on available data. Do not fabricate numbers or facts.

Inform if the order was fulfilled or cancelled.
"""

memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[lookup_stock_symbol, get_stock_quotes, place_order],
    system_prompt=system_message,
    middleware=[post_model_hook],
    checkpointer=memory,
)

# bdata = agent.get_graph().draw_mermaid_png()
# with open("diagram.png", "wb") as f:
#     f.write(bdata)

user_message = "Buy $1000 of Visa stock at the current price."

config = {"configurable": {"thread_id": "session_1"}}
response = agent.invoke({"messages": HumanMessage(user_message)}, config=config)
for message in response["messages"]:
    print(f"\nType: {message.type}:\n{message}")

state = agent.get_state(config)
print(f"\n\nstate.next: {state.next}")

print(f"\n\n'__interrupt__' in  response = {"__interrupt__" in response}")

interrupts = response["__interrupt__"]
for intr in interrupts:
    print(f"Interrupted: {intr.id}, {intr.value}")

def print_tool_approval(payload):
    tool = payload.get("awaiting", "unknown_tool")
    args = payload.get("args", {})

    print("----- Approval needed -----")
    print(f"Tool: {tool}")

    if isinstance(args, dict) and args:
        print("Parameters:")
        for k, v in args.items():
            print(f"parameter: {k}: {v}")
    elif args:
        print(f"Parameters: {args}")
    else:
        print("No parameters")

print_tool_approval(interrupts[0].value)

user_input = input("Do I proceed (Y/N)? ")
approved = user_input.lower() == "y"

final_state = agent.get_state(config=config)
if final_state.next:
    print("===== Resuming the Graph =====")
    response = agent.invoke(Command(resume={"approved": approved}), config=config)
    # for message in response["messages"]:
    #     print(f"\nType: {message.type}:\n{message}")

print(f"\n===== Final Response =====\n{response["messages"][-1].content}")
