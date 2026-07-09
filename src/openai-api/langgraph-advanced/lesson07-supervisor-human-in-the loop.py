from langchain.tools import tool
from typing import Union, TypedDict
from pydantic import Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import after_model
from langgraph.types import interrupt, Command
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
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

financial_agent_prompt = """
You are a financial advisor assistant. Use the provided tools to ground your answers in an up-to-date market data.
Be concise, factual, and risk-aware.

Do not handle questions not related to financial matters.

Be decisive: when you have sufficient information to act, proceed with tool calls without asking for confirmation.
Only if information is mising or uncertain, ask a concise clarifying question.

When preparing or describing actions, include appropriate parameters 
(e.g., symbol, shares, limit price, budgets) based on available data. Do not fabricate numbers or facts.

Inform if the order was fulfilled or cancelled.
"""

financial_agent = create_agent(
    model=llm,
    tools=[lookup_stock_symbol, get_stock_quotes, place_order],
    system_prompt=financial_agent_prompt,
    middleware=[post_model_hook],
    name="financial_agent",
)

# Local dummy database
location_db = {
    "france": {"capital": "Paris", "landmark": "Eiffel Tower"},
    "japan": {"capital": "Tokyo", "landmark": "Mount Fuji"},
    "egypt": {"capital": "Cairo", "landmark": "Pyramids of Giza"},
    "brazil": {"capital": "Brasília", "landmark": "Christ the Redeemer"},
    "australia": {"capital": "Canberra", "landmark": "Sydney Opera House"}
}

@tool
def get_country_capital(country: str) -> str:
    """
    Gets the capital a given country.

    Args:
        country (str): The country.
    
    Returns:
        str: The capital of the given country, or an error message.
    """

    # print(f"\n----- TOOL CALL -----\nget_country_capital({country})")

    key = country.lower().strip()
    if key in location_db:
        found = location_db[key]
        return found["capital"]
    
    return f"Capital not found for {country}."

@tool
def get_country_landmark(country: str) -> str:
    """
    Gets the landmark a given country.

    Args:
        country (str): The country.
    
    Returns:
        str: The landmark of the given country, or an error message.
    """

    # print(f"\n----- TOOL CALL -----\nget_country_landmark({country})")

    key = country.lower().strip()
    if key in location_db:
        found = location_db[key]
        return found["landmark"]
    
    return f"Landmark not found for {country}"

country_agent_prompt = """
You are a country information assistant. Use the provided tools provide your answer.
Do not answer questions not related to country capital or landmark.
"""

country_agent = create_agent(
    model=llm,
    tools=[get_country_capital, get_country_landmark],
    system_prompt=country_agent_prompt,
    middleware=[],
    name="country_agent",
)

supervisor_prompt = """
You are a helpful supervisor that manages 2 agents: 'financial_agent' and a 'country_agent'.

Analyze the input. Do not pass the whole input to agents, just pass relevant info as described below.

Delegate to the 'financial_agent' regarding financial operations.
Delegate to the 'country_agent' queries on capital or landmark of countries.
You can query the agents any number of times as needed.

If the 'financial_agent' or 'country_agent' do not have an answer, do not add any info any more.
Respond with all info returned by the agents.
"""

supervisor = create_supervisor(
    agents=[financial_agent, country_agent],
    model=llm,
    prompt=supervisor_prompt,
)

supervisor_node = supervisor.compile()

def asker_node(state: MessagesState):
    if state["messages"]:
        # for message in state["messages"]:
        #     print(f"\nType: {message.type}:\n{message}")
        print(f"\n===== Response =====\n{state["messages"][-1].content}")
    
    user_input = input("\nHow can I help you? ")
    return {"messages": [HumanMessage(content=user_input)]}

def decide_next(state: MessagesState):
    human_input = state["messages"][-1].content
    normalized_input = human_input.lower().strip()

    if not normalized_input:
        return "asker_node"
    elif any(word in normalized_input for word in ["exit", "quit", "bye"]):
        return "end"
    else:
        return "supervisor"

graph = StateGraph(MessagesState)

graph.add_node("asker_node", asker_node)
graph.add_node("supervisor", supervisor_node)

graph.add_edge(START, "asker_node")
graph.add_conditional_edges(
    "asker_node",
    decide_next,
    {
        "asker_node": "asker_node",
        "supervisor": "supervisor",
        "end": END,
    }
)
graph.add_edge("supervisor", "asker_node")

memory = MemorySaver()
config = {"configurable": {"thread_id": "session_1"}}

app = graph.compile(checkpointer=memory)
# bdata = app.get_graph().draw_mermaid_png()
# with open("diagram.png", "wb") as f:
#     f.write(bdata)

response = app.invoke({}, config=config)

if "__interrupt__" in response:
    interrupts = response["__interrupt__"]
    for intr in interrupts:
        print(f"Interrupted: {intr.id}, {intr.value}")

    print_tool_approval(interrupts[0].value)

    user_input = input("Do I proceed (Y/N)? ")
    approved = user_input.lower() == "y"

    state = app.get_state(config=config)
    if state.next:
        print("===== Resuming the Graph =====")
        response = app.invoke(Command(resume={"approved": approved}), config=config)
        # for message in response["messages"]:
        #     print(f"\nType: {message.type}:\n{message}")

# Buy $1000 of Visa stock at the current price. Get me the landmarks of Egypt, Philippines, China and Australia.
# How about France, Germany, South Korea and Japan?
