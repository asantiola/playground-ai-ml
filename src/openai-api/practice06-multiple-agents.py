from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence, Union
from pydantic import BaseModel, Field
from datetime import datetime
import os
import operator

llm = ChatOpenAI(
    model="ai/gemma4:4B-128k",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

embeddings = OpenAIEmbeddings(
    model="ai/embeddinggemma:300M-Q8_0",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
    # disable check_embedding_ctx_length if your local model has different constraints
    check_embedding_ctx_length=False,
)

@before_model
def pre_model_hook(state: dict, config: any) -> dict:
    print("\n===== BEFORE LLM CALL =====\n")
    for message in state["messages"]:
        print(f"\nTYPE: {message.type}: {message.content}")
    return state

@after_model
def post_model_hook(state: dict, config: dict) -> dict:
    print(f"\n===== AFTER LLM CALL =====\n")
    for message in state["messages"]:
        print(f"\nTYPE: {message.type}: {message.content}")
    return state

@tool
def get_current_date_time() -> str:
    """Returns the current date time"""

    return datetime.now()

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

HOME = os.environ["HOME"]
persist_directory = HOME + "/repo/playground-ai-ml/.chromadb"
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

financial_tools = [lookup_stock_symbol, get_stock_quotes, retriever_tool]

financial_system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024,
based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data in 2024.
You also have access to other tools related to stocks.
You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
# Please always cite the specific parts of the documents you use in your answers.
# """

financial_agent = create_agent(
    model=llm.bind_tools(financial_tools),
    tools=financial_tools,
    system_prompt=financial_system_prompt,
    # middleware=[pre_model_hook, post_model_hook],
)

datetime_system_prompt="""
You are a helpful AI assistant that gets date and time. 
Use only the provided tools.
Always respond with both date and time information.
"""
datetime_agent = create_agent(
    model=llm.bind_tools([get_current_date_time]),
    tools=[get_current_date_time],
    system_prompt=datetime_system_prompt,
    # middleware=[pre_model_hook, post_model_hook],
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def financial_node(state: AgentState) -> AgentState:
    response = financial_agent.invoke(state)
    return {"messages": [response["messages"][-1]]}

class DateTimeResponse(BaseModel):
    year: int = Field(description="The year")
    month: int = Field(description="The month")
    day: int = Field(description="The day")
    hour: int = Field(description="The hour")
    min: int = Field(description="The minute")
    sec: int = Field(description="The second")

llm_formatter = llm.with_structured_output(schema=DateTimeResponse)
def datetime_node(state: AgentState) -> AgentState:
    unformatted_response = datetime_agent.invoke(state)
    messages = [SystemMessage(content="Extract the DateTimeResponse"), unformatted_response["messages"][-1].content]
    formatted_response = llm_formatter.invoke(messages)
    print(f"===== AI RESPONSE =====\nDateTimeResponse: {formatted_response}\n")
    return {"messages": [unformatted_response["messages"][-1]]}

def general_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def query_node(state: AgentState) -> AgentState:
    system_prompt="""
    Analyze the input in context of previous inputs, then classify the current input.
    If input is about financial information or performance, return 'financial_agent'
    If input is about retrieving date/time, return 'datetime'
    if input is one of the following: 'exit', 'quit', or 'bye', return 'exit'
    Otherwise, return 'general'
    """
    prompt = input("\nWhat is your question: ")
    human_message = HumanMessage(content=prompt)
    response = llm.invoke([SystemMessage(content=system_prompt), human_message])
    classification = response.content

    if classification == "financial_agent":
        return Command(
            update={"messages": [human_message]},
            goto="financial_node",
        )
    elif classification == "datetime":
        return Command(
            update={"messages": [human_message]},
            goto="datetime_node"
        )
    elif classification == "exit":
        return Command(
            goto=END
        )
    else:
        return Command(
            update={"messages": [human_message]},
            goto="general_node",
        )

def responder_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        print(f"===== AI RESPONSE =====\n{last_message.content}\n")
    
    # print("\n===== RESPONDER NODE =====")
    # for message in state["messages"]:
    #     print(f"\nTYPE: {message.type}: {message.content}")

    return None

def build_and_compile():
    graph = StateGraph(AgentState)
    graph.add_node("query_node", query_node)
    graph.add_node("financial_node", financial_node)
    graph.add_node("general_node", general_node)
    graph.add_node("responder_node", responder_node)
    graph.add_node("datetime_node", datetime_node)

    graph.add_edge(START, "query_node")
    graph.add_edge("financial_node", "responder_node")
    graph.add_edge("general_node", "responder_node")
    graph.add_edge("datetime_node", "query_node")
    graph.add_edge("responder_node", "query_node")

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    return app

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "session01"}}
    app = build_and_compile()
    app.invoke({}, config=config)
