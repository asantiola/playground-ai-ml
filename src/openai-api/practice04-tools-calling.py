from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def get_weather(location: str):
    """Use this to get the weather for a specific location."""
    if "sf" in location.lower() or "san francisco" in location.lower():
        return "It's sunny and 20°C in San Francisco."
    else:
        return "I don't know the weather there."

@tool
def get_stock_price(ticker: str):
    """Use this to get the stock price for a company."""
    return f"The stock price of {ticker} is $150."

# Create a list of tools
tools = [get_weather, get_stock_price]

# docker desktop
base_url = "http://localhost:12434/engines/v1"

llm = ChatOpenAI(
    base_url=base_url,
    api_key = "docker",
    model="ai/gpt-oss:20B",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)

def invoke(question: str):
    answer = llm_with_tools.invoke([
        HumanMessage(content=question)
    ])
    print(f"question: {question}\nanswer: {answer.tool_calls}\n\n")

invoke("What is the weather in Manila?")
