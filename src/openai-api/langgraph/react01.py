import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.tools import Tool, tool

HOME=os.environ["HOME"]
api_url = "http://localhost:12434/engines/v1"
api_key = "docker"
llm_model = "ai/gpt-oss:latest"
db_uri = "sqlite:///" + HOME + "/repo/playground-ai-ml/data/sql-agentic-ai.db"

llm = ChatOpenAI(
    model=llm_model,
    temperature=0,
    base_url=api_url,
    api_key=api_key,
)

db = SQLDatabase.from_uri(db_uri)
sql_agent = create_sql_agent(llm=llm, db=db, agent_type="tool-calling", verbose=False)

def sql_agent_wrapper(input: dict[str, any]):
    response = sql_agent.invoke(input)
    return response["output"]

sql_query_tool = Tool(
    name="sql_agent",
    description="This tool is an agent that queries the Albatross company database for a given input.",
    func=sql_agent_wrapper,
)

@tool
def calculator_add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def calculator_subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b

@tool
def calculator_multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def calculator_divide(a: float, b: float) -> float:
    """Add two numbers."""
    return a / b

tools = []
