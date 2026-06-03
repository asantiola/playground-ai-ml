from operator import contains
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
import json
import os

workspaces = os.environ.get(
    "WORKSPACES",
    "/workspaces"
)

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/engines/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

llm = ChatOpenAI(
    model="ai/gemma4:E4B",
    base_url=openai_base_url,
    api_key=api_key,
)

financial_file = workspaces + "/playground-ai-ml/data/financial01.json"

try:
    with open(financial_file, "r") as f:
        financial_data = json.load(f)
except FileNotFoundError:
    print(f"Error: file not found: {financial_file}")
    exit
except json.JSONDecodeError:
    print(f"Error: file {financial_file} is not a valid JSON")
    exit

df = pd.DataFrame(financial_data["market_data"])

system_prompt = "You are an expert assistant good at analyzing Pandas dataframes converted to CSV"
human_prompt = f"Given the dataset {df.to_csv()}, give the top 3 performing companies, and explain why."

messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
response = llm.invoke(messages)
print(response.content)
