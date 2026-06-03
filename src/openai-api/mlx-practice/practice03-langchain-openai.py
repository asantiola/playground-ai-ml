from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

# practice code using langchain_openai.ChatOpenAI

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
    # model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    model="mlx-community/gpt-oss-20b-MXFP4-Q4",
    temperature=0,
    base_url=openai_base_url,
    api_key=api_key,
)

messages = [
    ("system", "You are a helpful and concise coding assistant. Just provide the code, no need for quotes."),
    ("human", "{input}"),
]

prompt = ChatPromptTemplate.from_messages(messages)

chain = (prompt | llm)

input = "Write a short Python function to calculate the factorial of a number."
response = chain.invoke(input)
print(f"response:\n{response.content}")
