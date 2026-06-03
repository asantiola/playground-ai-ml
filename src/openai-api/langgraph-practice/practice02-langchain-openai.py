from langchain_openai import ChatOpenAI
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

# docker desktop
llm = ChatOpenAI(
    model="ai/gemma4:E4B", 
    temperature=0,
    base_url=openai_base_url,
    api_key=api_key,
)

messages = [
    ("system", "You are a helpful AI assistant. Show me your thinking process."),
    ("human", "Tell me a joke."),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)
