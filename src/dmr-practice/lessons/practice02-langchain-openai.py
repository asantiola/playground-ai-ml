from langchain_openai import ChatOpenAI

# practice code using langchain_openai.ChatOpenAI

llm = ChatOpenAI(
    model="ai/llama3.1",
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

messages = [
    ("system", "You are a helpful AI assistant."),
    ("human", "What is the capital of the United States of America?"),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)
