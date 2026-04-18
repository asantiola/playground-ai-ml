from langchain_openai import ChatOpenAI

# practice code using langchain_openai.ChatOpenAI

# # docker desktop
# llm = ChatOpenAI(
#     # model="ai/llama3.1",
#     model="ai/gpt-oss:latest", 
#     temperature=0,
#     base_url="http://localhost:12434/engines/v1",
#     api_key="docker",
# )

# podman desktop
llm = ChatOpenAI(
    # model="ai/llama3.1",
    model="openai/gpt-oss-20b:latest", 
    temperature=0,
    base_url="http://localhost:12434/v1",
    api_key="podman",
)

messages = [
    ("system", "You are a helpful AI assistant. Show me your thinking process."),
    ("human", "Tell me a joke."),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)
