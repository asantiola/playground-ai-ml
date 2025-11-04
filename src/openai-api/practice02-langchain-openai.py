from langchain_openai import ChatOpenAI

# practice code using langchain_openai.ChatOpenAI

llm = ChatOpenAI(
    # model="ai/llama3.1",
    model="ai/gpt-oss:latest", 
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

messages = [
    ("system", "You are a helpful AI assistant."),
    ("human", "Create a short story about a chicken and a cat. Just create the story. No need to provide the analysis or explanation."),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)
