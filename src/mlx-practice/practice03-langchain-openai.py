from langchain_openai import ChatOpenAI

# practice code using langchain_openai.ChatOpenAI

llm = ChatOpenAI(
    model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    # model="mlx-community/gpt-oss-20b-MXFP4-Q4",
    temperature=0,
    base_url="http://localhost:12434/v1",
    api_key="mlx-lm",
)

messages = [
    ("system", "You are a helpful AI assistant."),
    ("human", "Create a short story about a chicken and a cat. Just create the story. No need to provide the analysis or explanation."),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)
