from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# practice code using langchain_openai.ChatOpenAI

llm = ChatOpenAI(
    # model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    model="mlx-community/gpt-oss-20b-MXFP4-Q4",
    temperature=0,
    base_url="http://localhost:12434/v1",
    api_key="mlx-lm",
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
