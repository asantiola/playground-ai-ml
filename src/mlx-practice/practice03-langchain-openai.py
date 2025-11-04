from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# practice code using langchain_openai.ChatOpenAI

llm = ChatOpenAI(
    # model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    # model="mlx-community/gpt-oss-20b-MXFP4-Q4",
    # model="mlx-community/phi-4-4bit",
    model="mlx-community/Devstral-Small-2507-4bit",
    temperature=0,
    base_url="http://localhost:12434/v1",
    api_key="mlx-lm",
)

messages = [
    ("system", "You are a helpful AI assistant, Mario from the Super Mario Brothers!"),
    ("human", "{input}"),
]

prompt = ChatPromptTemplate.from_messages(messages)

chain = (prompt | llm | StrOutputParser())

input = "Tell me a short joke."
output = chain.invoke(input)
print(output)
