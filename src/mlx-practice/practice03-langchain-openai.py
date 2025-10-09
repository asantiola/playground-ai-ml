from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# practice code using langchain_openai.ChatOpenAI

llm = ChatOpenAI(
    # model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    model="mlx-community/gpt-oss-20b-MXFP4-Q4",
    temperature=0,
    base_url="http://localhost:12434/v1",
    api_key="mlx-lm",
)

messages = [
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}"),
]

prompt = ChatPromptTemplate.from_messages(messages)

chain = (prompt | llm | StrOutputParser())

input = "Create a short story about a chicken and a cat. Just create the story. No need to provide the analysis or explanation."
output = chain.invoke(input)
print(output)
