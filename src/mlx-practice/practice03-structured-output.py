from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# practice code using langchain_openai.ChatOpenAI

llm = ChatOpenAI(
    model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    # model="mlx-community/gpt-oss-20b-MXFP4-Q4",
    temperature=0,
    base_url="http://localhost:12434/v1",
    api_key="mlx-lm",
)

messages = [
    ("system", "You are a helpful ."),
    ("human", "{input}"),
]

class Joke(BaseModel):
    setup: str = Field(
        description="The setup of the joke."
    )
    punch_line: str = Field(
        description="The punchline of the joke."
    )

structured_llm = llm.with_structured_output(Joke)
prompt = ChatPromptTemplate.from_messages(messages)

chain = prompt | structured_llm
# chain = prompt | llm | StrOutputParser()

input = "Tell me a joke about cheese."
output = chain.invoke(input)
print(output)
