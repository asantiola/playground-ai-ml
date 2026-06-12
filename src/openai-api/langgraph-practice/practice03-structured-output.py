from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
import os

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

# docker desktop
llm = ChatOpenAI(
    model="mlx-community/gemma-4-12B-it-6bit", 
    temperature=0,
    base_url=openai_base_url,
    api_key=api_key,
)

class Response(BaseModel):
    answer: str = Field(
        description="The answer to the question"
    )

    confidence_score: float = Field(
        description="Score on how confident you are on the answer, from 0 to 1."
    )

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("""
            You are a helpful assistant that answers the input.
            Keep your answer brief.
        """),
        HumanMessagePromptTemplate.from_template("Input: {input}"),
    ]
)

chain = (prompt | llm.with_structured_output(schema=Response))

response = chain.invoke({ "input": "What is the capital of France?" })
print(f"response:\n{response}\n")

response = chain.invoke({ "input": "How do you measure your confidence in your answer?" })
print(f"response:\n{response}\n")
