from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model="ai/gpt-oss:latest", 
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
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
