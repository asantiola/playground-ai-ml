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
    # model="mlx-community/gemma-4-12B-it-qat-6bit",
    # model="unsloth/gemma-4-12b-it-GGUF:UD-Q6_K_XL",
    model="google/gemma-4-12B-it-qat-q4_0-gguf",
    base_url=openai_base_url,
    api_key=api_key,
    temperature=1.0,
    extra_body={
        "top_p": 0.95,
        "top_k": 64,
    },
)

class Response(BaseModel):
    reasoning: str = Field(
        description="Step-by-step logical deduction analyzing each hint to answer the question."
    )

    answer: str = Field(
        description="The answer to the question"
    )

    confidence_score: float = Field(
        description="Score on how confident you are on the answer, from 0 to 1."
    )

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("""
            You are a logical assistant. In the reasoning field, write a concise step-by-step 
            elimination log (1-2 short sentences per hint) BEFORE giving the final answer.
        """),
        HumanMessagePromptTemplate.from_template("Input: {input}"),
    ]
)

chain = (prompt | llm.with_structured_output(schema=Response))

simple_question = "What is the capital of France?"

response = chain.invoke({ "input": simple_question })
print(f"response:\n{response}\n")

puzzle_question = """Crack the code, solve for the 3 unique digits code.
Hints:
- 294: Exactly one number is correct and well placed.
- 836: Exactly one number is correct but wrongly placed.
- 165: Exactly one number is correct and well placed.
- 874: Nothing is correct.
- 473: Exactly one number is correct and well placed.
"""

response = chain.invoke({ "input": puzzle_question })
print(f"answer: {response.answer}, confidence_score: {response.confidence_score}")
