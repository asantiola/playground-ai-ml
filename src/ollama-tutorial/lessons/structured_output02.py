from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import os
import json
import datetime
import calendar

ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
ollama_url = f"http://{ollama_host}:11434"

ollama_config_file = "/workspace/data/ollama_conf.json"
with open(ollama_config_file) as file:
    ollama_config = json.load(file)

llm_model = ollama_config.get("llm_model", "llama3.1")
llm_temp = ollama_config.get("llm_temp", 0.0)

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")
print("\n", "-------------------------\n")

class DatesInfo(BaseModel):
    dates: List[str] = Field(
        description="List of strings of dates."
    )
    confidence_score: float = Field(
        description="Confidence score on the scale of 0 to 1 for the date selection."
    )

system_prompt = """
Extract a list of strings of dates from the input.
Use the date format 'DD-MMM-YYYY'.
List of strings of dates example: ['01-JAN-2002', '31-DEC-2024'].
If input says last Wednesday, it means the nearest previous Wednesday from today.
Take care of leap year in your calculations.
Provide a confidence score on the scale of 0 to 1 for your answer.
No need to explain your answer.
Today is {today} and it is a {day_of_week}.
"""

now = datetime.datetime.now()
today=now.strftime("%02d-%b-%Y")
day_of_week = calendar.day_name[now.weekday()]

prompt = ChatPromptTemplate([
    ("system", system_prompt.format(today=today, day_of_week=day_of_week)),
    ("human", "Input: {input}")
])

print(f"prompt: {prompt}\n")

llm = ChatOllama(
    base_url=ollama_url,
    model=llm_model,
    temperature=llm_temp,
)
chain = prompt | llm.with_structured_output(DatesInfo)

def parse_dates(input: str) -> List[str]:
    print(f"input: {input}\n")
    result = chain.invoke({ "input": input })
    print(f"result: {result}\n")
    if result == None:
        return None
    if result.confidence_score < 0.75:
        return None
    return result.dates

parse_dates("Get me the logs last Thursday to Saturday.")
parse_dates("Rerun my functions for last 3 days.")
