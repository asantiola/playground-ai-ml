from ollama import Client
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

system_prompt_fmt = """
Extract a list of strings of dates from the input.
Use the date format 'DD-MMM-YYYY'.
List of strings of dates example: ['01-JAN-2002', '31-DEC-2024'].
If input says last Wednesday, it means the nearest previous Wednesday from today.
Split date ranges into individual days.
Take care of leap year in your calculations.
Provide a confidence score on the scale of 0 to 1 for your answer.
No need to explain your answer.
Today is {today} and it is a {day_of_week}.
"""

user_prompt_fmt = "Input: {input}"

now = datetime.datetime.now()
today=now.strftime("%02d-%b-%Y")
day_of_week = calendar.day_name[now.weekday()]

client = Client(
    host=ollama_url
)

print(f"system prompt: {system_prompt_fmt.format(today=today, day_of_week=day_of_week)}\n")
print(f"user prompt: {user_prompt_fmt}\n")

def parse_dates(input: str) -> List[str]:
    print(f"input: {input}\n")
    result = client.chat(
        model=llm_model,
        options={
            "temperature": llm_temp,
        },
        messages=[
            {
                "role": "system",
                "content": system_prompt_fmt.format(today=today, day_of_week=day_of_week)
            },
            {
                "role": "user",
                "content": user_prompt_fmt.format(input=input)
            }
        ],
        format=DatesInfo.model_json_schema()
    )
    print(f"result: {result}\n")
    datesInfo = DatesInfo.model_validate_json(result.message.content)
    print(f"datesInfo: {datesInfo}\n")
    if datesInfo == None:
        return None
    if datesInfo.confidence_score < 0.75:
        return None
    return datesInfo.dates

parse_dates("Get me the logs last Thursday up to last Saturday.")
parse_dates("Rerun my functions for last 3 days.")
