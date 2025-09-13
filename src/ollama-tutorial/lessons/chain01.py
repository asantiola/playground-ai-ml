from typing import List
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import os
import json

ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
ollama_url = f"http://{ollama_host}:11434"

ollama_config_file = "/workspace/data/ollama_conf.json"
with open(ollama_config_file) as file:
    ollama_config = json.load(file)

llm_model = ollama_config.get("llm_model", "llama3.1")
llm_temp = ollama_config.get("llm_temp", 0.0)

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")

llm = ChatOllama(
    model=llm_model,
    temperature=llm_temp,
    base_url=ollama_url
)

def chain(input: str, prompts: List[str]) -> str:
    """Chain multiple LLM calls sequentially, passing results between steps."""
    result = input
    for i, prompt in enumerate(prompts, 1):
        print(f"\nStep {i}:")
        prompt_template = PromptTemplate(
            template=prompt + "{input}",
            input_variables=["input"]
        )
        chain = prompt_template | llm
        result = chain.invoke({ "input": result })
        print(result.content)
    return result.content

data_processing_steps = [
    """Extract only the numerical values and their associated metrics from the text.
    Format each as 'value: metric' on a new line.
    Example format:
    92: customer satisfaction
    45%: revenue growth""",
    
    """Convert all numerical values to percentages where possible.
    If not a percentage or points, convert to decimal (e.g., 92 points -> 92%).
    Keep one number per line.
    Example format:
    92%: customer satisfaction
    45%: revenue growth""",
    
    """Sort all lines in descending order by numerical value.
    Keep the format 'value: metric' on each line.
    Example:
    92%: customer satisfaction
    87%: employee satisfaction""",
    
    """Format the sorted data as a markdown table with columns:
    | Metric | Value |
    |:--|--:|
    | Customer Satisfaction | 92% |"""
]

report = """
Q3 Performance Summary:
Our customer satisfaction score rose to 92 points this quarter.
Revenue grew by 45% compared to last year.
Market share is now at 23% in our primary market.
Customer churn decreased to 5% from 8%.
New user acquisition cost is $43 per user.
Product adoption rate increased to 78%.
Employee satisfaction is at 87 points.
Operating margin improved to 34%.
"""

print("\nInput:")
print(report)

formatted_result = chain(report, data_processing_steps)

print("\nFormatted Result:\n")
print(formatted_result)
