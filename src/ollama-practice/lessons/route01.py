from ollama import Client, ChatResponse
from pydantic import BaseModel, Field
from typing import Dict
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

client = Client(
    host=ollama_url
)

class Route(BaseModel):
    support_team: str = Field(
        description="Selected support team."
    )
    reasoning: str = Field(
        description="Reasoning for the selection."
    )

def route(input: str, routes: Dict[str, str]) -> Route:
    prompt = "Analyze the input and select the most appropriate support team from these options: {options}.\nInput: {input}"

    response: ChatResponse = client.chat(
        model=llm_model,
        messages=[
            {
                "role": "user",
                "content": prompt.format(options=list(routes.keys()), input=input)
            }
        ],
        format=Route.model_json_schema()
    )

    return Route.model_validate_json(response.message.content)

def support(prompt: str, input: str) -> str:
    response: ChatResponse = client.chat(
        model=llm_model,
        messages=[
            {
                "role": "user",
                "content": prompt + input
            }
        ]
    )
    return response.message.content

support_routes = {
    "billing": """You are a billing support specialist. Follow these guidelines:
    1. Always start with "Billing Support Response:"
    2. First acknowledge the specific billing issue
    3. Explain any charges or discrepancies clearly
    4. List concrete next steps with timeline
    5. End with payment options if relevant
    
    Keep responses professional but friendly.
    
    Input: """,
    
    "technical": """You are a technical support engineer. Follow these guidelines:
    1. Always start with "Technical Support Response:"
    2. List exact steps to resolve the issue
    3. Include system requirements if relevant
    4. Provide workarounds for common problems
    5. End with escalation path if needed
    
    Use clear, numbered steps and technical details.
    
    Input: """,
    
    "account": """You are an account security specialist. Follow these guidelines:
    1. Always start with "Account Support Response:"
    2. Prioritize account security and verification
    3. Provide clear steps for account recovery/changes
    4. Include security tips and warnings
    5. Set clear expectations for resolution time
    
    Maintain a serious, security-focused tone.
    
    Input: """,
    
    "product": """You are a product specialist. Follow these guidelines:
    1. Always start with "Product Support Response:"
    2. Focus on feature education and best practices
    3. Include specific examples of usage
    4. Link to relevant documentation sections
    5. Suggest related features that might help
    
    Be educational and encouraging in tone.
    
    Input: """
}

# Test with different support tickets
tickets = [
    """Subject: Can't access my account
    Message: Hi, I've been trying to log in for the past hour but keep getting an 'invalid password' error. 
    I'm sure I'm using the right password. Can you help me regain access? This is urgent as I need to 
    submit a report by end of day.
    - John""",
    
    """Subject: Unexpected charge on my card
    Message: Hello, I just noticed a charge of $49.99 on my credit card from your company, but I thought
    I was on the $29.99 plan. Can you explain this charge and adjust it if it's a mistake?
    Thanks,
    Sarah""",
    
    """Subject: How to export data?
    Message: I need to export all my project data to Excel. I've looked through the docs but can't
    figure out how to do a bulk export. Is this possible? If so, could you walk me through the steps?
    Best regards,
    Mike"""
]

for ticket in tickets:
    rt = route(ticket, support_routes)
    if rt == None:
        print(f"Cannot get route for ticket: {ticket}")
        continue
    
    solution = support(support_routes[rt.support_team], ticket)
    print(f"Routing: {rt.support_team}\n")
    print(f"Ticket: {ticket}\n")
    print(f"Solution: {solution}")
    print("----------\n\n")
