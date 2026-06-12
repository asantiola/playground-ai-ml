# A great test for "creative reasoning" with local models.
# Supervisor: Coordinates a story request (e.g., "Describe a cyberpunk city").
# Agent A (The Geographer): Describes the climate, layout, and architecture.
# Agent B (The Historian): Creates a timeline of events that led to the city's current state.
# Agent C (The Social Critic): Describes the different factions and power structures.

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph_supervisor import create_supervisor
from langchain.agents import create_agent
import os

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

llm = ChatOpenAI(
    model="mlx-community/gemma-4-12B-it-6bit",
    base_url=openai_base_url,
    api_key=api_key,
)

geographer_prompt = "You are a Geographer. Describe climates, terrain, and city layouts."
geographer = create_agent(
    model=llm, 
    system_prompt=geographer_prompt,
    name="geographer",
    tools=[]
)

historian_prompt = "You are a Historian. Create timelines and backstories for civilizations."
historian = create_agent(
    model=llm,
    system_prompt=historian_prompt,
    name="historian",
    tools=[]
)

critic_prompt = "You are a Social Critic. Describe power structures, factions, and daily life."
critic = create_agent(
    model=llm,
    system_prompt=critic_prompt,
    name="critic",
    tools=[]
)

supervisor_prompt = """
You are a world creator supervisor.
You are managing several agents.

Delegate to the 'geographer' to create current climate, layout, and architecture.
Delegate to the 'historian' to create timeline of events that led to the city's current state.
Delegate to the 'critic' to describe different factions and power structures.

After you collect all responses from agents, create the world.
Just respond the final world created.
"""

supervisor = create_supervisor(
    agents=[geographer, historian, critic],
    model=llm,
    prompt=supervisor_prompt
)

app = supervisor.compile()
# bdata = app.get_graph().draw_mermaid_png()
# with open("diagram.png", "wb") as f:
#     f.write(bdata)

messages = [HumanMessage(content="Build a deep-sea underwater city called 'Aqualon'.")] 
result = app.invoke({"messages": messages})
print("\n=== ANSWER ===")
print(result['messages'][-1].content)
