# A great test for "creative reasoning" with local models.
# Supervisor: Coordinates a story request (e.g., "Describe a cyberpunk city").
# Agent A (The Geographer): Describes the climate, layout, and architecture.
# Agent B (The Historian): Creates a timeline of events that led to the city's current state.
# Agent C (The Social Critic): Describes the different factions and power structures.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph_supervisor import create_supervisor
from langchain.agents import create_agent
from typing import Annotated, Sequence, TypedDict
import operator

llm = ChatOpenAI(
    model="ai/gemma4:E4B",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
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
You are managing a geographer, historian, critic.
Delegate to the 'geographer' to create current climate, layout, and architecture.
Delegate to the 'historian' to create timeline of events that led to the city's current state.
Delegate to the 'critic' to describe different factions and power structures.

After you collect their inputs, create the world requested.
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
