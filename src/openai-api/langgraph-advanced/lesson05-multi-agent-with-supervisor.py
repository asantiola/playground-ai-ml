
# A great test for "creative reasoning" with local models.
# Supervisor: Coordinates a story request (e.g., "Describe a cyberpunk city").
# Agent A (The Geographer): Describes the climate, layout, and architecture.
# Agent B (The Historian): Creates a timeline of events that led to the city's current state.
# Agent C (The Social Critic): Describes the different factions and power structures.






## Gemini:
# from typing import Annotated, List, Tuple, Union
# from typing_extensions import TypedDict

# from langchain_openai import ChatOpenAI
# from langchain_core.messages import BaseMessage, HumanMessage
# from langchain.agents import AgentExecutor, create_openai_functions_agent
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langgraph.graph import StateGraph, END

# # 1. Configuration - Point to your local provider
# local_llm = ChatOpenAI(
#     base_url="http://localhost:11434/v1", # Change to your provider's port
#     api_key="ollama", # Usually a placeholder for local setups
#     model="llama3"    # Your local model name
# )

# # 2. Define the State
# class AgentState(TypedDict):
#     messages: Annotated[List[BaseMessage], "The messages in the conversation"]
#     next_agent: str

# # 3. Helper to create agents
# def create_world_agent(llm, system_prompt: str):
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         MessagesPlaceholder(variable_name="messages"),
#     ])
#     return prompt | llm

# # 4. Define the Specialists
# geographer = create_world_agent(local_llm, "You are a Geographer. Describe climates, terrain, and city layouts.")
# historian = create_world_agent(local_llm, "You are a Historian. Create timelines and backstories for civilizations.")
# critic = create_world_agent(local_llm, "You are a Social Critic. Describe power structures, factions, and daily life.")

# # 5. Define the Supervisor Logic
# members = ["Geographer", "Historian", "Social_Critic"]
# system_prompt = (
#     "You are the Supervisor. Your job is to coordinate a world-building team."
#     " Based on the conversation, decide who should speak next: {members}."
#     " If the task is finished, respond with FINISH."
# )

# options = ["FINISH"] + members

# def supervisor_node(state: AgentState):
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         MessagesPlaceholder(variable_name="messages"),
#         ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
#     ]).transition(options=options)
    
#     # In a local setup without function calling, we use a simple string parser
#     chain = prompt | local_llm
#     response = chain.invoke({"messages": state["messages"], "members": members, "options": options})
    
#     # Extract the name from the response (simple parsing for local models)
#     content = response.content.upper()
#     next_ = "FINISH"
#     for m in members:
#         if m.upper() in content:
#             next_ = m
#     return {"next_agent": next_}

# # 6. Define Worker Nodes
# def geographer_node(state: AgentState):
#     result = geographer.invoke(state)
#     return {"messages": [HumanMessage(content=result.content, name="Geographer")]}

# def historian_node(state: AgentState):
#     result = historian.invoke(state)
#     return {"messages": [HumanMessage(content=result.content, name="Historian")]}

# def critic_node(state: AgentState):
#     result = critic.invoke(state)
#     return {"messages": [HumanMessage(content=result.content, name="Social_Critic")]}

# # 7. Build the Graph
# workflow = StateGraph(AgentState)

# workflow.add_node("Supervisor", supervisor_node)
# workflow.add_node("Geographer", geographer_node)
# workflow.add_node("Historian", historian_node)
# workflow.add_node("Social_Critic", critic_node)

# # We always return to the Supervisor to decide the next move
# for member in members:
#     workflow.add_edge(member, "Supervisor")

# # The Supervisor decides where to go
# conditional_map = {m: m for m in members}
# conditional_map["FINISH"] = END
# workflow.add_conditional_edges("Supervisor", lambda x: x["next_agent"], conditional_map)

# workflow.set_entry_point("Supervisor")
# graph = workflow.compile()

# # 8. Run it
# inputs = {"messages": [HumanMessage(content="Build a deep-sea underwater city called 'Aqualon'.")]}
# for output in graph.stream(inputs):
#     for key, value in output.items():
#         print(f"--- {key} ---")
#         print(value)
