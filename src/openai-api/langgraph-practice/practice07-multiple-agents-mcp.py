from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
import asyncio
import os

HOME = os.environ["HOME"]
command_path = HOME + "/playground-ai-ml/.venv/bin/python"
mcp_path = HOME + "/playground-ai-ml/src/openai-api/langgraph-practice/practice07-mcp-server.py"

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://model-runner.docker.internal/engines/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

async def run_client():
    client = MultiServerMCPClient(
        {
            "my_local_server": {
                "command": command_path,
                "args": [mcp_path],
                "transport": "stdio",
                "env": {"ANONYMIZED_TELEMETRY": "false"}, # to disable telemetry
            }
        }
    )

    tools = await client.get_tools()
    llm = ChatOpenAI(
        model="ai/gemma4:E4B",
        base_url=openai_base_url,
        api_key=api_key,
    )
    llm_with_tools = llm.bind_tools(tools)
    agent = create_agent(
        model=llm_with_tools,
        tools=tools,
    )

    query = "What is the time now and what is the stock price of Visa? Were they mentioned in 2024 financial documents?"
    response = await agent.ainvoke({"messages":[HumanMessage(content=query)]})

    print(f"\n==== AI RESPONSE ====\n{response["messages"][-1].content}")

if __name__ == "__main__":
    asyncio.run(run_client())
