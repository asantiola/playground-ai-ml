from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
import asyncio
import os

HOME = os.environ["HOME"]
command_path = HOME + "/repo/playground-ai-ml/.venv/bin/python"
mcp_path = HOME + "/repo/playground-ai-ml/src/openai-api/langgraph-practice/practice07-mcp-server.py"

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
        model="ai/gemma4:4B-128k",
        base_url="http://localhost:12434/engines/v1",
        api_key="docker",
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
