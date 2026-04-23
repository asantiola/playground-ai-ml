"""Date and Weather Agent Demo

This script demonstrates a simple conversational chatbot built with
LangGraph's :func:`create_react_agent` and LangChain's
``ChatOpenAI`` wrapper.  It uses a local LLM endpoint (docker DMR)
running the ``gpt-oss:20B`` model.

Two tools are provided:

* ``get_weather(city: str)`` – returns a static string with dummy
  weather data.
* ``compute_date_from_today(delta: int)`` – calculates the date
  ``delta`` days after the current day.

The agent is initialized with a :class:`MemorySaver` to preserve
conversation context across calls.

The ``main`` block demonstrates a query that uses both tools in a
single turn.
"""

from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def get_weather(city: str):
    """Return dummy weather data for a given city.

    Args:
        city: Name of the city.

    Returns:
        A string describing the weather.
    """
    return f"The weather in {city} is sunny, 25C."


@tool
def compute_date_from_today(delta: int):
    """Return the date ``delta`` days from today.

    Args:
        delta: Number of days to add to today's date.

    Returns:
        A string representation of the calculated date in YYYY-MM-DD format.
    """
    target_date = datetime.now().date() + timedelta(days=delta)
    return target_date.strftime("%Y-%m-%d")


TOOLS = [get_weather, compute_date_from_today]

# ---------------------------------------------------------------------------
# LLM and agent setup
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model="ai/gpt-oss:20B",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

memory = MemorySaver()
agent_executor = create_react_agent(llm, TOOLS, checkpointer=memory)


def chat(query: str, thread_id: str):
    """Send a user query to the agent and print the response.

    Args:
        query: The user message.
        thread_id: Identifier for the conversation thread.
    """
    print(f"\nUser: {query}")
    config = {"configurable": {"thread_id": thread_id}}
    response = agent_executor.invoke({"messages": [("user", query)]}, config=config)
    print(f"Agent: {response['messages'][-1].content}")


if __name__ == "__main__":
    THREAD = "session_date_001"

    # Demo query that uses both tools.
    chat("What date will it be tomorrow?", THREAD)

    chat("What was the date 2 days ago?", THREAD)
