from datetime import datetime, date, time, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool, BaseTool
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict

api_url = "http://localhost:12434/engines/v1"
api_key = "docker"
llm_model = "ai/gpt-oss:latest"

llm = ChatOpenAI(
    model=llm_model,
    temperature=0,
    base_url=api_url,
    api_key=api_key,
)

@tool
def get_current_datetime():
    """
    Gets the current date and time, returned as string in YYYYMMDDHHMMSS format
    """
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S").strip()

@tool
def add_datetime_seconds(dt: str, sec: int):
    """
    Adds 'sec' seconds to the provided date time 'dt'. Returns new datetime as string in YYYYMMDDHHMMSS format.
    Args:
        dt: datetime as a string in YYYYMMDDHHMMSS format
        sec: number of seconds to be added
    """
    dt_year = int(dt[0:4])
    dt_mon = int(dt[4:6])
    dt_day = int(dt[6:8])
    dt_hour = int(dt[8:10])
    dt_min = int(dt[10:12])
    dt_sec = int(dt[12:14])

    dtdt = datetime.combine(date(dt_year, dt_mon, dt_day), time(dt_hour, dt_min, dt_sec))
    new_dtdt = dtdt + timedelta(seconds=sec)
    return new_dtdt.strftime("%Y%m%d%H%M%S").strip()

tools = [get_current_datetime, add_datetime_seconds,]
tools_map = {
    "get_current_datetime": get_current_datetime,
    "add_datetime_seconds": add_datetime_seconds,
}

now = get_current_datetime.invoke({})
ten_mins_ago = add_datetime_seconds.invoke({ "dt": now, "sec": -600 })
print(f"now          : {now}")
print(f"ten mins ago : {ten_mins_ago}\n")

class State(TypedDict):
    question: str
    tools: list[str]
    agent_scratchpad: str

def print_state(state: State):
    if 'question' in state: print(f"state.question:\n{state['question']}\n")
    # if 'tools' in state: print(f"state.tools:\n{state['tools']}\n")
    if 'agent_scratchpad' in state: print(f"state.agent_scratchpad:\n{state['agent_scratchpad']}\n")

def reason(state: State):
    prompt = PromptTemplate.from_template(
        """
        Question: This is the question you need to reply to.
        These are the only available tools available to you: {tools}
        Respond with one action that you need to do to reply to the question, in sentence form.
        
        Question: {question}
        Thought: {agent_scratchpad}
        """
    )

    chain = prompt | llm | StrOutputParser()
    thought = chain.invoke({ 
        "question": state["question"], 
        "tools": state["tools"], 
        "agent_scratchpad": state["agent_scratchpad"]
    })
    state["agent_scratchpad"] += " " + thought
    return state

def action(state: State):
    prompt = PromptTemplate.from_template(
        """
        Question: This is the question you need to reply to.
        Given the question, thought and available tools, call the tool.
        These are the only available tools available to you: {tools}
        Question: {question}
        Thought: {agent_scratchpad}
        """
    )
    chain = prompt | llm.bind_tools(state["tools"])
    response = chain.invoke({ 
        "question": state["question"], 
        "tools": state["tools"], 
        "agent_scratchpad": state["agent_scratchpad"]
    })
    
    observation = ""
    for tool_call in response.tool_calls:
        if function_to_call := tools_map.get(tool_call["name"]):
            tool_response = function_to_call.invoke(tool_call["args"])
            observation += " " + tool_response
    
    state["agent_scratchpad"] += f" Observation after calling {tool_call["name"]} = {observation} "
    return state

question = """
    Get the exact time 10 minutes ago since current date and time,
    and compute the number of seconds elapsed since the first of the current month at 00:00.
"""

state = {"question": question,
    "tools": tools,
    "agent_scratchpad": ""
}
print_state(state)

state = reason(state)
print_state(state)

state = action(state)
print_state(state)

state = reason(state)
print_state(state)

state = action(state)
print_state(state)
