import json
from mlx_vlm import load, generate

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a specific location."""

    print(f"\n[TOOL EXECUTION] Running 'get_weather' for {location} in {unit}...")
    if "tokyo" in location.lower():
        return json.dumps({"location": location, "temperature": 18, "condition": "Rainy", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": 27, "condition": "Sunny", "unit": unit})

def calculate_investment(principal: float, rate: float, years: int) -> str:
    """Calculate future value of a principal sum compounded annually."""
    print(f"\n[TOOL EXECUTION] Running 'calculate_investment' for ${principal} at {rate}% over {years} years...")
    future_value = principal * ((1 + (rate / 100)) ** years)
    return json.dumps({"principal": principal, "future_value": round(future_value, 2), "years": years})

TOOL_MAP = {
    "get_weather": get_weather,
    "calculate_investment": calculate_investment
}

TOOLS_SCHEMA = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a specific location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. Tokyo, Japan"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate_investment",
        "description": "Calculate future value of a principal sum compounded annually.",
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "The initial money invested"},
                "rate": {"type": "number", "description": "Annual interest rate in percentage, e.g., 5.5"},
                "years": {"type": "integer", "description": "Number of years to mature"}
            },
            "required": ["principal", "rate", "years"]
        }
    }
]

SYSTEM_PROMPT = f"""You are a helpful assistant with access to tools. 
If the user's request requires information from a tool, you MUST reply with a JSON object calling that tool. Do not include any conversational text or markdown formatting if a tool is needed.

Available tools:
{json.dumps(TOOLS_SCHEMA, indent=2)}

If you need to call a tool, format your output exactly like this JSON structure:
{{
  "tool_call": {{
    "name": "tool_name",
    "arguments": {{
      "arg1": "value1"
    }}
  }}
}}
"""

def parse_and_execute_tool(model_output: str) -> str:
    """Parses raw text from the model, finds the tool block, and runs the local function."""
    clean_output = model_output.strip()
    
    # Strip markdown code blocks if the model accidentally generates them
    if clean_output.startswith("```"):
        lines = clean_output.split("\n")
        if lines[0].startswith("```json") or lines[0].startswith("```"):
            clean_output = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(clean_output)
        if "tool_call" in data:
            tool_name = data["tool_call"]["name"]
            arguments = data["tool_call"]["arguments"]
            
            if tool_name in TOOL_MAP:
                result = TOOL_MAP[tool_name](**arguments)
                return f"Tool Execution Result: {result}"
            else:
                return f"Error: Tool '{tool_name}' is not registered."
        else:
            return f"No tool call requested. Direct response: {model_output}"
            
    except json.JSONDecodeError:
        return f"Failed to parse JSON tool call. Raw response: {model_output}"

model_id = "mlx-community/gemma-4-12B-it-6bit"
print(f"Loading model {model_id}...")
model, processor = load(model_id)
config = model.config

def run_pipeline(user_query: str):
    print(f"\n" + "="*50)
    print(f"User Query: {user_query}")
    print("="*50)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    response = generate(
        model, 
        processor, 
        prompt=formatted_prompt, 
        max_tokens=4096, 
        temperature=1.0, 
        verbose=False,
        extra_sampling_args={"json_schema": TOOLS_SCHEMA}
    )
    
    print(f"Model Output:\n{response.text}")
    
    execution_result = parse_and_execute_tool(response.text)
    print(execution_result)

run_pipeline("Hey, can you check what the weather looks like in Tokyo right now?")

run_pipeline("If I invest $10,000 at a 6.5% interest rate for 5 years, how much will I have?")
