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

You must ALWAYS respond with valid JSON.

Available tools:
{json.dumps(TOOLS_SCHEMA, indent=2)}

If no tool is needed:

{{
  "content": "your response"
}}

If one or more tools are needed:

{{
  "tool_calls": [
    {{
      "name": "tool_name",
      "arguments": {{
        "arg1": "value1"
      }}
    }}
  ]
}}

Rules:
- Return either "content" OR "tool_calls", never both.
- tool_calls must be an array.
- Do not wrap JSON in markdown.
- Do not output anything except the JSON object.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string"
        },
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [tool["name"] for tool in TOOLS_SCHEMA]
                    },
                    "arguments": {
                        "type": "object"
                    }
                },
                "required": ["name", "arguments"]
            }
        }
    }
}

def parse_response(model_output: str):
    clean_output = model_output.strip()

    if clean_output.startswith("```"):
        lines = clean_output.split("\n")
        clean_output = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(clean_output)
    except Exception as e:
        return {
            "content": f"Failed to parse model output: {e}"
        }

def execute_tool_calls(response_obj):
    if "tool_calls" not in response_obj:
        return response_obj

    results = []

    for call in response_obj["tool_calls"]:
        tool_name = call["name"]
        arguments = call.get("arguments", {})

        if tool_name not in TOOL_MAP:
            results.append({
                "tool": tool_name,
                "error": "Tool not registered"
            })
            continue

        try:
            result = TOOL_MAP[tool_name](**arguments)

            results.append({
                "tool": tool_name,
                "result": json.loads(result)
            })

        except Exception as e:
            results.append({
                "tool": tool_name,
                "error": str(e)
            })

    return {
        "tool_results": results
    }

model_id = "mlx-community/gemma-4-12B-it-qat-6bit"
print(f"Loading model {model_id}...")
model, processor = load(model_id)
config = model.config

def run_pipeline(user_query: str):
    print(f"\n{'='*50}")
    print(f"User Query: {user_query}")
    print("="*50)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    formatted_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response = generate(
        model,
        processor,
        prompt=formatted_prompt,
        max_tokens=4096,
        temperature=1.0,
        verbose=False,
        extra_sampling_args={
            "json_schema": RESPONSE_SCHEMA
        }
    )

    parsed = parse_response(response.text)

    print("\nParsed Response:")
    print(json.dumps(parsed, indent=2))

    if "tool_calls" in parsed:
        results = execute_tool_calls(parsed)

        print("\nTool Results:")
        print(json.dumps(results, indent=2))
    else:
        print("\nAssistant:")
        print(parsed["content"])

input="""Hey, can you check what the weather looks like in Tokyo right now?
If I invest $10,000 at a 6.5% interest rate for 5 years, how much will I have?
"""

run_pipeline(input)
