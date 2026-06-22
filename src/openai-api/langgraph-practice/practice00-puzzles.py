from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from practice00_common import selection
import os
import glob

workspaces = os.environ.get(
    "WORKSPACES",
    "/workspaces"
)

# Docker Model Runner
# openai_base_url = os.environ.get(
#     "OPENAI_BASE_URL", 
#     "http://localhost:12434/engines/v1"
# )

# MLX
openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

model_names = [
    "mlx-community/gemma-4-12B-it-6bit",
    "ggml-org/gpt-oss-20b-GGUF",
    "mlx-community/phi-4-6bit",
]

model_details = [
    {
        "model": "mlx-community/gemma-4-12B-it-6bit",
        "temperature": 0.0,
        "max_completion_tokens": 4096
    },
    {
        "model": "ggml-org/gpt-oss-20b-GGUF",
        "temperature": 1.0,
        "max_completion_tokens": None
    },
    {
        "model": "mlx-community/phi-4-6bit",
        "temperature": 0.0,
        "max_completion_tokens": 4096
    },
]

# model_chosen = selection("models", model_names, model_details)
model_chosen = model_details[0]

llm = ChatOpenAI(
    model=model_chosen["model"],
    base_url=openai_base_url,
    api_key=api_key,
    temperature=model_chosen["temperature"],
    max_completion_tokens=model_chosen["max_completion_tokens"]
)

system_prompt_show = """You are an expert mathematical logician who specializes in combinatorics and probability puzzles.
You approach problems step-by-step, verify boundary conditions, 
and rigorously check your assumptions before calculating the final answer.
"""

system_prompt_hide = """You are a precise, direct calculator. 
Your task is to solve mathematical and logic puzzles. 

You MUST structure your response exactly like this:
<thinking>
[Write all your step-by-step logic, calculations, and analysis here. Take as much space as you need.]
</thinking>
[Write ONLY the final answer here. If it is a number, print just the number. If it requires a brief label, keep it under 10 words.]
"""

puzzles_names = []
puzzles = []
puzzle_path = os.path.join(workspaces, "playground-ai-ml/data/puzzles")
search_pattern = os.path.join(puzzle_path, "*.txt")
for file_path in sorted(glob.glob(search_pattern)):
    with open(file_path, "r") as file:
        name = file.readline().strip()
        puzzles_names.append(name)

        puzzle = file.read()
        puzzles.append(puzzle)

selected_puzzle = selection("puzzle", puzzles_names, puzzles)

show_think = [True, "show thinking tokens", system_prompt_show]
hide_think = [False, "hide thinking tokens", system_prompt_hide]
option_think = selection("an option", [show_think[1], hide_think[1]], [show_think, hide_think]) 
system_prompt = option_think[2]

messages = [SystemMessage(content=system_prompt), HumanMessage(content=selected_puzzle)]
full_response = ""
thinking_ended = option_think[0]

print(f"Puzzle/Riddle:\n{selected_puzzle}\n")

print("\nResult:")
for chunk in llm.stream(messages):
    full_response += chunk.content
    
    if "</thinking>" in full_response and not thinking_ended:
        thinking_ended = True
        # Print anything that came after </thinking> in this chunk
        parts = full_response.split("</thinking>")
        if len(parts) > 1:
            print(parts[1].strip(), end="", flush=True)
    elif thinking_ended:
        # We are safely past the thinking phase, print normally
        print(chunk.content, end="", flush=True)
        
print("\n")
