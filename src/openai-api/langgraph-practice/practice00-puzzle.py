from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    # "http://localhost:12434/v1"     # Docker Model Runner
    "http://localhost:12434/v1"             # MLX
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

llm = ChatOpenAI(
    # model="ai/phi4:14B-Q4_K_M",
    # model="mlx-community/gemma-4-12B-it-6bit",
    # model="mlx-community/phi-4-6bit",
    model="mlx-community/gemma-4-12B-it-6bit",
    base_url=openai_base_url,
    api_key=api_key,
)

system_prompt = """You are an expert mathematical logician who specializes in combinatorics and probability puzzles.
You approach problems step-by-step, verify boundary conditions, 
and rigorously check your assumptions before calculating the final answer.
"""

human_prompt = """There are five houses of different colors adjacent to one another on a road. 
In each house lives a man of different nationality. 
Each man has a favorite drink, a favorite brand of cigarettes, and keeps a different kind of pet.

The Englishman lives in the red house.
The Swede keeps dogs.
The Dane drinks tea.
The green house is just to the left of the white house.
The owner of the green house drinks coffee.
The Pall Mall smoker keeps birds.
The owner of the yellow house smokes Dunhills.
The man in the center house drinks milk.
The Norwegian lives in the first house.
The Blend Smoker has a neighbor who keeps cats.
The man who smokes Blue Masters drinks bier.
The man who keeps horses lives next to the Dunhill smoker.
The German smokes Prince.
The Norwegian lives next to the blue house.
The Blend smoker has a neighbor who drinks water.

Who owns the fish?
"""

messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
print("")
