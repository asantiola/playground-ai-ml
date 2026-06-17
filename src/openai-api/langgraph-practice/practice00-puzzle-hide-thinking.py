from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

def selection(what, choices_names, choices):
    print(f"Select a {what}:")
    for index, option in enumerate(choices_names, start=1):
        print(f"[{index}] {option}")

    while True:
        try:
            choice = int(input("\nEnter the number of your choice: "))

            if 1 <= choice <= len(choices):
                return choices[choice - 1]
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

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

llm = ChatOpenAI(
    model="mlx-community/gemma-4-12B-it-6bit",
    base_url=openai_base_url,
    api_key=api_key,
    temperature=0,
    max_completion_tokens=256
)

system_prompt = """You are a precise, direct calculator. 
Your task is to solve mathematical and logic puzzles. 

You MUST structure your response exactly like this:
<thinking>
[Write all your step-by-step logic, calculations, and analysis here. Take as much space as you need.]
</thinking>
[Write ONLY the final answer here. If it is a number, print just the number. If it requires a brief label, keep it under 10 words.]
"""

puzzle_einstein = """There are five houses of different colors adjacent to one another on a road. 
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

puzzle_drunks = """A line of 100 airline passengers are waiting to board a plane.
They each hold a ticket to one of the 100 seats on that flight.
For convenience, let’s say that the n-th passenger in line has a ticket for the seat number n.
Being drunk, the first person in line picks a random seat (equally likely for each seat).
All of the other passengers are sober, and will go to their proper seats unless it is already occupied;
In that case, they will randomly choose a free seat. You’re person number 100.
What is the probability that you end up in your seat (i.e., seat #100) ?
"""

puzzle_floors = """A building has 10 floors above the basement.
If 12 people get into an elevator at the basement, and each chooses a floor at random to get out, 
independently of the others,
at how many floors do you expect the elevator to make a stop to let out one or more of these 12 people?
"""

puzzle_code = """Crack the code, solve for the 3 unique digits code.
Hints:
- 294: Exactly one number is correct and well placed.
- 836: Exactly one number is correct but wrongly placed.
- 165: Exactly one number is correct and well placed.
- 874: Nothing is correct.
- 473: Exactly one number is correct and well placed.
"""

puzzle_alchemist_vault = """Four magical artifacts—the Ruby, the Sapphire, the Emerald, and the Amethyst—are locked inside
four identical moving vaults labeled Vault A, Vault B, Vault C, and Vault D.

Initially, the items are placed like this:
- Vault A contains the Ruby.
- Vault B contains the Sapphire.
- Vault C contains the Emerald.
- Vault D contains the Amethyst.

The vaults are arranged in a straight row from left to right: [A] - [B] - [C] - [D].

An alchemist performs a sequence of five operations. Read each step carefully:
1. The Left-Shift: The alchemist takes the physical vault on the far left and moves it to the far right of the row. 
   (The contents stay inside their respective vaults).
2. The Blue Swap: The alchemist opens the physical vault currently in the 2nd position from the left, 
   and the physical vault currently in the 4th position from the left. He swaps the contents inside them.
3. The Inverse Mirror: The entire physical row of vaults is completely reversed from left to right. 
   (The vault on the far left goes to the far right, the 2nd goes to the 3rd, etc.)
4. The Gem Extraction: The alchemist opens Vault B and Vault D. He takes the gemstone out of Vault B and places it into Vault D, 
   and takes the gemstone out of Vault D and places it into Vault B.
5. The Final Shift: The alchemist takes the physical vault currently in the 3rd position from the left and moves it to the far left 
   of the row, sliding the others over to make room.

After all five steps are completed, which gemstone is inside Vault A, and what is the exact contents of the physical vault 
currently sitting in the 3rd position from the left?
"""

puzzles_names = [
    "Einstein's' Puzzle", 
    "Drunk Passengers", 
    "Building Floors", ""
    "Crack the Code",
    "The Alchemist’s Moving Vault"
]
puzzles = [
    puzzle_einstein, 
    puzzle_drunks, 
    puzzle_floors, 
    puzzle_code, 
    puzzle_alchemist_vault
]

selected_puzzle = selection("puzzle", puzzles_names, puzzles)

messages = [SystemMessage(content=system_prompt), HumanMessage(content=selected_puzzle)]
full_response = ""
thinking_ended = False

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
