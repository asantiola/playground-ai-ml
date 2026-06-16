from mlx_vlm import load, stream_generate, generate
import mlx.core as mx
import sys
import gc

model_path_gemma4_12b = "mlx-community/gemma-4-12B-it-6bit"
model_path_gemma4_e4b = "mlx-community/gemma-4-E4B-it-qat-6bit"

model_paths = [model_path_gemma4_12b, model_path_gemma4_e4b]

def mlx_vlm_call(model_path, messages, streaming=True):
    model, processor = load(model_path)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if streaming:
        print("Streaming:")
        for chunk in stream_generate(
            model,
            processor, 
            prompt,
            max_tokens=8192
        ):
            sys.stdout.write(chunk.text)
            sys.stdout.flush()
        print("\n\n")
    else:
        print("Generating:")
        response = generate(model, processor, prompt, max_tokens=8192, verbose=False)
        print(f"response.text: {response.text}")
        print("")
        print(f"response.token: {response.token}")
        print("")
        print(f"response.logprobs: {response.logprobs}")
        print("")
        print(f"response.prompt_tokens: {response.prompt_tokens}")
        print("")
        print(f"response.generation_tokens: {response.generation_tokens}")
        print("")
        print(f"response.total_tokens: {response.total_tokens}")
        print("")
        print(f"response.prompt_tps: {response.prompt_tps}")
        print("")
        print(f"response.generation_tps: {response.generation_tps}")
        print("")
        print(f"response.peak_memory: {response.peak_memory}")
        print("\n\n")

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

selected_model = selection("model", model_paths, model_paths)


system_prompt = """You are an expert mathematical logician who specializes in combinatorics and probability puzzles.
You approach problems step-by-step, verify boundary conditions, 
and rigorously check your assumptions before calculating the final answer.
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

streaming = selection("streaming", ["Streaming", "Generation"], [True, False])

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": selected_puzzle,
    }
]

print(f"selected_model: {selected_model}")
print(f"selected_puzzle: {selected_puzzle}")

mlx_vlm_call(selected_model, messages, streaming)
