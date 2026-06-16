from mlx_lm import load as lm_load, stream_generate as lm_stream_generate, generate as lm_generate
from mlx_vlm import load as vlm_load, stream_generate as vlm_stream_generate, generate as vlm_generate
import mlx.core as mx
import sys
import gc

model_path_phi4_14b="mlx-community/phi-4-6bit"
model_path_gemma4_12b="mlx-community/gemma-4-12B-it-6bit"
model_paths=[model_path_phi4_14b, model_path_gemma4_12b]

def mlx_lm_call(model_path, messages, streaming=True):
    model, tokenizer = lm_load(model_path)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if streaming:
        print("Streaming:")
        for chunk in lm_stream_generate(
            model,
            tokenizer, 
            prompt,
            max_tokens=8192
        ):
            sys.stdout.write(chunk.text)
            sys.stdout.flush()
        print("\n\n")
    else:
        print("Generating:")
        response = lm_generate(model, tokenizer, prompt, max_tokens=8192, verbose=False)
        print(f"response: {response}")
        print("\n\n")

    del model
    del tokenizer
    gc.collect()
    mx.clear_cache()

def mlx_vlm_call(model_path, messages, streaming=True):
    model, tokenizer = vlm_load(model_path)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if streaming:
        print("Streaming:")
        for chunk in vlm_stream_generate(
            model,
            tokenizer, 
            prompt,
            max_tokens=8192
        ):
            sys.stdout.write(chunk.text)
            sys.stdout.flush()
        print("\n\n")
    else:
        print("Generating:")
        response = vlm_generate(model, tokenizer, prompt, max_tokens=8192, verbose=False)
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
framework = {
    model_path_phi4_14b: mlx_lm_call,
    model_path_gemma4_12b: mlx_vlm_call,
}

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

puzzle_drunks="""A line of 100 airline passengers are waiting to board a plane.
They each hold a ticket to one of the 100 seats on that flight.
For convenience, let’s say that the n-th passenger in line has a ticket for the seat number n.
Being drunk, the first person in line picks a random seat (equally likely for each seat).
All of the other passengers are sober, and will go to their proper seats unless it is already occupied;
In that case, they will randomly choose a free seat. You’re person number 100.
What is the probability that you end up in your seat (i.e., seat #100) ?
"""

puzzle_floors="""A building has 10 floors above the basement.
If 12 people get into an elevator at the basement, and each chooses a floor at random to get out, 
independently of the others,
at how many floors do you expect the elevator to make a stop to let out one or more of these 12 people?
"""

puzzle_code="""Crack the code, solve for the 3 unique digits code.
Hints:
- 294: Exactly one number is correct and well placed.
- 836: Exactly one number is correct but wrongly placed.
- 165: Exactly one number is correct and well placed.
- 874: Nothing is correct.
- 473: Exactly one number is correct and well placed.
"""

puzzles_names = ["Einstein's' Puzzle", "Drunk Passengers", "Building Floors", "Crack the Code"]
puzzles = [puzzle_einstein, puzzle_drunks, puzzle_floors, puzzle_code]

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

framework[selected_model](selected_model, messages, streaming)
