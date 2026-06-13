from mlx_vlm import load, stream_generate, generate
import sys

model_path="mlx-community/gemma-4-12B-it-6bit"

system_prompt = """You are an expert mathematical logician who specializes in combinatorics and probability puzzles.
You approach problems step-by-step, verify boundary conditions, 
and rigorously check your assumptions before calculating the final answer.
"""

puzzle_prompt_einstein = """There are five houses of different colors adjacent to one another on a road. 
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

puzzle_prompt_drunk = """A line of 100 airline passengers are waiting to board a plane.
They each hold a ticket to one of the 100 seats on that flight.
For convenience, let’s say that the n-th passenger in line has a ticket for the seat number n.
Being drunk, the first person in line picks a random seat (equally likely for each seat).
All of the other passengers are sober, and will go to their proper seats unless it is already occupied;
In that case, they will randomly choose a free seat. You’re person number 100.
What is the probability that you end up in your seat (i.e., seat #100) ?
"""

puzzle_prompt_floors = """A building has 10 floors above the basement.
If 12 people get into an elevator at the basement, and each chooses a floor at random to get out, 
independently of the others,
at how many floors do you expect the elevator to make a stop to let out one or more of these 12 people?
"""

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": puzzle_prompt_einstein,
    }
]

model, tokenizer = load(model_path)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("Streaming:")
for chunk in stream_generate(model, tokenizer, prompt):
    sys.stdout.write(chunk.text)
    sys.stdout.flush()
print("\n\n")

print("Generating:")
response = generate(model, tokenizer, prompt, verbose=False)
print(f"text: {response.text}")
print("")
print(f"token: {response.token}")
print("")
print(f"logprobs: {response.logprobs}")
print("")
print(f"prompt_tokens: {response.prompt_tokens}")
print("")
print(f"generation_tokens: {response.generation_tokens}")
print("")
print(f"total_tokens: {response.total_tokens}")
print("")
print(f"prompt_tps: {response.prompt_tps}")
print("")
print(f"generation_tps: {response.generation_tps}")
print("")
print(f"peak_memory: {response.peak_memory}")
print("\n\n")
