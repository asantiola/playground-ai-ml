from mlx_lm import load, stream_generate, generate
import mlx.core as mx
import sys
import gc

model_path = "mlx-community/phi-4-6bit"
streaming = True

def mlx_lm_call(model_path, messages, streaming=True):
    model, tokenizer = load(model_path)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if streaming:
        print("Streaming:")
        for chunk in stream_generate(
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
        response = generate(model, tokenizer, prompt, max_tokens=8192, verbose=False)
        print(f"response: {response}")
        print("\n\n")

    del model
    del tokenizer
    gc.collect()
    mx.clear_cache()


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

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": puzzle_einstein,
    }
]

mlx_lm_call(model_path, messages, streaming)
