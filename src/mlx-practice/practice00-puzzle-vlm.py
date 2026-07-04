from mlx_vlm import load, stream_generate, generate
import mlx.core as mx
import sys
import gc

model_path = "mlx-community/gemma-4-12B-it-6bit"
streaming = True

def mlx_vlm_call(model_path, messages, streaming=True):
    model, processor = load(model_path)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_response = ""
    show_thinking = True

    if streaming:
        print("Streaming:")
        for chunk in stream_generate(
            model,
            processor, 
            prompt,
            max_tokens=8192
        ):
            full_response += chunk.text
    
            if "</thinking>" in full_response and not show_thinking:
                show_thinking = True
                # Print anything that came after </thinking> in this chunk
                parts = full_response.split("</thinking>")
                if len(parts) > 1:
                    print(parts[1].strip(), end="", flush=True)
            elif show_thinking:
                # We are safely past the thinking phase, print normally
                print(chunk.text, end="", flush=True)
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

system_prompt = """You are a precise, direct calculator. 
Your task is to solve mathematical and logic puzzles. 

You MUST structure your response exactly like this:
'<thinking>
[Write all your step-by-step logic, calculations, and analysis here. Take as much space as you need.]
</thinking>'
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

mlx_vlm_call(model_path, messages, streaming)
