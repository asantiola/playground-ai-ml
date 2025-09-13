from ollama import Client, ChatResponse
from pydantic import BaseModel, Field
import os
import json

ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
ollama_url = f"http://{ollama_host}:11434"

ollama_config_file = "/workspace/data/ollama_conf.json"
with open(ollama_config_file) as file:
    ollama_config = json.load(file)

llm_model = ollama_config.get("llm_model", "llama3.1")
llm_temp = ollama_config.get("llm_temp", 0.0)

print(f"Using LLM model: {llm_model}")
print(f"Using LLM temp: {llm_temp}")
print("\n", "-------------------------\n")

options = {
    "temperature": llm_temp
}

client = Client(
    host=ollama_url
)

debate_config_file = "/workspace/data/debate_conf.json"
with open(debate_config_file) as file:
    debate_config = json.load(file)

max_rounds : int = debate_config.get("max_rounds")
lhs_speaker : str = debate_config.get("lhs_speaker")
rhs_speaker : str = debate_config.get("rhs_speaker")
theme : str = debate_config.get("theme")
lhs_opinion : str = debate_config.get("lhs_opinion")
rhs_opinion : str = debate_config.get("rhs_opinion")
prompt_lhs_fmt : str = debate_config.get("prompt_lhs")
prompt_rhs_fmt : str = debate_config.get("prompt_rhs")
prompt_conclude_fmt : str = debate_config.get("prompt_conclude")

prompt_lhs = prompt_lhs_fmt.format(theme=theme, lhs_opinion=lhs_opinion)
prompt_rhs = prompt_rhs_fmt.format(theme=theme, rhs_opinion=rhs_opinion)
prompt_conclude = prompt_conclude_fmt.format(max_rounds=max_rounds, theme=theme)

print(f"Max rounds: {max_rounds}")
print(f"Prompt LHS: {prompt_lhs}", "\n", "-------------------------\n")
print(f"Prompt RHS: {prompt_rhs}", "\n", "-------------------------\n")
print(f"Prompt Conclusion: {prompt_conclude}", "\n", "-------------------------\n")

class Conclusion(BaseModel):
    is_concluded: bool = Field(
        description="Indicates if the debate is already concluded."
    )
    score: float = Field(
        description="Confidence score on the scale of 0 to 1 on your answer."
    )
    reasoning: str = Field(
        description="The reasoning for the answer."
    )

def conclude(prompt: str, history: str, rounds: int) -> Conclusion:
    response : ChatResponse = client.chat(
        messages=[
            {
                "role": "user",
                "content": prompt.format(rounds=rounds, history=history)
            }
        ],
        model=llm_model,
        options=options,
        format=Conclusion.model_json_schema()
    )
    return Conclusion.model_validate_json(response.message.content)

def initiate(prompt: str) -> str:
    response : ChatResponse = client.chat(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=llm_model,
        options=options
    )
    return response.message.content

def argue(prompt: str, input: str) -> str:
    response : ChatResponse = client.chat(
        messages=[
            {
                "role": "user",
                "content": prompt.format(input=input)
            }
        ],
        model=llm_model,
        options=options
    )
    return response.message.content

concluded = Conclusion(
    is_concluded = False,
    score = 1.0,
    reasoning = ""
)
rounds : int = 1

history = ""
rhs_msg = ""

while not concluded.is_concluded:
    print(f"rounds={rounds}", "\n", "-------------------------\n", flush=True)

    lhs_msg = argue(prompt_lhs, rhs_msg)
    print(f"{lhs_speaker}: {lhs_msg}", "\n", "-------------------------\n", flush=True)

    history += lhs_speaker + ": " + lhs_msg + "\n"

    rhs_msg = argue(prompt_rhs, lhs_msg)
    print(f"{rhs_speaker}: {rhs_msg}", "\n", "-------------------------\n", flush=True)

    history += rhs_speaker + ": " + rhs_msg + "\n"

    concluded = conclude(prompt_conclude, history, rounds)
    print(f"Arbiter: {concluded}", "\n", "-------------------------\n", flush=True)

    if rounds > max_rounds:
        print("Terminating debate due to timeout.")
        break
    rounds += 1
