from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Any, Iterator, List, Optional
from mlx_vlm import load
from mlx_vlm.generate import stream_generate

model_path="mlx-community/gemma-4-12B-it-qat-6bit"

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

class MLXVLChat(BaseChatModel):
    model_path: str
    model: Any = None
    tokenizer: Any = None

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        # Load the model and tokenizer upon initialization
        self.model, self.tokenizer = load(self.model_path)

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Converts LangChain messages into the standard OpenAI/MLX format."""
        formatted = []
        for msg in messages:
            # Map LangChain roles to dictionary roles
            if msg.type == "system":
                role = "system"
            elif msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            else:
                role = "user"
            formatted.append({"role": role, "content": msg.content})
        return formatted

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Implements the streaming logic required by LangChain."""
        # 1. Format messages for the tokenizer
        mlx_messages = self._convert_messages(messages)
        
        # 2. Apply the chat template (crucial: add_generation_prompt=True)
        prompt = self.tokenizer.apply_chat_template(
            mlx_messages, tokenize=False, add_generation_prompt=True
        )

        # 3. Stream tokens from MLX and yield them as LangChain chunks
        for result in stream_generate(self.model, self.tokenizer, prompt):
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=result.text)
            )
            yield chunk

    # LangChain requires this abstract method to be defined, even if we mostly use stream
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("For simplicity, this wrapper currently only supports .stream()")

    @property
    def _llm_type(self) -> str:
        return "mlx_vlm_local"

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

llm = MLXVLChat(model_path=model_path)

messages = [
    SystemMessage(content=system_prompt), 
    HumanMessage(content=puzzle_prompt_einstein)
]

print("Thinking...\n")

for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
print("")
