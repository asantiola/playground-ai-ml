from typing import Any, Iterator, List, Optional, Type, Union
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessageChunk, AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult, ChatGeneration
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from mlx_vlm import load
from mlx_vlm.generate import stream_generate

model_path="mlx-community/gemma-4-12B-it-6bit"

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
        mlx_messages = self._convert_messages(messages)
        prompt = self.tokenizer.apply_chat_template(
            mlx_messages, tokenize=False, add_generation_prompt=True
        )

        for result in stream_generate(self.model, self.tokenizer, prompt):
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=result.text)
            )
            yield chunk

    def _generate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        **kwargs: Any
    ) -> ChatResult:
        """Required by LangChain for invoke() and structured output workflows."""
        # Accumulate stream into a single string for sync invocations
        full_text = ""
        for chunk in self._stream(messages, stop=stop, **kwargs):
            full_text += chunk.message.content
        
        generation = ChatGeneration(message=AIMessage(content=full_text))
        return ChatResult(generations=[generation])

    def with_structured_output(
        self,
        schema: Union[Type[BaseModel], dict],
        **kwargs: Any
    ) -> Runnable[Any, Any]:
        """
        Binds a Pydantic schema or JSON schema to the model.
        Forces the model to output JSON and parses it back into the schema.
        """
        from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
        
        # 1. Determine the parser based on input type
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            parser = PydanticOutputParser(pydantic_object=schema)
        else:
            parser = JsonOutputParser()

        # 2. Inject instructions into the system prompt via a Runnable pipeline
        def transform_prompt(input_args: Any) -> List[BaseMessage]:
            # Expecting messages or a prompt template output
            messages = input_args.to_messages() if hasattr(input_args, "to_messages") else input_args
            
            # Append JSON formatting instructions to the first system message, or inject one
            format_instructions = parser.get_format_instructions()
            
            if messages and messages[0].type == "system":
                messages[0].content += f"\n\n{format_instructions}"
            else:
                messages.insert(0, SystemMessage(content=format_instructions))
                
            return messages

        # 3. Chain: Inject Schema -> Run Model -> Parse Output JSON
        return transform_prompt | self | parser

    @property
    def _llm_type(self) -> str:
        return "mlx_vlm_local"

llm = MLXVLChat(model_path=model_path)

class Response(BaseModel):
    answer: str = Field(
        description="The answer to the question"
    )

    confidence_score: float = Field(
        description="Score on how confident you are on the answer, from 0 to 1."
    )

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("""
            You are a helpful assistant that answers the input.
            Keep your answer brief.
        """),
        HumanMessagePromptTemplate.from_template("Input: {input}"),
    ]
)

chain = (prompt | llm.with_structured_output(schema=Response))

response = chain.invoke({ "input": "What is the capital of France?" })
print(f"response:\n{response}\n")

response = chain.invoke({ "input": "How do you measure your confidence in your answer?" })
print(f"response:\n{response}\n")
