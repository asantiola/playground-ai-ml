from langchain_core.tools import tool
from typing import Any, Iterator, List, Optional, Type, Union
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessageChunk, AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult, ChatGeneration
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from mlx_vlm import load
from mlx_vlm.generate import stream_generate
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages.tool import ToolCall
import json
import re

model_path="mlx-community/gemma-4-12B-it-6bit"

class MLXVLChat(BaseChatModel):
    model_path: str
    model: Any = None
    tokenizer: Any = None

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        self.model, self.tokenizer = load(self.model_path)

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Converts LangChain messages into standard OpenAI/MLX format."""
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
        mlx_messages = self._convert_messages(messages)
        prompt = self.tokenizer.apply_chat_template(
            mlx_messages, tokenize=False, add_generation_prompt=True
        )

        for result in stream_generate(self.model, self.tokenizer, prompt):
            yield ChatGenerationChunk(message=AIMessageChunk(content=result.text))

    def _generate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        **kwargs: Any
    ) -> ChatResult:
        full_text = ""
        for chunk in self._stream(messages, stop=stop, **kwargs):
            full_text += chunk.message.content
        
        tool_calls = []
        final_content = full_text  # Default to returning full text

        if kwargs.get("bound_tools"):
            # 1. Try extracting standard JSON array format first
            json_match = re.search(r"\[\s*\{.*\}\s*\]", full_text, re.DOTALL)
            if json_match:
                try:
                    raw_calls = json.loads(json_match.group(0))
                    for i, call in enumerate(raw_calls):
                        tool_calls.append({
                            "name": call.get("name"),
                            "args": call.get("arguments", {}),
                            "id": f"call_{i}",
                            "type": "tool_call"
                        })
                    final_content = ""  # Clear content if it's purely a tool call
                except json.JSONDecodeError:
                    pass

            # 2. Gemma-4 XML-style Tag Fallback
            if not tool_calls:
                # Corrected tokens: matches <|tool_call>call:tool_name{...}<tool_call|>
                xml_match = re.search(
                    r"<\|tool_call\>call:(\w+)\s*\{(.*?)\}\s*<tool_call\|>", 
                    full_text, 
                    re.DOTALL
                )
                if xml_match:
                    tool_name = xml_match.group(1)
                    raw_args = xml_match.group(2).strip()
                    
                    # Robust pseudo-JSON converter (converts key: "value" to "key": "value")
                    try:
                        # Fixes unquoted keys
                        valid_json_args = re.sub(r'(\s*?)(\w+)\s*:', r'\1"\2":', raw_args)
                        args = json.loads(f"{{{valid_json_args}}}")
                    except json.JSONDecodeError:
                        args = {"location": raw_args.replace('"', '').strip()} # Fallback

                    # CRITICAL: LangChain expects tool calls to match the standard ToolCall interface 
                    tool_calls.append({
                        "name": tool_name,
                        "args": args,
                        "id": "call_gemma_xml_0", # A valid string ID
                    })
                    final_content = ""  # Wipe the content so LangChain treats this cleanly as a tool call

        generation = ChatGeneration(
            message=AIMessage(
                content=final_content, 
                tool_calls=tool_calls,
                id=f"lc_run--{re.sub(r'[^a-f0-9-]', '', str(dir))[:8]}-0" # Mocking standard trace ID syntax
            )
        )
        return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: List[Any],
        **kwargs: Any
    ) -> Runnable[Any, Any]:
        """
        Binds tools to the model by formatting their definitions into system 
        prompts and enforcing JSON structural generation layouts.
        """
        # Convert all standard LangChain tools/functions into standardized schemas
        openai_tools = [convert_to_openai_tool(t) for t in tools]
        
        # Build tool instructions
        tool_instruction = (
            "You have access to the following tools. If you need to use a tool, you MUST "
            "respond ONLY with a JSON list containing the tool calls, using this format:\n"
            '[{"name": "tool_name", "arguments": {"arg_name": "value"}}]\n'
            "If no tool is needed or applicable, answer normally with standard text.\n\n"
            f"Available tools:\n{json.dumps(openai_tools, indent=2)}"
        )

        def transform_prompt_for_tools(input_args: Any) -> List[BaseMessage]:
            messages = input_args.to_messages() if hasattr(input_args, "to_messages") else input_args
            
            # Inject structural system rules to the conversation matrix
            if messages and messages[0].type == "system":
                messages[0].content += f"\n\n{tool_instruction}"
            else:
                messages.insert(0, SystemMessage(content=tool_instruction))
            return messages

        # Bind parameters down to _generate dynamically via the Runnable pipeline
        model_runner = self.bind(bound_tools=openai_tools)
        return transform_prompt_for_tools | model_runner

    def with_structured_output(
        self,
        schema: Union[Type[BaseModel], dict],
        **kwargs: Any
    ) -> Runnable[Any, Any]:
        from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
        
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            parser = PydanticOutputParser(pydantic_object=schema)
        else:
            parser = JsonOutputParser()

        def transform_prompt(input_args: Any) -> List[BaseMessage]:
            messages = input_args.to_messages() if hasattr(input_args, "to_messages") else input_args
            format_instructions = parser.get_format_instructions()
            
            if messages and messages[0].type == "system":
                messages[0].content += f"\n\n{format_instructions}"
            else:
                messages.insert(0, SystemMessage(content=format_instructions))
            return messages

        return transform_prompt | self | parser

    @property
    def _llm_type(self) -> str:
        return "mlx_vlm_local"

@tool
def get_weather(location: str):
    """Use this to get the weather for a specific location."""
    if "sf" in location.lower() or "san francisco" in location.lower():
        return "It's sunny and 20°C in San Francisco."
    else:
        return "I don't know the weather there."

@tool
def get_stock_price(ticker: str):
    """Use this to get the stock price for a company."""
    return f"The stock price of {ticker} is $150."

# Create a list of tools
tools = [get_weather, get_stock_price]

llm = MLXVLChat(model_path=model_path)

llm_with_tools = llm.bind_tools(tools)

def invoke(question: str):
    answer = llm_with_tools.invoke([
        HumanMessage(content=question)
    ])
    print(f"question: {question}\nanswer: {answer}\n\n")

invoke("What is the weather in Manila?")
invoke("What is the capital of New Zealand?")
