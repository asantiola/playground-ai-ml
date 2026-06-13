from typing import Any, Iterator, List, Optional, Union, Type
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from mlx_vlm import load, generate, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
import base64
import os
import json
import re
import tempfile

model_path="mlx-community/gemma-4-12B-it-6bit"

class MLXVLChat(BaseChatModel):
    model_path: str
    model: Any = None
    processor: Any = None

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        self.model, self.processor = load(self.model_path)

    def _convert_messages(self, messages: List[BaseMessage]) -> tuple[List[dict], List[str]]:
        """
        Converts LangChain messages into standard MLX-VLM template structures
        and extracts base64 or file-path images.
        """
        formatted = []
        image_paths = []
        
        for msg in messages:
            if msg.type == "system":
                role = "system"
                content = msg.content
            elif msg.type == "ai":
                role = "assistant"
                content = msg.content
            else:
                role = "user"
                
                if isinstance(msg.content, list):
                    text_pieces = []
                    for block in msg.content:
                        if block.get("type") == "text":
                            text_pieces.append(block.get("text", ""))
                        elif block.get("type") == "image_url":
                            img_url = block.get("image_url", {}).get("url", "")
                            
                            if img_url.startswith("data:image"):
                                base64_data = img_url.split(",")[1]
                                img_bytes = base64.b64decode(base64_data)
                                
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                                    tmp_file.write(img_bytes)
                                    image_paths.append(tmp_file.name)
                            else:
                                image_paths.append(img_url)
                    
                    content = " ".join(text_pieces)
                else:
                    content = msg.content

            formatted.append({"role": role, "content": content})
            
        return formatted, image_paths

    def _generate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        **kwargs: Any
    ) -> ChatResult:
        mlx_messages, image_paths = self._convert_messages(messages)
        
        formatted_prompt = apply_chat_template(
            self.processor,
            self.model.config,
            mlx_messages,
            num_images=len(image_paths)
        )
        
        active_image = image_paths[0] if image_paths else None
        
        output_obj = generate(
            self.model,
            self.processor,
            formatted_prompt,
            image=active_image,
            temperature=kwargs.get("temperature", 0.0)
        )
        
        if hasattr(output_obj, "text"):
            full_text = output_obj.text
        elif isinstance(output_obj, str):
            full_text = output_obj
        else:
            full_text = str(output_obj)
        
        if active_image and "tmp" in active_image:
            try:
                os.remove(active_image)
            except OSError:
                pass

        tool_calls = []
        final_content = full_text
        
        if kwargs.get("bound_tools"):
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
                    final_content = ""
                except json.JSONDecodeError:
                    pass

            if not tool_calls:
                xml_match = re.search(
                    r"<\|tool_call\>call:(\w+)\s*\{(.*?)\}\s*<tool_call\|>", 
                    full_text, 
                    re.DOTALL
                )
                if xml_match:
                    tool_name = xml_match.group(1)
                    raw_args = xml_match.group(2).strip()
                    try:
                        valid_json_args = re.sub(r'(\s*?)(\w+)\s*:', r'\1"\2":', raw_args)
                        args = json.loads(f"{{{valid_json_args}}}")
                    except json.JSONDecodeError:
                        args = {"location": raw_args.replace('"', '').strip()}

                    tool_calls.append({
                        "name": tool_name,
                        "args": args,
                        "id": "call_gemma_xml_0",
                    })
                    final_content = ""

        generation = ChatGeneration(
            message=AIMessage(
                content=final_content, 
                tool_calls=tool_calls,
                id="mlx_vlm_generation"
            )
        )
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Implements actual real-time token streaming with vision support."""

        mlx_messages, image_paths = self._convert_messages(messages)
        
        formatted_prompt = apply_chat_template(
            self.processor,
            self.model.config,
            mlx_messages,
            num_images=len(image_paths)
        )
        
        active_image = image_paths[0] if image_paths else None
        
        for response_chunk in stream_generate(
            self.model,
            self.processor,
            formatted_prompt,
            image=active_image,
        ):
            # Extract text from mlx_vlm chunk structure safely
            if hasattr(response_chunk, "text"):
                chunk_text = response_chunk.text
            elif isinstance(response_chunk, str):
                chunk_text = response_chunk
            else:
                chunk_text = str(response_chunk)
                
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))

        # 4. Cleanup temporary image files after stream completes
        if active_image and "tmp" in active_image:
            try:
                os.remove(active_image)
            except OSError:
                pass

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> Runnable[Any, Any]:
        openai_tools = [convert_to_openai_tool(t) for t in tools]
        tool_instruction = (
            "You have access to the following tools. If you need to use a tool, you MUST "
            "respond ONLY with a JSON list containing the tool calls, using this format:\n"
            '[{"name": "tool_name", "arguments": {"arg_name": "value"}}]\n'
            "If no tool is needed or applicable, answer normally with standard text.\n\n"
            f"Available tools:\n{json.dumps(openai_tools, indent=2)}"
        )

        def transform_prompt_for_tools(input_args: Any) -> List[BaseMessage]:
            messages = input_args.to_messages() if hasattr(input_args, "to_messages") else input_args
            if messages and messages[0].type == "system":
                messages[0].content += f"\n\n{tool_instruction}"
            else:
                messages.insert(0, SystemMessage(content=tool_instruction))
            return messages

        model_runner = self.bind(bound_tools=openai_tools)
        return transform_prompt_for_tools | model_runner

    def with_structured_output(self, schema: Union[Type[BaseModel], dict], **kwargs: Any) -> Runnable[Any, Any]:
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

workspaces = os.environ.get(
    "WORKSPACES",
    "/workspaces"
)

llm = MLXVLChat(model_path=model_path)

def encode_image(image_path):
    """Convert an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe(image_path):
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe this image. If you see text, print what you read.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                },
            },
        ]
    )

    response = llm.invoke([message])
    print(f"\n===== AI RESPONSE =====\n{response.content}\n")

path_vulture = workspaces + "/playground-ai-ml/data/images/vulture.jpg"
path_screenshot = workspaces + "/playground-ai-ml/data/images/screenshot-sample.png"
path_handwriting = workspaces + "/playground-ai-ml/data/images/handwriting.jpg"
path_meme = workspaces + "/playground-ai-ml/data/images/meme.jpg"

describe(path_vulture)
describe(path_screenshot)
describe(path_handwriting)
describe(path_meme)
