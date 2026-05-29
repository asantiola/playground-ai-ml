from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
import os

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://model-runner.docker.internal/engines/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

llm = ChatOpenAI(
    model="ai/gemma4:E4B",
    base_url=openai_base_url,
    api_key=api_key,
)

def encode_audio(audio_path):
    """Convert an audio file to a base64 string."""
    with open(audio_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe(audio_path):
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe this audio file.",
            },
            {
                "type": "input_audio",
                "input_audio": {
                    "data": encode_audio(audio_path),
                    "format": "mp3", # OpenAI strictly requires "wav" or "mp3"
                },
            },
        ]
    )

    response = llm.invoke([message])
    print(f"\n===== AI RESPONSE =====\n{response.content}\n")

HOME=os.environ["HOME"]
path_quickbrownfox = HOME + "/playground-ai-ml/data/audios/QuickBrownFox.mp3"

describe(path_quickbrownfox)
