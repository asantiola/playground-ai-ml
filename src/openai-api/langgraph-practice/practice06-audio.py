from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import os

workspaces = os.environ.get(
    "WORKSPACES",
    "/workspaces"
)

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

llm = ChatOpenAI(
    model="mlx-community/gemma-4-12B-it-6bit",
    base_url=openai_base_url,
    api_key=api_key,
)

def encode_audio(audio_path):
    """Convert an audio file to a base64 string."""
    with open(audio_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe(audio_path):
    system_prompt = (
        "You are an expert audio analyst and acoustic engineer. Your task is to analyze "
        "audio clips with precise attention to detail, covering speech content, vocal characteristics, "
        "background environment, and audio quality."
    )
    human_prompt = (
        "What is being said in the audio file? If not in English, translate it too."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": human_prompt,
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
    ]

    response = llm.invoke(messages)
    print(f"\n===== AI RESPONSE =====\n{response.content}\n")

# recorded using MacOS Voice Memos, Settings -> Audio Quality -> Lossless
# drag recording to data folder
# brew install ffmpeg
# `ffmpeg -i boses.m4a -ar 16000 -ac 1 boses.wav`

path_boses = workspaces + "/playground-ai-ml/data/audios/boses.wav"
path_bahay_kubo = workspaces + "/playground-ai-ml/data/audios/bahay_kubo.wav"

describe(path_boses)
describe(path_bahay_kubo)
