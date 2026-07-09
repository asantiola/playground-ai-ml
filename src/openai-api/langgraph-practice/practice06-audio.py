from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from practice00_common import selection
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
    model="mlx-community/gemma-4-12B-it-qat-6bit",
    base_url=openai_base_url,
    api_key=api_key,
    temperature=0.1,
    extra_body={
        "top_p": 0.95,
        "top_k": 64,
    },
)

def encode_audio(audio_path):
    """Convert an audio file to a base64 string."""
    with open(audio_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

prompt = """Describe what you hear in this audio file thoroughly.
    If you hear words, print all you understand. If it is not in English, translate it.
    If it is a puzzle, approach problems step-by-step, verify boundary conditions, 
    and rigorously check your assumptions before calculating the final answer.
"""

def describe(audio_path):
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt,
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

    print("\n===== AI RESPONSE =====\n")
    for chunk in llm.stream([message]):
        print(chunk.content, end="", flush=True)

# recorded using MacOS Voice Memos, Settings -> Audio Quality -> Lossless
# drag recording to data folder
# brew install ffmpeg
# `ffmpeg -i boses.m4a -ar 16000 -ac 1 boses.wav`
# `ffmpeg -i boses.m4a -ar 16000 -ac 1 boses.mp3`

path_brown_fox = os.path.join(workspaces, "playground-ai-ml/data/audios/brown_fox.wav")
path_itik = os.path.join(workspaces, "playground-ai-ml/data/audios/itik.mp3")
path_puzzle = os.path.join(workspaces, "playground-ai-ml/data/audios/puzzle.mp3")

audios_names = [
    "brown fox",
    "itik",
    "puzzle",
]

audios = [
    path_brown_fox,
    path_itik,
    path_puzzle,
]

what = selection("audio", audios_names, audios)
describe(what)
