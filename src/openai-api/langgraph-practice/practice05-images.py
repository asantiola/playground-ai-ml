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
    model="mlx-community/gemma-4-12B-it-6bit",
    base_url=openai_base_url,
    api_key=api_key,
    temperature=0.0,
)

def encode_image(image_path):
    """Convert an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

prompt = """Give a brief description of what you see, then do the following:
- If you see text, print what you see. If it is not in English, translate it.
- If it is a puzzle, approach problems step-by-step, verify boundary conditions, 
and rigorously check your assumptions before calculating the final answer.
"""

def describe(image_path):
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                },
            },
        ]
    )

    print("\n===== AI RESPONSE =====\n")
    for chunk in llm.stream([message]):
        print(chunk.content, end="", flush=True)

path_vulture = os.path.join(workspaces, "playground-ai-ml/data/images/vulture.jpg")
path_screenshot = os.path.join(workspaces, "playground-ai-ml/data/images/screenshot-sample.png")
path_handwriting = os.path.join(workspaces, "playground-ai-ml/data/images/handwriting.jpg")
path_meme = os.path.join(workspaces, "playground-ai-ml/data/images/meme.jpg")
path_logic_puzzle = os.path.join(workspaces, "playground-ai-ml/data/images/logic_puzzle.jpg")

images_names = [
    "vulture",
    "screenshot",
    "handwriting",
    "meme",
    "logic puzzle",
]

images = [
    path_vulture,
    path_screenshot,
    path_handwriting,
    path_meme,
    path_logic_puzzle,
]

what = selection("image", images_names, images)
describe(what)
