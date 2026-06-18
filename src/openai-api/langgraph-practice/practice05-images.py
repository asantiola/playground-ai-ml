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
)

def encode_image(image_path):
    """Convert an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe(image_path):
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe this image. If you see text, print what you read. If it is not in English, translate it too.",
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

images_names = [
    "vulture",
    "screenshot",
    "handwriting",
    "meme",
]

images = [
    path_vulture,
    path_screenshot,
    path_screenshot,
    path_meme,
]

what = selection("image", images_names, images)
describe(what)
