from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
import os

workspaces = os.environ.get(
    "WORKSPACES",
    "/workspaces"
)

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/engines/v1"
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
