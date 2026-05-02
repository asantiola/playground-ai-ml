from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
import os

llm = ChatOpenAI(
    model="ai/gemma4:4B-128k",
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
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
                "text": "Describe this image.",
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

HOME=os.environ["HOME"]
path_armadillo = HOME + "/repo/playground-ai-ml/data/images/armadillo.jpg"
path_vulture = HOME + "/repo/playground-ai-ml/data/images/vulture.jpg"

describe(path_armadillo)
describe(path_vulture)
