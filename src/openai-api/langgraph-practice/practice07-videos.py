from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
import os
import cv2

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
    temperature=1.0,
    extra_body={
        "top_p": 0.95,
        "top_k": 64,
    },
)

def extract_frames_from_mp4(video_path, fps=1):
    """Extracts frames from an MP4 video at a given frames-per-second rate."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate interval to grab frames
    interval = int(round(video_fps / fps))
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % interval == 0:
            # Convert frame to jpeg bytes
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                # Convert to base64 string
                base64_image = base64.b64encode(buffer).decode('utf-8')
                frames.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                })
        count += 1
        
    cap.release()
    return frames

def describe(video_path):
    frames = extract_frames_from_mp4(video_path, fps=2)
    content = [
        {
            "type": "text",
            "text": "Describe the video based on the following frames.",
        },
        *frames
    ]
    message = HumanMessage(content)

    response = llm.invoke([message])
    print(f"\n===== AI RESPONSE =====\n{response.content}\n")

path_plants = os.path.join(workspaces, "playground-ai-ml/data/videos/plants.mp4")

describe(path_plants)
