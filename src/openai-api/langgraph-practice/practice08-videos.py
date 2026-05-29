from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
import os
import cv2

llm = ChatOpenAI(
    model="ai/gemma4:E4B",
    base_url="http://model-runner.docker.internal/engines/v1",
    api_key="docker",
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

HOME=os.environ["HOME"]
path_plants = HOME + "/repo/playground-ai-ml/data/videos/plants.mp4"

describe(path_plants)
