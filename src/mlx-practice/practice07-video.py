from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from PIL import Image
import cv2

model_path = "mlx-community/gemma-4-12B-it-qat-6bit"
model, processor = load(model_path)

video_path = "./data/videos/plants.mp4"
cap = cv2.VideoCapture(video_path)

video_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {video_fps}")

# Calculate frame interval for half a second (0.5 seconds)
# e.g., if video is 30 FPS, it samples every 15 frames. If 60 FPS, every 30 frames.
frame_interval = max(1, int(video_fps * 0.5))

frames = []
frame_count = 0
max_frames = 16 # Keep this limited to protect unified memory budget

while cap.isOpened() and len(frames) < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Dynamic check: Samples exactly once every half second (see frame_interval)
    if frame_count % frame_interval == 0:
        # Convert OpenCV BGR to PIL RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        frames.append(pil_img)
        
    frame_count += 1

cap.release()
print(f"Extracted {len(frames)} frames from the video.")

messages = [
    {
        "role": "user",
        "content": "Analyze these sequential video frames and describe what action is taking place."
    }
]

formatted_prompt = apply_chat_template(
    processor, 
    model.config, 
    messages, 
    num_images=len(frames)
)

output = generate(
    model, 
    processor, 
    formatted_prompt,
    image=frames,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)

print(output.text)
