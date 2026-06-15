from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "mlx-community/gemma-4-12B-it-6bit"
model, processor = load(model_path)

# recorded using MacOS Voice Memos, Settings -> Audio Quality -> Lossless
# drag recording to data folder
# brew install ffmpeg
# `ffmpeg -i boses.m4a -ar 16000 -ac 1 boses.wav`
audio_path = "./data/audios/boses.wav"

messages = [
    {
        "role": "user",
        "content": "Describe what you hear in the sound file."
    }
]

# 3. Apply the chat template
# This ensures <|audio|> token wrappers are cleanly placed in the prompt
formatted_prompt = apply_chat_template(
    processor, 
    model.config, 
    messages, 
    num_audios=1
)

# 4. Generate the transcription
# Let mlx_vlm parse and stream the audio internally by passing the file list
output = generate(
    model, 
    processor, 
    formatted_prompt,
    audio=audio_path,
    temperature=0.0      
)

print(output.text)
