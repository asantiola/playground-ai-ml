raise Exception("not yet supported by Docker Desktop?")

from openai import OpenAI
import base64
import os

client = OpenAI(
    base_url="http://model-runner.docker.internal/engines/diffusers/v1",
    api_key = "docker"
)

response = client.images.generate(
    model="ai/stable-diffusion:latest",
    prompt="A serene cinematic shot of a of meerkat on a warthig.",
    size="512x512",
    response_format="b64_json"
)

image_b64 = response.data[0].b64_json
images_bytes = base64.b64decode(image_b64)

HOME=os.environ["HOME"]
output_name = HOME + "/playground-ai-ml/data/images/generated.png"
with open(output_name, "wb") as f:
    f.write(images_bytes)

# # causing error:
# openai.InternalServerError: unable to load runner: error waiting for runner to be ready: 
# Diffusers terminated unexpectedly: Diffusers failed: sandbox-exec: empty subpath pattern
