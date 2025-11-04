import torch

print(f"PyTorch version: {torch.__version__}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

if device == "mps":
    x = torch.rand(size=(3, 4)).to(device)
    print(f"Tensor on MPS device: {x}")
