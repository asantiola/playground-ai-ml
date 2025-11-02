# playground-ai-ml

### create virtual env
- ctrl-shift-P, Python: Create Environment
- python3 -m venv .venv; source .venv/bin/activate;

### Install libraries
- pip install --upgrade pip
- pip install pandas numpy matplotlib
- pip install langchain langchain-openai langchain-community langchain-huggingface langchain_chroma
- pip install openai chroma
- https://pytorch.org/
    - CUDA (XPS 9500 / CUDA 12.6)
        - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    - No GPU
        - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    - MPS (Macbook Pro M4 Pro)
        - pip install torch torchvision torchaudio
- https://www.tensorflow.org/api_docs/python/tf
    - Metal: I had to use Python 3.12.12!
        - pip install tensorflow-macos tensorflow-metal

### VSCode extensions
- ms-python.python
- ms-python.vscode-pylance
- ms-python.debugpy
- ms-toolsai.jupyter
