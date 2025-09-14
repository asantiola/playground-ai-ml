# playground-ai-ml

### create virtual env
- ctrl-shift-P, Python: Create Environment
- python3 -m venv .venv; source .venv/bin/activate;

### Install libraries
- pip install --upgrade pip
- pip install pandas numpy matplotlib
- pip install openai
- pip install langchain langchain-openai
- https://pytorch.org/
    - CUDA (XPS 9500 / CUDA 12.6)
        - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    - No GPU
        - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    - MPS (Macbook Pro M4 Pro)
        - pip install torch torchvision torchaudio

### VSCode extensions
- ms-python.python
- ms-python.vscode-pylance
- ms-python.debugpy
- ms-toolsai.jupyter
