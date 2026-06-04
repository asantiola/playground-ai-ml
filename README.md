# playground-ai-ml

### install python version in MacOS
```
brew install python@3.12
```

### create virtual env
- ctrl-shift-P, Python: Create Environment
- python3 -m venv .venv; source .venv/bin/activate;

### Install libraries
- pip install
```
pip install --upgrade pip
pip install --upgrade pandas numpy matplotlib
pip install --upgrade langchain langchain-openai langchain-community langchain-huggingface langchain_chroma
pip install --upgrade langgraph langgraph_supervisor
pip install --upgrade openai chroma
pip install --upgrade opencv-python-headless
pip install --upgrade mcp langchain-mcp-adapters
pip install --upgrade streamlit
```
- https://pytorch.org/
    - CUDA (XPS 9500 / CUDA 12.6)
    ```
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```
    - No GPU
    ```
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```
    - MPS (Macbook Pro M4 Pro)
    ```
    pip install --upgrade torch torchvision torchaudio
    ```
- https://www.tensorflow.org/api_docs/python/tf
    - Metal: I had to use Python 3.12.12!
    ```
    pip install tensorflow-macos tensorflow-metal
    ```

### VSCode extensions
- ms-python.python
- ms-toolsai.jupyter
- saoudrizwan.claude-dev

### Automatically activate venv if present
- add in ~/.bashrc
```
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
```
- set vscode setting:
```
"python.terminal.activateEnvironment": false
```

### For local models
- Docker Model Runner
    - http://localhost:12434/engines/v1
    - http://model-runner.docker.internal/engines/v1
