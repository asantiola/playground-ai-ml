$WORKDIR = "C:\Users\asant\workspaces\playground-ai-ml\src\apps\st-chatbot"
$VENV_DIR = "C:\Users\asant\workspaces\playground-ai-ml\.venv"

Set-Location $WORKDIR
& "$VENV_DIR\Scripts\Activate.ps1"

streamlit run st-chatbot.py
