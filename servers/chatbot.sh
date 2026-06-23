#!/bin/sh

WORKDIR="/Users/asantiola/workspaces/playground-ai-ml/src/apps/st-chatbot"
VENV_DIR="/Users/asantiola/workspaces/playground-ai-ml/.venv"
cd "$WORKDIR"
source "$VENV_DIR/bin/activate"

streamlit run st-chatbot.py
