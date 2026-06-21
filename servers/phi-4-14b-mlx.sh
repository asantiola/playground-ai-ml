#!/bin/sh

WORKDIR="/Users/asantiola/workspaces/playground-ai-ml/"
VENV_DIR="/Users/asantiola/workspaces/playground-ai-ml/.venv"
cd "$WORKDIR"
source "$VENV_DIR/bin/activate"

set -x
python -m mlx_lm server --model mlx-community/phi-4-6bit --host localhost --port 12434 --max-tokens 8192
