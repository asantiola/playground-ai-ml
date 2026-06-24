#!/bin/sh

WORKDIR="/Users/asantiola/workspaces/playground-ai-ml/"
VENV_DIR="/Users/asantiola/workspaces/playground-ai-ml/.venv"
cd "$WORKDIR"
source "$VENV_DIR/bin/activate"

set -x
python -m mlx_vlm.server --model mlx-community/gemma-4-12B-it-qat-6bit --host localhost --port 12434 --max-kv-size 16384
