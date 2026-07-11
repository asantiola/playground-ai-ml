#!/bin/sh

WORKDIR="/Users/asantiola/workspaces/playground-ai-ml/"
VENV_DIR="/Users/asantiola/workspaces/playground-ai-ml/.venv"
cd "$WORKDIR"
source "$VENV_DIR/bin/activate"

MODEL=${1:-"mlx-community/gemma-4-12B-it-qat-6bit"}

set -x
python -m mlx_vlm.server --model $MODEL --host localhost --port 12434 --max-kv-size 16384
