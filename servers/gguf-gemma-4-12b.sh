#!/bin/sh

WORKDIR="/Users/asantiola/workspaces/playground-ai-ml/"
cd "$WORKDIR"

set -x
llama-server -hf google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0 --host localhost --port 12434
