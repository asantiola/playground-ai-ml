# Running servers:

## mlx_vlm.server
### Usual setup
```
python -m mlx_vlm.server \
    --host localhost \
    --port 12434 \
    --max-kv-size 16384 \
    --model mlx-community/gemma-4-12B-it-6bit
```
### Using mtp
```
python -m mlx_vlm.server \
    --host localhost \
    --port 12434 \
    --max-kv-size 16384 \
    --model mlx-community/gemma-4-12B-it-6bit \
    --draft-model mlx-community/gemma-4-12B-it-assistant-6bit \
    -draft-kind mtp
```

## mlx_lm.server
```
python -m mlx_lm server \
    --host localhost \
    --port 12434 \
    --max-tokens 8192 \
    --model mlx-community/phi-4-6bit
```

## llama-server
```
brew install llama.cpp
llama-server -p 12434 \
    -hf google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0 \
    -ngl -1 \
    -t 0 -tb 0 \
    -fa on \
    --parallel 1 \
    -c 16384 \
    -ctk q4_0 -ctv q4_0
```
- tried these models
    - google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0
    - microsoft/phi-4-gguf:Q4_0 
