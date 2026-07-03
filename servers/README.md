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
llama-server -hf microsoft/phi-4-gguf:Q4_K_S \
    --host localhost \
    --port 12434 \
    --n-gpu-layers all \
    --flash-attn on \
    --parallel 1 \
    --ctx-size 16384 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0
```
- tried these models
    - google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0
    - microsoft/phi-4-gguf:Q4_0 

### MLX notes
## MTP Gemma4
```
python -m mlx_vlm.server --model mlx-community/gemma-4-12B-it-6bit --draft-model mlx-community/gemma-4-12B-it-assistant-6bit --draft-kind mtp --host localhost --port 12434 --max-kv-size 16384
```

## Phi4
```
python -m mlx_lm server --model mlx-community/phi-4-6bit --host localhost --port 12434 --max-tokens 8192
```

### llama notes
## gpt-oss
```
llama-server -hf ggml-org/gpt-oss-20b-GGUF --host localhost --port 12434 -c 0 --jinja
```
