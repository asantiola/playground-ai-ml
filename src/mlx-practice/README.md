# Practice using mlx-lm python package
- having issues with prompts?

### Install
```
pip install --upgrade mlx-vlm mlx-lm mlx
```

### Uninstall
```
pip uninstall mlx-vlm mlx-lm mlx
```

### sample CLI
```
mlx_lm.generate --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --prompt "What is the capital of France?"
mlx_lm.generate --model mlx-community/gpt-oss-20b-MXFP4-Q4 --prompt "What is the capital of France?"
```

### starting the server
```
mlx_lm.server --help
mlx_lm.server --port 12434 --model mlx-community/gpt-oss-20b-MXFP4-Q4
mlx_vlm.server --port 12434 --model mlx-community/gemma-4-e4b-it-4bit
```
