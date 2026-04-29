### Docker Desktop (Docker Model Runner)
- previously active use of docker.io/ai/gpt-oss:20B
- switched to docker.io/ai/gemma4:4B
- tried using gemma4:4B for Cline, cannot adjust the 4096k context size, so, 
```
# as of 29 Apr 2026, this command did not help
docker model configure --context-size 8192 <model-name>
# so did this instead:
docker model package --from <base-model> --context-size 131072 <new-model-name>:large-ctx
```

### Podman Desktop (Podman AI Lab)
- as of 29 Apr 2026, performance was not preferred, so uninstalled
