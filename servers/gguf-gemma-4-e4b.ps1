$ModelName = "google/gemma-4-E4B-it-qat-q4_0-gguf:Q4_0"
$HostAddress = "localhost"
$PortNumber = "12434"
llama serve -hf $ModelName --host $HostAddress --port $PortNumber
