$ErrorActionPreference = "Stop"
$modelDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelPath = Join-Path $modelDir "model_int8.onnx"

if (Test-Path $modelPath) {
    Write-Host "Model already exists at $modelPath"
    exit 0
}

$url = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_int8.onnx"
Write-Host "Downloading nomic-embed-text-v1.5 INT8 ONNX model (137MB)..."
Invoke-WebRequest -Uri $url -OutFile $modelPath
Write-Host "Downloaded to $modelPath"