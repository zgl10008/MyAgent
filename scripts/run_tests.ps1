# 一键运行单元测试（Windows PowerShell）
# 用法: powershell -ExecutionPolicy Bypass -File scripts/run_tests.ps1

$ErrorActionPreference = "Stop"

# 定位项目根目录
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# 测试环境变量（避免本机 DEBUG=release 等环境变量干扰）
$env:DEBUG = "false"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "==> Running pytest in $ProjectRoot"
python -m pytest

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nAll tests passed. (exit 0)"
} else {
    Write-Host "`nTests failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
