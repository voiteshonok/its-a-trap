# Build a single video-picker.exe with models and static assets embedded (PyInstaller).
# Output: dist\video-picker-windows.exe
#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ModelsMd = Join-Path $Root "models\md_v5a_1_3_640_640_static.onnx"
if (-not (Test-Path $ModelsMd)) {
    throw "models\ is missing (ONNX weights required)"
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Create a venv first: uv venv; uv pip install -e ."
}

Write-Host "==> Installing PyInstaller"
& $VenvPython -m pip install -q pyinstaller

Write-Host "==> Building one-file executable (this may take several minutes)"
& $VenvPython -m PyInstaller (Join-Path $Root "packaging\video_picker.spec") --noconfirm --clean

$Built = Join-Path $Root "dist\video-picker.exe"
$Final = Join-Path $Root "dist\video-picker-windows.exe"
if (Test-Path $Final) { Remove-Item -Force $Final }
Move-Item -Force $Built $Final

Write-Host ""
Write-Host "Build complete:"
Write-Host "  $Final"
Write-Host "  size: $((Get-Item $Final).Length / 1MB) MB"
Write-Host ""
Write-Host "Run: $Final"
