#!/usr/bin/env bash
# Build a single executable with models and static assets embedded (PyInstaller).
# Output: dist/video-picker-linux | dist/video-picker-macos
# Run on the target OS (Linux or macOS).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

OS="$(uname -s)"
case "${OS}" in
  Linux) OUT_NAME="video-picker-linux" ;;
  Darwin) OUT_NAME="video-picker-macos" ;;
  *)
    echo "error: build_onefile.sh supports Linux and macOS only (got ${OS})" >&2
    echo "       On Windows run: scripts\\build_onefile.ps1" >&2
    exit 1
    ;;
esac

if [[ ! -f "${ROOT}/models/md_v5a_1_3_640_640_static.onnx" ]]; then
  echo "error: ${ROOT}/models/ is missing (ONNX weights required)" >&2
  exit 1
fi

if [[ ! -d "${ROOT}/.venv" ]]; then
  echo "error: create a venv first: uv venv && uv pip install -e ." >&2
  exit 1
fi

PY="${ROOT}/.venv/bin/python"
echo "==> Installing PyInstaller"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "${PY}" -q pyinstaller
else
  "${PY}" -m pip install -q pyinstaller
fi

echo "==> Building one-file executable (this may take several minutes)"
"${PY}" -m PyInstaller "${ROOT}/packaging/video_picker.spec" --noconfirm --clean

BUILT="${ROOT}/dist/video-picker"
FINAL="${ROOT}/dist/${OUT_NAME}"
rm -f "${FINAL}"
mv "${BUILT}" "${FINAL}"
chmod +x "${FINAL}"

echo ""
echo "Build complete:"
echo "  ${FINAL}"
echo "  size: $(du -h "${FINAL}" | cut -f1)"
echo ""
echo "Run: ${FINAL}"
