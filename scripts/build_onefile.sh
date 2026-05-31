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

ICON_NAME="video-picker"
ICON_SRC="${ROOT}/static/icon.png"
SHARE_ICONS="${ROOT}/dist/share/icons/hicolor/256x256/apps"
SHARE_APPS="${ROOT}/dist/share/applications"
mkdir -p "${SHARE_ICONS}" "${SHARE_APPS}"
cp "${ICON_SRC}" "${SHARE_ICONS}/${ICON_NAME}.png"
cp "${ICON_SRC}" "${ROOT}/dist/${ICON_NAME}.png"
cat > "${SHARE_APPS}/${ICON_NAME}.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Video Picker
Comment=Run MegaDetector on videos
Exec=${FINAL} %F
Icon=${ICON_NAME}
Terminal=false
Categories=Utility;Science;
StartupWMClass=${ICON_NAME}
EOF
cp "${SHARE_APPS}/${ICON_NAME}.desktop" "${ROOT}/dist/${ICON_NAME}.desktop"
chmod +x "${ROOT}/dist/${ICON_NAME}.desktop"

if command -v gio >/dev/null 2>&1; then
  gio set "${FINAL}" metadata::custom-icon "file://${ROOT}/dist/${ICON_NAME}.png" 2>/dev/null || true
  gio set "${ROOT}/dist/${ICON_NAME}.desktop" metadata::trusted true 2>/dev/null || true
fi

echo ""
echo "Build complete:"
echo "  ${FINAL}"
echo "  ${ROOT}/dist/${ICON_NAME}.desktop  (launcher with icon in file manager)"
echo "  size: $(du -h "${FINAL}" | cut -f1)"
echo ""
echo "Run: ${FINAL}"
echo "Or:  gtk-launch ${ICON_NAME}   (after: xdg-desktop-menu install --novendor dist/share/applications/${ICON_NAME}.desktop)"
