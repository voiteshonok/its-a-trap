# PyInstaller one-file bundle (models + static embedded).
# Build: scripts/build_onefile.sh   (Linux / macOS)
#        scripts/build_onefile.ps1  (Windows)

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

root = Path(SPECPATH).resolve().parent

md_model = root / "models" / "md_v5a_1_3_640_640_static.onnx"
species_model = root / "models" / "spicesNet_v401a.onnx"
species_labels = root / "static" / "spicesNet_labels_v401a.txtset"

for path, label in (
    (md_model, "MegaDetector ONNX"),
    (species_model, "SpeciesNet ONNX"),
    (species_labels, "SpeciesNet labels"),
):
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")

datas = [
    (str(md_model), "models"),
    (str(species_model), "models"),
    (str(species_labels), "static"),
]
binaries = []
hiddenimports = [
    "video_picker.worker",
    "video_picker.pipeline",
    "video_picker.megadetector_video",
    "video_picker.utils",
    "video_picker.paths",
]

for pkg in ("onnxruntime", "cv2", "numpy"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception:
        hiddenimports += collect_submodules(pkg)

a = Analysis(
    [str(root / "video_picker" / "__main__.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "numpy.tests",
        "numpy.f2py.tests",
        "pytest",
        "tkinter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe_name = "video-picker"
if sys.platform == "win32":
    exe_name = "video-picker.exe"

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=exe_name.replace(".exe", ""),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
)
