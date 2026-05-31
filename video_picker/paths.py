"""Paths for dev installs and PyInstaller one-file bundles."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_MD_MODEL = "md_v5a_1_3_640_640_static.onnx"
_SPECIES_MODEL = "spicesNet_v401a.onnx"
_SPECIES_LABELS = "spicesNet_labels_v401a.txtset"
_APP_ICON = "icon.png"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def resource_root() -> Path:
    """Directory containing bundled models/ and static/ (or project root in dev)."""
    env = os.environ.get("VIDEO_PICKER_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def install_root() -> Path:
    """Project / bundle root (used to chdir in dev directory installs)."""
    return resource_root()


def megadetector_model_path() -> Path:
    return resource_root() / "models" / _MD_MODEL


def speciesnet_model_path() -> Path:
    return resource_root() / "models" / _SPECIES_MODEL


def speciesnet_labels_path() -> Path:
    return resource_root() / "static" / _SPECIES_LABELS


def app_icon_path() -> Path:
    return resource_root() / "static" / _APP_ICON
