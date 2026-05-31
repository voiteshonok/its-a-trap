"""Linux desktop / dock icon integration and Qt icon loading."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication, QIcon, QPixmap

from video_picker.paths import app_icon_path, install_root, is_frozen

DESKTOP_ID = "video-picker"
STARTUP_WM_CLASS = "video-picker"


def _icon_source() -> Path | None:
    path = app_icon_path()
    return path if path.is_file() else None


def _portable_share_root() -> Path | None:
    base = Path(sys.executable).resolve().parent if is_frozen() else install_root()
    share = base / "share"
    desktop = share / "applications" / f"{DESKTOP_ID}.desktop"
    return share if desktop.is_file() else None


def setup_before_qt_app() -> None:
    if sys.platform != "linux":
        return
    os.environ.setdefault("WAYLAND_APP_ID", DESKTOP_ID)
    share = _portable_share_root()
    if share is None:
        return
    data_dir = str(share)
    existing = os.environ.get("XDG_DATA_DIRS", "/usr/share:/usr/local/share")
    if data_dir not in existing.split(":"):
        os.environ["XDG_DATA_DIRS"] = f"{data_dir}:{existing}"


def install_linux_desktop_entry() -> None:
    """Install/update ~/.local/share launcher so GNOME/KDE can show the app icon."""
    if sys.platform != "linux":
        return
    src = _icon_source()
    if src is None:
        return

    apps_dir = Path.home() / ".local/share/applications"
    icon_root = Path.home() / ".local/share/icons/hicolor"
    for size in (48, 256):
        (icon_root / f"{size}x{size}" / "apps").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, icon_root / f"{size}x{size}" / "apps" / f"{DESKTOP_ID}.png")

    apps_dir.mkdir(parents=True, exist_ok=True)
    exe = Path(sys.executable).resolve()
    exec_line = str(exe) if is_frozen() else f"{exe} -m video_picker"

    desktop_path = apps_dir / f"{DESKTOP_ID}.desktop"
    desktop_path.write_text(
        "\n".join(
            [
                "[Desktop Entry]",
                "Type=Application",
                "Name=Video Picker",
                "Comment=Run MegaDetector on videos",
                f"Exec={exec_line} %F",
                f"Icon={DESKTOP_ID}",
                "Terminal=false",
                "Categories=Utility;Science;",
                f"StartupWMClass={STARTUP_WM_CLASS}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if shutil.which("update-desktop-database"):
        subprocess.run(
            ["update-desktop-database", str(apps_dir)],
            check=False,
            capture_output=True,
        )
    cache_bin = shutil.which("gtk-update-icon-cache")
    if cache_bin:
        subprocess.run(
            [cache_bin, "-f", "-t", str(icon_root)],
            check=False,
            capture_output=True,
        )


def load_app_icon() -> QIcon | None:
    path = _icon_source()
    if path is None:
        return None
    pix = QPixmap(str(path))
    if pix.isNull():
        return None
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(
            pix.scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
    return icon if not icon.isNull() else None


def configure_qt_application(app) -> QIcon | None:
    app.setApplicationName(DESKTOP_ID)
    app.setOrganizationName(DESKTOP_ID)
    app.setApplicationDisplayName("Video Picker")
    if sys.platform == "linux":
        QGuiApplication.setDesktopFileName(DESKTOP_ID)
    icon = load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    return icon
