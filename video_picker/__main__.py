"""Entry point for `python -m video_picker` and PyInstaller one-file builds."""

from __future__ import annotations

import sys


def main() -> None:
    if "--worker" in sys.argv:
        from video_picker.worker import main as worker_main

        raise SystemExit(worker_main())
    from video_picker.app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
