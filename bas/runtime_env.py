from __future__ import annotations

import os

from .paths import PROJECT_ROOT


def prepare_runtime_environment() -> None:
    local = PROJECT_ROOT / "local_settings"
    yolo_dir = local / "ultralytics"
    mpl_dir = local / "matplotlib"
    for path in (yolo_dir, mpl_dir):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
