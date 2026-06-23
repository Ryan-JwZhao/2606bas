from __future__ import annotations

import os
from types import ModuleType
from typing import Optional

from .paths import PROJECT_ROOT

torch: Optional[ModuleType] = None
TORCH_IMPORT_ERROR: Optional[BaseException] = None


def prepare_runtime_environment() -> None:
    local = PROJECT_ROOT / "local_settings"
    yolo_dir = local / "ultralytics"
    mpl_dir = local / "matplotlib"
    for path in (yolo_dir, mpl_dir):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def preload_torch() -> Optional[ModuleType]:
    global torch, TORCH_IMPORT_ERROR
    if torch is not None:
        return torch
    try:
        import torch as torch_module  # type: ignore

        torch = torch_module
        TORCH_IMPORT_ERROR = None
        return torch
    except BaseException as exc:
        TORCH_IMPORT_ERROR = exc
        return None


def detector_uses_torch(backend: str | None) -> bool:
    return str(backend or "").strip().lower() in {"ultralytics", "yolo"}


def preload_torch_for_backend(backend: str | None) -> Optional[ModuleType]:
    prepare_runtime_environment()
    if detector_uses_torch(backend):
        return preload_torch()
    return torch
