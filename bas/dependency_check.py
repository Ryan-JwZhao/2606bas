from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .config import AppConfig
from .runtime_env import prepare_runtime_environment
from .user_settings import UserSettings


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _active_config() -> AppConfig:
    return UserSettings.load().apply_to_config(AppConfig.load()).resolve_paths()


def _active_detector_backend() -> str:
    cfg = _active_config()
    return str(cfg.detector.backend or "disabled").lower()


def needs_yolo_dependencies() -> bool:
    return _active_detector_backend() in {"ultralytics", "yolo"}


def yolo_dependencies_available() -> bool:
    return _yolo_import_status()[0] == "ok"


def dependency_report() -> list[dict[str, str]]:
    cfg = _active_config()
    rows: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str = "") -> None:
        rows.append({"name": name, "status": status, "detail": detail})

    for module in ["numpy", "cv2", "PyQt5", "yaml"]:
        add(f"module:{module}", "ok" if _has_module(module) else "missing")

    backend = str(cfg.detector.backend or "disabled").lower()
    add("detector_backend", "ok", backend)
    if backend in {"ultralytics", "yolo"}:
        add("module:ultralytics", *_yolo_import_status())
        add("detector.device", *_device_status(str(cfg.detector.device or "auto")))
        _check_path(rows, "detector.model_path", cfg.detector.model_path)
        _check_path(rows, "detector.class_file_path", cfg.detector.class_file_path)

    if _has_module("cv2"):
        try:
            import cv2  # type: ignore

            add("opencv:aruco", "ok" if hasattr(cv2, "aruco") else "warning", "ChArUco uses encoded-grid fallback when unavailable.")
        except Exception as exc:
            add("opencv:aruco", "warning", str(exc))

    _check_path(rows, "camera.distortion_correction_file", cfg.camera.distortion_correction_file, optional=not cfg.camera.distortion_correction_enabled)
    _check_path(rows, "calibration.camera_file", cfg.calibration.camera_file, optional=True)
    _check_path(rows, "calibration.projection_file", cfg.calibration.projection_file, optional=True)
    _check_path(rows, "geometry.outline_path", cfg.geometry.outline_path, optional=True)
    _check_path(rows, "geometry.inline_path", cfg.geometry.inline_path, optional=True)
    _check_path(rows, "geometry.pocket_path", cfg.geometry.pocket_path, optional=True)
    return rows


def _check_path(rows: list[dict[str, str]], name: str, value: str | None, optional: bool = False) -> None:
    if not value:
        rows.append({"name": name, "status": "optional" if optional else "missing", "detail": "not configured"})
        return
    exists = Path(value).exists()
    rows.append({"name": name, "status": "ok" if exists else ("warning" if optional else "missing"), "detail": value})


def _yolo_import_status() -> tuple[str, str]:
    if not _has_module("ultralytics"):
        return "missing", "Run Setup_Environment.cmd or install requirements-yolo.txt."
    try:
        prepare_runtime_environment()
        import ultralytics  # type: ignore

        version = getattr(ultralytics, "__version__", "unknown")
        return "ok", str(version)
    except Exception as exc:
        return "warning", f"{type(exc).__name__}: {exc}"


def _device_status(device: str) -> tuple[str, str]:
    requested = str(device or "auto").strip().lower()
    if requested in {"", "auto", "cpu", "mps"}:
        return "ok", requested or "auto"
    if requested.isdigit() or requested.startswith("cuda"):
        if not _has_module("torch"):
            return "warning", f"{requested}; torch is not installed"
        normalized = f"cuda:{requested}" if requested.isdigit() else requested
        return "warning", f"{normalized}; runtime falls back to CPU when CUDA is unavailable"
    return "ok", requested


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check optional BAS runtime dependencies.")
    parser.add_argument("--needs-yolo", action="store_true", help="Return 0 when the active config uses Ultralytics.")
    parser.add_argument("--yolo-available", action="store_true", help="Return 0 when Ultralytics can be imported.")
    parser.add_argument("--print-active", action="store_true", help="Print active dependency state.")
    parser.add_argument("--json", action="store_true", help="Print dependency report as JSON.")
    args = parser.parse_args(argv)

    if args.print_active:
        rows = dependency_report()
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for row in rows:
                detail = f" {row['detail']}" if row.get("detail") else ""
                print(f"{row['status']:8s} {row['name']}{detail}")
        return 0
    if args.needs_yolo:
        return 0 if needs_yolo_dependencies() else 1
    if args.yolo_available:
        return 0 if yolo_dependencies_available() else 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
