from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from .app import run_headless, run_qt
from .calibration import create_calibration_service
from .capture import probe_cameras
from .config import AppConfig
from .logging_config import configure_logging
from .replay import ReplayReader
from .schemas import to_jsonable
from .ui import run_operator_ui
from .user_settings import UserSettings


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="bas", description="Billiards assistance system")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("ui", help="Open the desktop control console.")

    p_run = sub.add_parser("run", help="Run the online pipeline.")
    p_run.add_argument("--headless", action="store_true", help="Run without projection Qt window.")
    p_run.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means forever/until capture ends.")

    p_probe = sub.add_parser("probe-cameras", help="List OpenCV and Nori MJPG cameras.")
    p_probe.add_argument("--max-index", type=int, default=12)

    p_calib = sub.add_parser("inspect-calib", help="Print loaded calibration summary.")

    p_replay = sub.add_parser("replay-summary", help="Summarize a replay events.jsonl file.")
    p_replay.add_argument("path")

    p_smoke = sub.add_parser("smoke-run", help="Run synthetic pipeline for a fixed number of frames.")
    p_smoke.add_argument("--frames", type=int, default=90)

    args = parser.parse_args(argv)
    cfg = UserSettings.load().apply_to_config(AppConfig.load(args.config)).resolve_paths()

    if args.command in {None, "ui"}:
        return run_operator_ui(cfg)

    if args.command == "run":
        if args.command == "run" and args.headless:
            return run_headless(cfg, max_frames=args.max_frames)
        return run_qt(cfg)

    if args.command == "probe-cameras":
        configure_logging(cfg.logging.directory, cfg.logging.level)
        rows = probe_cameras(max_index=args.max_index, nori_sdk_root=cfg.camera.nori_sdk_root)
        if not rows:
            print("No cameras found.")
            return 0
        for backend, idx, width, height, fps in rows:
            print(f"{backend:10s} id={idx:<3d} {width}x{height} @ {fps:.2f}fps")
        return 0

    if args.command == "inspect-calib":
        configure_logging(cfg.logging.directory, cfg.logging.level)
        service = create_calibration_service(cfg.calibration)
        summary = {
            "calib_version": service.calib_version,
            "camera_valid": service.camera.is_valid,
            "camera_image_size": service.camera.image_size,
            "camera_source": service.camera.source_path,
            "projection_valid": service.projection.is_valid,
            "projection_source": service.projection.source_path,
            "projection_mode": service.projection.mode,
            "projector_size": service.projection.projector_size,
            "projection_error": service.projection.calibration_error_stats(),
            "table": to_jsonable(service.table),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "replay-summary":
        counts: dict[str, int] = {}
        last_frame = -1
        for event in ReplayReader(args.path).iter_events():
            typ = str(event.get("type", "unknown"))
            counts[typ] = counts.get(typ, 0) + 1
            payload = event.get("payload", {})
            if isinstance(payload, dict) and "frame_id" in payload:
                try:
                    last_frame = max(last_frame, int(payload["frame_id"]))
                except Exception:
                    pass
        print(json.dumps({"event_counts": counts, "last_frame_id": last_frame}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "smoke-run":
        cfg.camera.backend = "synthetic"
        cfg.detector.backend = "debug_color"
        cfg.replay.enabled = True
        cfg.projection.enabled = False
        return run_headless(cfg, max_frames=args.frames)

    return 0
