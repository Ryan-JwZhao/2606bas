from __future__ import annotations

import argparse
import json
from typing import Optional

from .capture import capture_frames_are_distortion_corrected
from .config import AppConfig
from .runtime_env import prepare_runtime_environment, preload_torch_for_backend
from .single_instance import acquire_runtime_single_instance
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

    p_doctor = sub.add_parser("doctor", help="Check runtime dependencies and configured file paths.")
    p_doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    p_calib = sub.add_parser("inspect-calib", help="Print loaded calibration summary.")

    p_verify = sub.add_parser("verify-calib", help="Verify calibration against a holdout JSON file.")
    p_verify.add_argument("holdout_json")

    p_replay = sub.add_parser("replay-summary", help="Summarize a replay events.jsonl file.")
    p_replay.add_argument("path")

    p_smoke = sub.add_parser("smoke-run", help="Run synthetic pipeline for a fixed number of frames.")
    p_smoke.add_argument("--frames", type=int, default=90)

    p_remote = sub.add_parser("remote-control", help="Queue a local command for the running desktop control console.")
    p_remote.add_argument(
        "action",
        help="Remote action name, for example: start-capture, toggle-shot-mode, free-shot-once, save-retro-clip.",
    )

    args = parser.parse_args(argv)
    instance_handle = None
    if args.command in {None, "ui", "run"}:
        instance_handle = acquire_runtime_single_instance()
        if instance_handle is None:
            return 0

    try:
        cfg = UserSettings.load().apply_to_config(AppConfig.load(args.config)).resolve_paths()
        prepare_runtime_environment()
        if args.command in {None, "ui", "run"}:
            preload_torch_for_backend(cfg.detector.backend)

        if args.command in {None, "ui"}:
            from .ui import run_operator_ui

            return run_operator_ui(cfg)

        if args.command == "run":
            from .app import run_headless, run_qt

            if args.command == "run" and args.headless:
                return run_headless(cfg, max_frames=args.max_frames)
            return run_qt(cfg)

        if args.command == "probe-cameras":
            from .capture import probe_cameras
            from .logging_config import configure_logging

            configure_logging(cfg.logging.directory, cfg.logging.level)
            rows = probe_cameras(max_index=args.max_index, nori_sdk_root=cfg.camera.nori_sdk_root)
            if not rows:
                print("No cameras found.")
                return 0
            for backend, idx, width, height, fps in rows:
                print(f"{backend:10s} id={idx:<3d} {width}x{height} @ {fps:.2f}fps")
            return 0

        if args.command == "doctor":
            from .dependency_check import dependency_report

            rows = dependency_report()
            if args.json:
                print(json.dumps(rows, ensure_ascii=False, indent=2))
            else:
                for row in rows:
                    detail = f" {row['detail']}" if row.get("detail") else ""
                    print(f"{row['status']:8s} {row['name']}{detail}")
            return 1 if any(row["status"] == "missing" for row in rows) else 0

        if args.command == "inspect-calib":
            from .calibration import create_calibration_service
            from .logging_config import configure_logging
            from .schemas import to_jsonable

            configure_logging(cfg.logging.directory, cfg.logging.level)
            service = create_calibration_service(
                cfg.calibration,
                frame_undistorted=capture_frames_are_distortion_corrected(cfg.camera),
            )
            summary = {
                "calib_version": service.calib_version,
                "camera_valid": service.camera.is_valid,
                "camera_image_size": service.camera.image_size,
                "camera_source": service.camera.source_path,
                "frame_undistorted": service.frame_undistorted,
                "projection_runtime_mode": service.projection_mode,
                "projection_valid": service.projection.is_valid,
                "projection_source": service.projection.source_path,
                "projection_mode": service.projection.mode,
                "projector_size": service.projection.projector_size,
                "projection_error": service.projection.calibration_error_stats(),
                "ball_compensation_valid": service.ball_compensation_model.is_valid,
                "ball_compensation_source": service.ball_compensation_model.source_path,
                "ball_compensation_mode": service.ball_compensation_model.mode,
                "table": to_jsonable(service.table),
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

        if args.command == "verify-calib":
            from .calibration import create_calibration_service, format_holdout_report, verify_holdout_file
            from .logging_config import configure_logging

            configure_logging(cfg.logging.directory, cfg.logging.level)
            service = create_calibration_service(
                cfg.calibration,
                frame_undistorted=capture_frames_are_distortion_corrected(cfg.camera),
            )
            report = verify_holdout_file(args.holdout_json, service)
            print(format_holdout_report(report))
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0

        if args.command == "replay-summary":
            from .replay import ReplayReader

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
            from .app import run_headless

            cfg.camera.backend = "synthetic"
            cfg.detector.backend = "debug_color"
            cfg.replay.enabled = True
            cfg.projection.enabled = False
            return run_headless(cfg, max_frames=args.frames)

        if args.command == "remote-control":
            from .remote_control import RemoteCommandQueue

            path = RemoteCommandQueue().enqueue(args.action, source="cli")
            print(str(path))
            return 0

        return 0
    finally:
        if instance_handle is not None:
            instance_handle.release()


if __name__ == "__main__":
    raise SystemExit(main())
