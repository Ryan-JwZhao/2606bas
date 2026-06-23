from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2

from ..config import ReplayConfig
from ..schemas import (
    DetectionsFrame,
    FramePacket,
    MatchStateFrame,
    ProjectionOverlay,
    ShotPlan,
    TracksFrame,
    to_jsonable,
)
from ..utils import wall_time_id


class ReplayRecorder:
    def __init__(self, config: ReplayConfig):
        self.config = config
        self.session_id = f"session_{wall_time_id()}"
        self.session_dir = Path(config.directory) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.session_dir / "events.jsonl").open("a", encoding="utf-8")
        self._video_writer = None
        self._debug_dir = self.session_dir / "frames"
        if config.write_debug_frames:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

    def write_frame_packet(self, frame: FramePacket) -> None:
        payload = {
            "frame_id": frame.frame_id,
            "ts_cam_ns": frame.ts_cam_ns,
            "camera_id": frame.camera_id,
            "image_uri": frame.image_uri,
            "exposure_meta": frame.exposure_meta,
            "calib_version": frame.calib_version,
        }
        self._write("frame", payload)
        if self.config.write_debug_frames and frame.image is not None:
            path = self._debug_dir / f"frame_{frame.frame_id:08d}.jpg"
            cv2.imwrite(str(path), frame.image)
        if self.config.write_video and frame.image is not None:
            self._ensure_video(frame)
            if self._video_writer is not None:
                self._video_writer.write(frame.image)

    def write_detections(self, value: DetectionsFrame) -> None:
        self._write("detections", to_jsonable(value))

    def write_tracks(self, value: TracksFrame) -> None:
        self._write("tracks", to_jsonable(value))

    def write_state(self, value: MatchStateFrame) -> None:
        self._write("state", to_jsonable(value))

    def write_plan(self, value: ShotPlan) -> None:
        self._write("plan", to_jsonable(value))

    def write_overlay(self, value: ProjectionOverlay) -> None:
        self._write("overlay", to_jsonable(value))

    def close(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._jsonl.close()

    def _write(self, event_type: str, payload: object) -> None:
        self._jsonl.write(json.dumps({"type": event_type, "payload": payload}, ensure_ascii=False) + "\n")

    def _ensure_video(self, frame: FramePacket) -> None:
        if self._video_writer is not None or frame.image is None:
            return
        h, w = frame.image.shape[:2]
        path = self.session_dir / "capture.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))

    def __enter__(self) -> "ReplayRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

