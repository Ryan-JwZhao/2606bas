from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .ball_compensation_sampling import BallCompensationSample


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HoldoutCheckpointObservation:
    location_index: int
    repeat_index: int
    sample: BallCompensationSample

    def to_dict(self) -> dict[str, Any]:
        return {
            "location_index": int(self.location_index),
            "repeat_index": int(self.repeat_index),
            "sample": self.sample.to_dict(),
        }


@dataclass(frozen=True)
class BallCompensationCheckpoint:
    context: dict[str, Any]
    sampling_grid_table_mm: tuple[tuple[float, float], ...]
    training_cursor: int = 0
    training_samples: tuple[BallCompensationSample, ...] = ()
    holdout_observations: tuple[HoldoutCheckpointObservation, ...] = ()
    saved_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))

    @property
    def holdout_completed_count(self) -> int:
        return len(self.holdout_observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "saved_at": str(self.saved_at),
            "context": dict(self.context),
            "sampling_grid_table_mm": [list(point) for point in self.sampling_grid_table_mm],
            "training_cursor": int(self.training_cursor),
            "training_samples": [sample.to_dict() for sample in self.training_samples],
            "holdout_observations": [item.to_dict() for item in self.holdout_observations],
        }


def make_ball_compensation_checkpoint(
    *,
    context: Mapping[str, Any],
    sampling_grid_table_mm: Sequence[Sequence[float]],
    training_cursor: int,
    training_samples: Sequence[BallCompensationSample],
    holdout_observations: Sequence[HoldoutCheckpointObservation] = (),
) -> BallCompensationCheckpoint:
    grid = np.asarray(sampling_grid_table_mm, dtype=np.float64).reshape((-1, 2))
    return BallCompensationCheckpoint(
        context=dict(context),
        sampling_grid_table_mm=tuple((float(point[0]), float(point[1])) for point in grid),
        training_cursor=max(0, int(training_cursor)),
        training_samples=tuple(training_samples),
        holdout_observations=tuple(holdout_observations),
    )


def save_ball_compensation_checkpoint(path: str | Path, checkpoint: BallCompensationCheckpoint) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def load_ball_compensation_checkpoint(path: str | Path) -> BallCompensationCheckpoint | None:
    source = Path(path)
    if not source.exists():
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported ball compensation checkpoint schema: {payload.get('schema_version')}")
    training = tuple(_sample_from_dict(item) for item in payload.get("training_samples", []))
    holdout = tuple(
        HoldoutCheckpointObservation(
            location_index=int(item["location_index"]),
            repeat_index=int(item["repeat_index"]),
            sample=_sample_from_dict(item["sample"]),
        )
        for item in payload.get("holdout_observations", [])
    )
    grid = np.asarray(payload.get("sampling_grid_table_mm", []), dtype=np.float64).reshape((-1, 2))
    return BallCompensationCheckpoint(
        context=dict(payload.get("context", {})),
        sampling_grid_table_mm=tuple((float(point[0]), float(point[1])) for point in grid),
        training_cursor=max(0, int(payload.get("training_cursor", 0))),
        training_samples=training,
        holdout_observations=holdout,
        saved_at=str(payload.get("saved_at", "unknown")),
    )


def delete_ball_compensation_checkpoint(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()


def ball_compensation_checkpoint_compatibility_errors(
    checkpoint: BallCompensationCheckpoint,
    *,
    context: Mapping[str, Any],
    sampling_grid_table_mm: Sequence[Sequence[float]],
) -> list[str]:
    errors: list[str] = []
    expected = dict(context)
    actual = dict(checkpoint.context)
    for key in (
        "projection_file",
        "detector_model_file",
        "camera_coordinate_domain",
        "frame_rotation_degrees",
        "frame_width",
        "frame_height",
    ):
        left = actual.get(key)
        right = expected.get(key)
        if key.endswith("_file"):
            left = _normalized_path(left)
            right = _normalized_path(right)
        if left != right:
            errors.append(f"{key} changed ({actual.get(key)!r} -> {expected.get(key)!r})")
    if not np.isclose(
        float(actual.get("ball_diameter_mm", -1.0)),
        float(expected.get("ball_diameter_mm", -2.0)),
        atol=0.05,
    ):
        errors.append("ball_diameter_mm changed")
    saved_grid = np.asarray(checkpoint.sampling_grid_table_mm, dtype=np.float64).reshape((-1, 2))
    expected_grid = np.asarray(sampling_grid_table_mm, dtype=np.float64).reshape((-1, 2))
    if saved_grid.shape != expected_grid.shape or not np.allclose(saved_grid, expected_grid, atol=0.5):
        errors.append("sampling grid changed")
    if checkpoint.training_cursor > expected_grid.shape[0]:
        errors.append("training cursor exceeds sampling grid")
    training_indices = [int(sample.sample_index) for sample in checkpoint.training_samples]
    if len(training_indices) != len(set(training_indices)):
        errors.append("training samples contain duplicate indices")
    if any(index < 0 or index >= checkpoint.training_cursor for index in training_indices):
        errors.append("training sample index is outside completed cursor")
    return errors


def checkpoint_from_audit_events(
    events_path: str | Path,
    *,
    frame_rotation_degrees: int,
    camera_coordinate_domain: str,
    ball_diameter_mm: float,
    sampling_detection: str,
) -> BallCompensationCheckpoint:
    events = [
        json.loads(line)
        for line in Path(events_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    started = _last_event(events, "session_started")
    loaded = _last_event(events, "calibration_loaded")
    accepted = sorted(
        (event for event in events if event.get("action") == "sample_accepted"),
        key=lambda event: int(event.get("metrics", {}).get("sample_index", 0)),
    )
    if not accepted:
        raise ValueError("Audit contains no accepted ball compensation samples.")
    samples = tuple(_sample_from_audit_event(event) for event in accepted)
    indices = [int(event["metrics"]["sample_index"]) for event in accepted]
    if indices != list(range(1, max(indices) + 1)):
        raise ValueError("Audit training samples are not contiguous and cannot be resumed safely.")
    session_context = dict(started.get("details", {}).get("context", {}))
    loaded_metrics = dict(loaded.get("metrics", {}))
    loaded_details = dict(loaded.get("details", {}))
    context = {
        "projection_file": loaded_details.get("projection_source") or session_context.get("projection_file"),
        "detector_model_file": session_context.get("detector_model"),
        "camera_coordinate_domain": str(camera_coordinate_domain),
        "frame_rotation_degrees": int(frame_rotation_degrees),
        "frame_width": int(loaded_metrics.get("frame_width", 0)),
        "frame_height": int(loaded_metrics.get("frame_height", 0)),
        "ball_diameter_mm": float(ball_diameter_mm),
        "sampling_detection": str(sampling_detection),
    }
    grid = [sample.target_table_mm for sample in samples]
    return make_ball_compensation_checkpoint(
        context=context,
        sampling_grid_table_mm=grid,
        training_cursor=max(indices),
        training_samples=samples,
    )


def _sample_from_audit_event(event: Mapping[str, Any]) -> BallCompensationSample:
    metrics = dict(event.get("metrics", {}))
    details = dict(event.get("details", {}))
    target = np.asarray(details["target_table_mm"], dtype=np.float64).reshape((2,))
    delta = np.asarray(details["delta_table_mm"], dtype=np.float64).reshape((2,))
    detected = np.asarray(details["detected_camera_px"], dtype=np.float64).reshape((2,))
    observed = target - delta
    return BallCompensationSample(
        sample_index=int(metrics["sample_index"]) - 1,
        target_table_mm=(float(target[0]), float(target[1])),
        detected_camera_px=(float(detected[0]), float(detected[1])),
        projected_target_px=(0.0, 0.0),
        expected_camera_px=(0.0, 0.0),
        observed_table_mm=(float(observed[0]), float(observed[1])),
        delta_table_mm=(float(delta[0]), float(delta[1])),
        detected_radius_px=float(metrics.get("detected_radius_px", 0.0)),
        detection_confidence=float(metrics.get("detection_confidence", 0.0)),
        stability_spread_px=float(metrics.get("stability_spread_px", 0.0)),
        geometry_quality=float(metrics.get("geometry_quality", 0.0)),
        geometry_method=str(details.get("geometry_method", "unknown")),
        detector_version=str(details.get("detector_version", "unknown")),
    )


def _sample_from_dict(payload: Mapping[str, Any]) -> BallCompensationSample:
    def point(key: str) -> tuple[float, float]:
        values = np.asarray(payload.get(key, [0.0, 0.0]), dtype=np.float64).reshape((2,))
        return float(values[0]), float(values[1])

    return BallCompensationSample(
        sample_index=int(payload.get("sample_index", 0)),
        target_table_mm=point("target_table_mm"),
        detected_camera_px=point("detected_camera_px"),
        projected_target_px=point("projected_target_px"),
        expected_camera_px=point("expected_camera_px"),
        observed_table_mm=point("observed_table_mm"),
        delta_table_mm=point("delta_table_mm"),
        detected_radius_px=float(payload.get("detected_radius_px", 0.0)),
        detection_confidence=float(payload.get("detection_confidence", 0.0)),
        stability_spread_px=float(payload.get("stability_spread_px", 0.0)),
        geometry_quality=float(payload.get("geometry_quality", 0.0)),
        geometry_method=str(payload.get("geometry_method", "unknown")),
        detector_version=str(payload.get("detector_version", "unknown")),
    )


def _last_event(events: Iterable[Mapping[str, Any]], action: str) -> Mapping[str, Any]:
    matches = [event for event in events if event.get("action") == action]
    if not matches:
        raise ValueError(f"Audit is missing required event: {action}")
    return matches[-1]


def _normalized_path(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return os.path.normcase(str(Path(str(value)).resolve(strict=False)))
