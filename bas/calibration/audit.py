from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np


LOGGER = logging.getLogger(__name__)
AUDIT_SCHEMA_VERSION = 1
FINAL_STATUSES = {"success", "failed", "aborted", "cancelled"}


class CalibrationAudit:
    """Durable, fail-open audit trail for one calibration workflow.

    Callers only report domain actions. Serialization, timestamps, sequence
    numbers, summaries and filesystem failures stay behind this interface.
    """

    def __init__(
        self,
        log_dir: str | Path,
        workflow: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.workflow = _safe_name(workflow)
        self.session_id = uuid.uuid4().hex[:12]
        self.started_at = _utc_now()
        self._started_monotonic = time.monotonic()
        self._sequence = 0
        self._finished = False
        self._write_error: Optional[str] = None
        self._context = _json_safe(dict(context or {}))
        self._warning_count = 0
        self._error_count = 0
        self._last_action = "session_started"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(log_dir) / "calibration" / datetime.now().strftime("%Y%m%d") / (
            f"{stamp}_{self.workflow}_{self.session_id}"
        )
        self.events_path = self.session_dir / "events.jsonl"
        self.summary_path = self.session_dir / "summary.json"
        self.event("session_started", details={"context": self._context})

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def write_error(self) -> Optional[str]:
        return self._write_error

    def event(
        self,
        action: str,
        *,
        status: str = "ok",
        metrics: Optional[Mapping[str, Any]] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._finished:
            return
        normalized_status = str(status or "ok").strip().lower()
        self._sequence += 1
        self._last_action = str(action)
        if normalized_status == "warning":
            self._warning_count += 1
        if normalized_status in {"failed", "error"}:
            self._error_count += 1
        payload = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "session_id": self.session_id,
            "workflow": self.workflow,
            "sequence": self._sequence,
            "timestamp": _utc_now(),
            "elapsed_ms": round((time.monotonic() - self._started_monotonic) * 1000.0, 3),
            "action": str(action),
            "status": normalized_status,
            "metrics": _json_safe(dict(metrics or {})),
            "details": _json_safe(dict(details or {})),
        }
        self._append(payload)

    def finish(
        self,
        status: str,
        *,
        quality: Optional[Mapping[str, Any]] = None,
        artifacts: Iterable[str | Path] = (),
        error: BaseException | str | None = None,
    ) -> Path:
        if self._finished:
            return self.summary_path
        normalized_status = str(status or "failed").strip().lower()
        if normalized_status not in FINAL_STATUSES:
            raise ValueError(f"unsupported calibration audit final status: {status}")
        error_payload: Optional[dict[str, str]] = None
        if error is not None:
            error_payload = {
                "type": type(error).__name__ if isinstance(error, BaseException) else "Error",
                "message": str(error),
            }
        self.event(
            "session_finished",
            status="ok" if normalized_status == "success" else normalized_status,
            metrics={"event_count": self._sequence + 1},
            details={"error": error_payload} if error_payload else {},
        )
        self._finished = True
        ended_at = _utc_now()
        summary = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "session_id": self.session_id,
            "workflow": self.workflow,
            "status": normalized_status,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "duration_ms": round((time.monotonic() - self._started_monotonic) * 1000.0, 3),
            "event_count": self._sequence,
            "warning_count": self._warning_count,
            "error_count": self._error_count,
            "last_action": self._last_action,
            "context": self._context,
            "quality": _json_safe(dict(quality or {})),
            "artifacts": [str(Path(item)) for item in artifacts],
            "error": error_payload,
            "events_path": str(self.events_path),
            "write_error": self._write_error,
        }
        self._write_summary(summary)
        LOGGER.info(
            "Calibration audit finished workflow=%s session=%s status=%s summary=%s",
            self.workflow,
            self.session_id,
            normalized_status,
            self.summary_path,
        )
        return self.summary_path

    def _append(self, payload: Mapping[str, Any]) -> None:
        if self._write_error is not None:
            return
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            self._write_error = str(exc)
            LOGGER.warning("Calibration audit event write failed: %s", exc)

    def _write_summary(self, summary: Mapping[str, Any]) -> None:
        if self._write_error is not None:
            return
        temporary = self.summary_path.with_suffix(".json.tmp")
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.summary_path)
        except OSError as exc:
            self._write_error = str(exc)
            LOGGER.warning("Calibration audit summary write failed: %s", exc)


def start_calibration_audit(
    log_dir: str | Path,
    workflow: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> CalibrationAudit:
    return CalibrationAudit(log_dir, workflow, context=context)


def load_calibration_audit_summaries(
    log_dir: str | Path,
    *,
    workflow: Optional[str] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    root = Path(log_dir) / "calibration"
    summaries: list[dict[str, Any]] = []
    if not root.exists():
        return summaries
    expected_workflow = _safe_name(workflow) if workflow else None
    paths = sorted(root.glob("*/*/summary.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(summary, dict):
            continue
        if expected_workflow and summary.get("workflow") != expected_workflow:
            continue
        summary["summary_path"] = str(path)
        summaries.append(summary)
        if len(summaries) >= max(1, int(limit)):
            break
    return summaries


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _safe_name(value: str | None) -> str:
    raw = str(value or "calibration").strip().lower()
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in raw)
    return cleaned.strip("_") or "calibration"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
