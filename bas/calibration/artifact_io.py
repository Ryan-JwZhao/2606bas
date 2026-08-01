from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a calibration artifact and require a JSON object at its root."""

    artifact_path = Path(path)
    with artifact_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream, parse_constant=_reject_nonfinite_json_number)
    if not isinstance(data, dict):
        raise ValueError("calibration artifact root must be a JSON object")
    return data


def atomic_write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    """Durably replace a JSON artifact without truncating its last good copy."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=artifact_path.parent,
            prefix=f".{artifact_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, artifact_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _reject_nonfinite_json_number(token: str) -> None:
    raise ValueError(f"calibration artifact contains non-finite JSON number: {token}")
