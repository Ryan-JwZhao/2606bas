from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import PROJECT_ROOT


REMOTE_CONTROL_ROOT = PROJECT_ROOT / "local_settings" / "stream_deck"
REMOTE_CONTROL_QUEUE_DIR = REMOTE_CONTROL_ROOT / "queue"

REMOTE_ACTION_ALIASES = {
    "start_capture": "start_capture",
    "start-capture": "start_capture",
    "stop_capture": "stop_capture",
    "stop-capture": "stop_capture",
    "toggle_capture": "toggle_capture",
    "toggle-capture": "toggle_capture",
    "start_projection": "start_projection",
    "start-projection": "start_projection",
    "stop_projection": "stop_projection",
    "stop-projection": "stop_projection",
    "toggle_projection": "toggle_projection",
    "toggle-projection": "toggle_projection",
    "toggle_shot_mode": "toggle_shot_mode",
    "toggle-shot-mode": "toggle_shot_mode",
    "toggle_target_group": "toggle_target_group",
    "toggle-target-group": "toggle_target_group",
    "free_shot_once": "free_shot_once",
    "free-shot-once": "free_shot_once",
    "black_shot_once": "black_shot_once",
    "black-shot-once": "black_shot_once",
    "toggle_star_formula": "toggle_star_formula",
    "toggle-star-formula": "toggle_star_formula",
}


def normalize_remote_action(action: str) -> str:
    normalized = REMOTE_ACTION_ALIASES.get(str(action or "").strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported remote action: {action}")
    return normalized


@dataclass(frozen=True)
class RemoteCommand:
    command_id: str
    action: str
    created_at_ms: int
    source: str = "cli"
    args: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = field(default=None, compare=False)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "action": self.action,
            "created_at_ms": int(self.created_at_ms),
            "source": self.source,
            "args": dict(self.args),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], *, path: Optional[Path] = None) -> "RemoteCommand":
        if not isinstance(payload, dict):
            raise ValueError("Remote command payload must be a JSON object.")
        action = normalize_remote_action(str(payload.get("action", "")))
        command_id = str(payload.get("command_id") or "")
        if not command_id:
            raise ValueError("Remote command is missing command_id.")
        created_at_ms = int(payload.get("created_at_ms") or 0)
        args = payload.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("Remote command args must be a JSON object.")
        return cls(
            command_id=command_id,
            action=action,
            created_at_ms=created_at_ms,
            source=str(payload.get("source") or "cli"),
            args=dict(args),
            path=path,
        )


class RemoteCommandQueue:
    def __init__(self, queue_dir: Path = REMOTE_CONTROL_QUEUE_DIR):
        self.queue_dir = Path(queue_dir)

    def enqueue(self, action: str, *, args: Optional[Dict[str, Any]] = None, source: str = "cli") -> Path:
        normalized = normalize_remote_action(action)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        created_at_ms = int(time.time() * 1000)
        command_id = f"{created_at_ms}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        command = RemoteCommand(
            command_id=command_id,
            action=normalized,
            created_at_ms=created_at_ms,
            source=str(source or "cli"),
            args=dict(args or {}),
        )
        tmp_path = self.queue_dir / f".{command_id}.tmp"
        final_path = self.queue_dir / f"{command_id}_{normalized}.json"
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(command.to_payload(), f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, final_path)
        return final_path

    def drain(self, *, limit: int = 32) -> list[RemoteCommand]:
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        commands: list[RemoteCommand] = []
        for path in sorted(self.queue_dir.glob("*.json"))[: max(1, int(limit))]:
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                command = RemoteCommand.from_payload(payload, path=path)
            except Exception:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
                continue
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            commands.append(command)
        return commands
