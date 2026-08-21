from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional


BALL_CODE_BY_GROUP = {
    "cue": "wb",
    "black": "bb",
    "solid": "sob",
    "stripe": "stb",
}

BALL_MESSAGE_BY_GROUP = {
    "cue": "白球进洞",
    "black": "黑八进洞",
    "solid": "全色球进洞",
    "stripe": "花色球进洞",
}

POCKET_NOTICE_EVENT_NAMES = {"POCKET_CONFIRMED", "POT_PROBABLE"}


@dataclass(frozen=True)
class PocketNotice:
    notice_id: str
    sequence: int
    decision_id: str
    group: str
    ball_code: str
    message: str
    track_id: object = None
    pocket_index: object = None

    def to_payload(self) -> dict[str, object]:
        return {
            "notice_id": self.notice_id,
            "sequence": int(self.sequence),
            "decision_id": self.decision_id,
            "group": self.group,
            "ball_code": self.ball_code,
            "message": self.message,
            "track_id": self.track_id,
            "pocket_index": self.pocket_index,
        }


class PocketNoticeTracker:
    """Keeps short-lived, de-duplicated pocket notices for server-side UIs."""

    def __init__(self, *, retention_s: float = 8.0, max_notices: int = 16, max_seen: int = 128) -> None:
        self.retention_s = max(1.0, float(retention_s))
        self.max_notices = max(1, int(max_notices))
        self.max_seen = max(self.max_notices, int(max_seen))
        self._session_id = uuid.uuid4().hex
        self._sequence = 0
        self._notices: deque[tuple[float, PocketNotice]] = deque()
        self._seen_keys: set[str] = set()
        self._seen_order: deque[str] = deque()

    def reset(self) -> None:
        self._notices.clear()
        self._seen_keys.clear()
        self._seen_order.clear()

    def observe(self, events: Iterable[object], *, now_s: Optional[float] = None) -> list[dict[str, object]]:
        now = time.monotonic() if now_s is None else float(now_s)
        self._expire(now)
        created: list[dict[str, object]] = []
        for event in list(events or []):
            if str(getattr(event, "name", "")).strip().upper() not in POCKET_NOTICE_EVENT_NAMES:
                continue
            payload = dict(getattr(event, "payload", None) or {})
            group = str(payload.get("group") or "").strip().lower()
            ball_code = BALL_CODE_BY_GROUP.get(group)
            if ball_code is None:
                continue
            key = self._event_key(event, payload, group)
            if key in self._seen_keys:
                continue
            self._remember_key(key)
            self._sequence += 1
            notice = PocketNotice(
                notice_id=f"{self._session_id}:{self._sequence}",
                sequence=self._sequence,
                decision_id=str(payload.get("decision_id") or "").strip(),
                group=group,
                ball_code=ball_code,
                message=BALL_MESSAGE_BY_GROUP[group],
                track_id=payload.get("track_id"),
                pocket_index=payload.get("pocket_index"),
            )
            self._notices.append((now + self.retention_s, notice))
            while len(self._notices) > self.max_notices:
                self._notices.popleft()
            created.append(notice.to_payload())
        return created

    def current(self, *, now_s: Optional[float] = None) -> list[dict[str, object]]:
        now = time.monotonic() if now_s is None else float(now_s)
        self._expire(now)
        return [notice.to_payload() for _, notice in self._notices]

    def _expire(self, now_s: float) -> None:
        while self._notices and self._notices[0][0] <= now_s:
            self._notices.popleft()

    def _remember_key(self, key: str) -> None:
        self._seen_keys.add(key)
        self._seen_order.append(key)
        while len(self._seen_order) > self.max_seen:
            expired = self._seen_order.popleft()
            self._seen_keys.discard(expired)

    @staticmethod
    def _event_key(event: object, payload: dict[str, object], group: str) -> str:
        decision_id = str(payload.get("decision_id") or "").strip()
        if decision_id:
            return f"decision:{decision_id}"
        return ":".join(
            [
                "event",
                str(getattr(event, "frame_id", "")),
                str(payload.get("track_id") or ""),
                str(payload.get("pocket_index") or ""),
                group,
            ]
        )


__all__ = [
    "BALL_CODE_BY_GROUP",
    "BALL_MESSAGE_BY_GROUP",
    "POCKET_NOTICE_EVENT_NAMES",
    "PocketNotice",
    "PocketNoticeTracker",
]
