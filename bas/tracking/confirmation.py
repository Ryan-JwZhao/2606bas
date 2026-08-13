from __future__ import annotations

from typing import Iterable, TypeVar


TrackT = TypeVar("TrackT")


def is_track_confirmed(track: object) -> bool:
    """Return whether a track has enough real detections for normal consumers.

    Older replay objects and hand-built test observations do not carry the
    field, so they remain confirmed for backward compatibility.
    """

    return bool(getattr(track, "confirmed", True))


def confirmed_tracks(tracks: Iterable[TrackT]) -> list[TrackT]:
    return [track for track in tracks if is_track_confirmed(track)]
