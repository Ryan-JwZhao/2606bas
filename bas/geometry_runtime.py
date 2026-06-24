from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .geometry import TableGeometry, TableGeometryLoader


@dataclass(frozen=True)
class GeometryFileFingerprint:
    path: Optional[str]
    mtime_ns: Optional[int]
    size_bytes: Optional[int]

    @classmethod
    def from_path(cls, path: Optional[str]) -> "GeometryFileFingerprint":
        if not path:
            return cls(path=None, mtime_ns=None, size_bytes=None)
        file_path = Path(path)
        if not file_path.exists():
            return cls(path=str(file_path), mtime_ns=None, size_bytes=None)
        stat = file_path.stat()
        return cls(path=str(file_path), mtime_ns=int(stat.st_mtime_ns), size_bytes=int(stat.st_size))


@dataclass(frozen=True)
class GeometrySnapshot:
    outline: GeometryFileFingerprint
    inline: GeometryFileFingerprint
    pocket: GeometryFileFingerprint

    @classmethod
    def from_paths(
        cls,
        outline_path: Optional[str],
        inline_path: Optional[str],
        pocket_path: Optional[str],
    ) -> "GeometrySnapshot":
        return cls(
            outline=GeometryFileFingerprint.from_path(outline_path),
            inline=GeometryFileFingerprint.from_path(inline_path),
            pocket=GeometryFileFingerprint.from_path(pocket_path),
        )


class RuntimeGeometryReloader:
    def __init__(self) -> None:
        self._snapshot: Optional[GeometrySnapshot] = None
        self._geometry = TableGeometry()

    def refresh(
        self,
        outline_path: Optional[str],
        inline_path: Optional[str],
        pocket_path: Optional[str],
    ) -> Tuple[TableGeometry, bool]:
        snapshot = GeometrySnapshot.from_paths(outline_path, inline_path, pocket_path)
        changed = self._snapshot != snapshot
        if changed:
            self._geometry = TableGeometryLoader.load_optional(outline_path, inline_path, pocket_path)
            self._snapshot = snapshot
        return self._geometry, changed
