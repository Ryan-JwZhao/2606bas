from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .geometry import TableGeometry, TableGeometryLoader

LOGGER = logging.getLogger(__name__)


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
            try:
                candidate = (
                    TableGeometryLoader.load(outline_path, inline_path, pocket_path)
                    if any((outline_path, inline_path, pocket_path))
                    else TableGeometry()
                )
            except Exception as exc:
                LOGGER.warning("Geometry reload deferred; keeping last valid geometry: %s", exc)
                return self._geometry, False
            validation_error = _configured_geometry_validation_error(
                candidate,
                outline_path=outline_path,
                inline_path=inline_path,
                pocket_path=pocket_path,
            )
            if validation_error is not None:
                LOGGER.warning(
                    "Geometry reload deferred; keeping last valid geometry: %s",
                    validation_error,
                )
                return self._geometry, False
            self._geometry = candidate
            self._snapshot = snapshot
        return self._geometry, changed


def _configured_geometry_validation_error(
    geometry: TableGeometry,
    *,
    outline_path: Optional[str],
    inline_path: Optional[str],
    pocket_path: Optional[str],
) -> Optional[str]:
    if outline_path and geometry.outer_norm.shape[0] < 3:
        return "configured outline has fewer than three points"
    if inline_path and not geometry.inline_norm:
        return "configured inline contains no usable lines"
    if pocket_path and not geometry.pockets_norm:
        return "configured pocket file contains no usable curves"
    if inline_path and pocket_path and geometry.inner_norm.shape[0] < 3:
        return "inline and pocket curves did not form a usable boundary"
    return None
