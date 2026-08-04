from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .geometry import TableGeometry, TableGeometryLoader

LOGGER = logging.getLogger(__name__)


class GeometryValidationError(RuntimeError):
    """Configured table geometry is missing, incomplete, or internally inconsistent."""


@dataclass(frozen=True)
class GeometryFileFingerprint:
    path: Optional[str]
    mtime_ns: Optional[int]
    size_bytes: Optional[int]
    content_sha256: Optional[str]

    @classmethod
    def from_path(cls, path: Optional[str]) -> "GeometryFileFingerprint":
        if not path:
            return cls(path=None, mtime_ns=None, size_bytes=None, content_sha256=None)
        file_path = Path(path)
        if not file_path.exists():
            return cls(path=str(file_path), mtime_ns=None, size_bytes=None, content_sha256=None)
        try:
            stat = file_path.stat()
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        except OSError:
            return cls(path=str(file_path), mtime_ns=None, size_bytes=None, content_sha256=None)
        return cls(
            path=str(file_path),
            mtime_ns=int(stat.st_mtime_ns),
            size_bytes=int(stat.st_size),
            content_sha256=digest,
        )


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
        self._failed_snapshot: Optional[GeometrySnapshot] = None
        self._geometry = TableGeometry()
        self._is_ready = False
        self._last_error: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """Whether a valid geometry (or an intentional no-geometry setup) has loaded."""

        return bool(self._is_ready)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

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
                self._defer_reload(snapshot, str(exc))
                return self._geometry, False
            validation_error = _configured_geometry_validation_error(
                candidate,
                outline_path=outline_path,
                inline_path=inline_path,
                pocket_path=pocket_path,
            )
            if validation_error is not None:
                self._defer_reload(snapshot, validation_error)
                return self._geometry, False
            self._geometry = candidate
            self._snapshot = snapshot
            self._failed_snapshot = None
            self._last_error = None
            self._is_ready = True
        return self._geometry, changed

    def _defer_reload(self, snapshot: GeometrySnapshot, error: str) -> None:
        if self._failed_snapshot != snapshot or self._last_error != error:
            LOGGER.warning("Geometry reload deferred; keeping last valid geometry: %s", error)
        self._failed_snapshot = snapshot
        self._last_error = str(error)


def load_validated_table_geometry(
    outline_path: Optional[str],
    inline_path: Optional[str],
    pocket_path: Optional[str],
    *,
    allow_empty: bool = True,
) -> TableGeometry:
    """Load geometry through the same fail-closed contract used by runtime hot reload."""

    reloader = RuntimeGeometryReloader()
    geometry, _ = reloader.refresh(outline_path, inline_path, pocket_path)
    if not reloader.is_ready:
        raise GeometryValidationError(reloader.last_error or "configured geometry is not runtime-ready")
    if geometry.is_empty and not allow_empty:
        raise GeometryValidationError("configured geometry is empty")
    return geometry


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
    if inline_path and pocket_path and not geometry.boundary_complete:
        return (
            "inline and pocket curves formed an incomplete boundary "
            f"({len(geometry.boundary_segments_norm)}/{geometry.boundary_source_count} segments used)"
        )
    if geometry.boundary_self_intersections:
        return (
            "inline and pocket curves formed a self-intersecting boundary "
            f"({geometry.boundary_self_intersections} intersections)"
        )
    if pocket_path and len(geometry.pockets_norm) != 6:
        return f"configured pocket file must contain exactly six usable curves (found {len(geometry.pockets_norm)})"
    return None
