from __future__ import annotations

from typing import Sequence

from ..schemas import Point, TableModel


def planning_pocket_points(table: TableModel) -> Sequence[Point]:
    """Return isolated shot targets, with a legacy fallback for test fixtures."""

    fitted = list(getattr(table, "planning_pockets_mm", ()) or ())
    if fitted:
        return fitted
    return list(getattr(table, "pockets_mm", ()) or ())


__all__ = ["planning_pocket_points"]
