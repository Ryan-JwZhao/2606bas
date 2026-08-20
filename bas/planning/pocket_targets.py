from __future__ import annotations

from typing import Sequence

from ..schemas import Point, TableModel


def planning_pocket_points(table: TableModel) -> Sequence[Point]:
    """Return isolated shot targets, with a legacy fallback for test fixtures."""

    fitted = list(getattr(table, "planning_pockets_mm", ()) or ())
    if fitted:
        return fitted
    return list(getattr(table, "pockets_mm", ()) or ())


def planning_pocket_mouth(
    table: TableModel,
    pocket_index: int,
) -> tuple[Point, Point] | None:
    """Return same-index jaw endpoints, or ``None`` to make planning fail closed."""

    mouths = list(getattr(table, "planning_pocket_mouths_mm", ()) or ())
    index = int(pocket_index)
    if index < 0 or index >= len(mouths):
        return None
    mouth = mouths[index]
    if len(mouth) != 2:
        return None
    return mouth[0], mouth[1]


__all__ = ["planning_pocket_mouth", "planning_pocket_points"]
