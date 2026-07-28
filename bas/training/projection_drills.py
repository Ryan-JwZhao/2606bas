from __future__ import annotations

from collections.abc import Callable

from ..config import ProjectionConfig
from ..schemas import ProjectionOverlay
from .models import TrainingScenario
from .stroke_check import (
    STROKE_CHECK_RENDERER_ID,
    TableProjection,
    build_stroke_check_overlay,
)


ProjectionDrillRenderer = Callable[..., ProjectionOverlay]

_RENDERERS: dict[str, ProjectionDrillRenderer] = {
    STROKE_CHECK_RENDERER_ID: build_stroke_check_overlay,
}


def build_projection_drill_overlay(
    scenario: TrainingScenario,
    config: ProjectionConfig,
    calibration: TableProjection,
    *,
    frame_id: int = 0,
) -> ProjectionOverlay:
    if not scenario.projection_only:
        raise ValueError(f"训练项目不是纯投影项目: {scenario.scenario_id}")
    renderer_id = str(scenario.projection_renderer_id or "").strip()
    renderer = _RENDERERS.get(renderer_id)
    if renderer is None:
        raise ValueError(f"未知纯投影训练渲染器: {renderer_id or '<empty>'}")
    return renderer(config, calibration, frame_id=frame_id)
