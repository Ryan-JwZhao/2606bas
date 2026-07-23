from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..schemas import TrackObservation
from .models import TrainingScenario
from .numbered_tracker import ball_number_from_track


READY_MESSAGE = "摆球已通过，可开始训练"


def validate_scenario_setup(
    scenario: TrainingScenario,
    tracks: Sequence[TrackObservation],
    *,
    ball_diameter_mm: float = 57.15,
    ball_center_reachable_polygon_mm: Sequence[tuple[float, float]] | None = None,
    pockets_mm: Sequence[tuple[float, float]] | None = None,
) -> tuple[bool, str]:
    visible = [track for track in tracks if track.visibility == "visible"]
    numbered = {number: track for track in visible if (number := ball_number_from_track(track)) is not None}
    visible_objects = {number for number in numbered if number != 0}
    required = set(scenario.required_balls)

    if scenario.require_cue_ball and 0 not in numbered:
        return False, "未识别到白球（0 号）"
    missing = sorted(required - visible_objects)
    if missing:
        return False, "缺少目标球：" + "、".join(str(number) for number in missing)
    if not scenario.allow_extra_object_balls:
        extras = sorted(visible_objects - required)
        if extras:
            return False, "请移走本项目外的球：" + "、".join(str(number) for number in extras)

    layout_tracks = [numbered[number] for number in scenario.required_balls]
    if scenario.layout == "line":
        ok, message = _validate_line(
            layout_tracks,
            scenario.required_balls,
            ball_diameter_mm,
            tolerance=scenario.constraints.line_tolerance_ball_diameters,
            max_span=scenario.constraints.max_span_ball_diameters,
            min_spacing=scenario.constraints.min_spacing_ball_diameters,
        )
        if not ok:
            return ok, message
    elif scenario.layout == "double_row":
        ok, message = _validate_double_row(numbered, ball_diameter_mm)
        if not ok:
            return ok, message
    elif scenario.layout == "straight_shot":
        ok, message = _validate_straight_shot(
            numbered[0],
            numbered[1],
            pockets_mm or (),
            ball_diameter_mm,
            scenario.constraints.line_tolerance_ball_diameters,
            scenario.constraints.max_span_ball_diameters,
        )
        if not ok:
            return ok, message
    elif scenario.layout == "zones":
        ok, message = _validate_zones(
            scenario,
            numbered,
            ball_center_reachable_polygon_mm or (),
        )
        if not ok:
            return ok, message

    ok, message = _validate_safety(
        scenario,
        numbered,
        ball_diameter_mm,
        ball_center_reachable_polygon_mm or (),
        pockets_mm or (),
    )
    if not ok:
        return ok, message
    return True, READY_MESSAGE


def _track_points(tracks: Iterable[TrackObservation]) -> tuple[np.ndarray, float]:
    items = list(tracks)
    use_mm = bool(items) and all(track.center_mm is not None for track in items)
    if use_mm:
        points = np.asarray([track.center_mm for track in items], dtype=np.float32)
        scale = 57.15
    else:
        points = np.asarray([track.center_px for track in items], dtype=np.float32)
        radii = [max(2.0, float(track.radius_px)) for track in items]
        scale = 2.0 * float(np.median(radii)) if radii else 24.0
    return points.reshape((-1, 2)), scale


def _validate_line(
    tracks: Sequence[TrackObservation],
    expected_order: Sequence[int],
    ball_diameter_mm: float,
    *,
    tolerance: float = 1.35,
    max_span: float | None = None,
    min_spacing: float = 0.0,
) -> tuple[bool, str]:
    if len(tracks) < 3:
        return True, READY_MESSAGE
    points, observed_scale = _track_points(tracks)
    if all(track.center_mm is not None for track in tracks):
        observed_scale = max(1.0, float(ball_diameter_mm))
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    main_axis = axes[0]
    projections = centered @ main_axis
    residual = centered - np.outer(projections, main_axis)
    if float(np.max(np.linalg.norm(residual, axis=1))) > observed_scale * float(tolerance):
        return False, "目标球未摆成一条直线，请调整偏离直线的球"
    sorted_indexes = np.argsort(projections)
    actual = tuple(int(expected_order[index]) for index in sorted_indexes)
    expected = tuple(int(number) for number in expected_order)
    if actual not in {expected, tuple(reversed(expected))}:
        return False, "直线中的球号顺序不正确"
    sorted_projections = np.sort(projections)
    if max_span is not None and float(np.ptp(sorted_projections)) > observed_scale * float(max_span):
        return False, "目标球直线过长，请缩短球间距离"
    if min_spacing > 0 and np.any(np.diff(sorted_projections) < observed_scale * float(min_spacing)):
        return False, "目标球间距过近，请将球分开"
    return True, READY_MESSAGE


def _validate_double_row(
    numbered: dict[int, TrackObservation],
    ball_diameter_mm: float,
) -> tuple[bool, str]:
    odd_numbers = (1, 3, 5, 7, 9)
    even_numbers = (2, 4, 6, 8)
    for numbers in (odd_numbers, even_numbers):
        ok, _ = _validate_line([numbered[number] for number in numbers], numbers, ball_diameter_mm)
        if not ok:
            return False, "单双号球需要分别摆成两条直线"
    odd_points, odd_scale = _track_points([numbered[number] for number in odd_numbers])
    even_points, even_scale = _track_points([numbered[number] for number in even_numbers])
    scale = max(odd_scale, even_scale, float(ball_diameter_mm) if numbered[1].center_mm is not None else 1.0)
    if float(np.linalg.norm(np.mean(odd_points, axis=0) - np.mean(even_points, axis=0))) < scale * 1.5:
        return False, "单双号两排距离过近"
    return True, READY_MESSAGE


def _validate_straight_shot(
    cue: TrackObservation,
    target: TrackObservation,
    pockets_mm: Sequence[tuple[float, float]],
    ball_diameter_mm: float,
    tolerance: float,
    max_span: float | None,
) -> tuple[bool, str]:
    if cue.center_mm is None or target.center_mm is None or not pockets_mm:
        return True, READY_MESSAGE
    cue_point = np.asarray(cue.center_mm, dtype=np.float32)
    target_point = np.asarray(target.center_mm, dtype=np.float32)
    shot = target_point - cue_point
    shot_length = float(np.linalg.norm(shot))
    if shot_length < float(ball_diameter_mm) * 1.5:
        return False, "白球与 1 号球距离过近，请稍微分开"
    if max_span is not None and shot_length > float(ball_diameter_mm) * float(max_span):
        return False, "白球与 1 号球距离过远，请按中短距离重新摆放"
    direction = shot / shot_length
    best_error = float("inf")
    for pocket in pockets_mm:
        to_pocket = np.asarray(pocket, dtype=np.float32) - target_point
        forward = float(np.dot(to_pocket, direction))
        if forward <= 0:
            continue
        lateral = abs(float(direction[0] * to_pocket[1] - direction[1] * to_pocket[0]))
        best_error = min(best_error, lateral)
    if best_error > float(ball_diameter_mm) * float(tolerance):
        return False, "白球、1 号球与可用袋口方向明显不共线"
    return True, READY_MESSAGE


def _validate_zones(
    scenario: TrainingScenario,
    numbered: dict[int, TrackObservation],
    reachable_polygon: Sequence[tuple[float, float]],
) -> tuple[bool, str]:
    if len(reachable_polygon) < 3:
        return False, "缺少球心可达区域标定，无法校验三点摆球"
    polygon = np.asarray(reachable_polygon, dtype=np.float32).reshape((-1, 2))
    low = np.min(polygon, axis=0)
    span = np.maximum(np.ptp(polygon, axis=0), 1.0)
    for zone in scenario.zones:
        track = numbered[zone.ball]
        if track.center_mm is None:
            return False, f"无法取得 {zone.ball} 号球的球桌坐标"
        normalized = (np.asarray(track.center_mm, dtype=np.float32) - low) / span
        delta = np.abs(normalized - np.asarray(zone.center, dtype=np.float32))
        tolerance = np.maximum(np.asarray(zone.tolerance, dtype=np.float32), 1e-4)
        if float(np.sum((delta / tolerance) ** 2)) > 1.0:
            return False, f"{zone.ball} 号球未放在对应的推荐区域"
    return True, READY_MESSAGE


def _validate_safety(
    scenario: TrainingScenario,
    numbered: dict[int, TrackObservation],
    ball_diameter_mm: float,
    reachable_polygon: Sequence[tuple[float, float]],
    pockets_mm: Sequence[tuple[float, float]],
) -> tuple[bool, str]:
    constraints = scenario.constraints
    if (
        float(constraints.min_spacing_ball_diameters) <= 0
        and float(constraints.edge_margin_ball_diameters) <= 0
        and float(constraints.pocket_clearance_ball_diameters) <= 0
    ):
        return True, READY_MESSAGE
    relevant = [numbered[number] for number in ((0,) + scenario.required_balls) if number in numbered]
    if not relevant or not all(track.center_mm is not None for track in relevant):
        return True, READY_MESSAGE
    points = np.asarray([track.center_mm for track in relevant], dtype=np.float32)
    numbers = [ball_number_from_track(track) for track in relevant]
    polygon = np.asarray(reachable_polygon, dtype=np.float32).reshape((-1, 2)) if len(reachable_polygon) >= 3 else None
    if polygon is not None:
        for number, point in zip(numbers, points):
            if not _point_in_polygon(point, polygon):
                return False, f"{number} 号球不在有效击球区域内"
            margin = float(scenario.constraints.edge_margin_ball_diameters) * float(ball_diameter_mm)
            if margin > 0 and _distance_to_polygon(point, polygon) < margin:
                return False, f"{number} 号球过于贴近库边，请向台内移动"
    clearance = float(scenario.constraints.pocket_clearance_ball_diameters) * float(ball_diameter_mm)
    if clearance > 0 and pockets_mm:
        pockets = np.asarray(pockets_mm, dtype=np.float32).reshape((-1, 2))
        for number, point in zip(numbers, points):
            if float(np.min(np.linalg.norm(pockets - point, axis=1))) < clearance:
                return False, f"{number} 号球过于靠近袋口，请重新摆放"
    spacing = float(scenario.constraints.min_spacing_ball_diameters) * float(ball_diameter_mm)
    if spacing > 0:
        for left in range(len(points)):
            for right in range(left + 1, len(points)):
                if float(np.linalg.norm(points[left] - points[right])) < spacing:
                    return False, f"{numbers[left]} 号球与 {numbers[right]} 号球距离过近"
    return True, READY_MESSAGE


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _distance_to_polygon(point: np.ndarray, polygon: np.ndarray) -> float:
    best = float("inf")
    previous = polygon[-1]
    for current in polygon:
        edge = current - previous
        length_sq = float(np.dot(edge, edge))
        amount = 0.0 if length_sq <= 1e-9 else float(np.clip(np.dot(point - previous, edge) / length_sq, 0.0, 1.0))
        closest = previous + edge * amount
        best = min(best, float(np.linalg.norm(point - closest)))
        previous = current
    return best
