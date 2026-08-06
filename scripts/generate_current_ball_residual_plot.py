from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bas.ui.ball_compensation_residual_view import (
    render_training_residual_bubble_chart,
    save_residual_view,
)


DEFAULT_SETTINGS = PROJECT_ROOT / "local_settings" / "user_settings.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "local_settings"
    / "analysis"
    / "current_ball_compensation_residuals_53_points.png"
)
DEFAULT_EXCLUDED_POINTS = (16, 22, 56)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return payload


def resolve_compensation_file(settings_path: Path) -> Path:
    settings = load_json(settings_path)
    configured = settings.get("engineered_ball_compensation_file")
    if not configured:
        raise ValueError("当前设置未配置 engineered_ball_compensation_file")
    path = Path(str(configured))
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"球心补偿配置不存在: {path}")
    return path


def parse_point_numbers(value: str) -> tuple[int, ...]:
    """Parse one-based point numbers from a comma-separated CLI value."""

    if not value.strip():
        return ()
    numbers: list[int] = []
    for raw in value.replace("，", ",").split(","):
        point_number = int(raw.strip())
        if point_number <= 0:
            raise ValueError("排除点编号必须是大于 0 的整数")
        if point_number not in numbers:
            numbers.append(point_number)
    return tuple(numbers)


def report_excluding_points(
    compensation: dict[str, Any],
    *,
    excluded_point_numbers: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a residual report excluding explicit one-based source point numbers."""

    report = compensation.get("training_residual_report")
    if not isinstance(report, dict):
        raise ValueError("球心补偿配置缺少 training_residual_report")
    samples = report.get("samples")
    if not isinstance(samples, list):
        raise ValueError("training_residual_report.samples 必须是数组")

    excluded_set = set(excluded_point_numbers)
    shown: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    for item in samples:
        if not isinstance(item, dict):
            continue
        point_number = int(item.get("sample_index", -1)) + 1
        (omitted if point_number in excluded_set else shown).append(item)

    found = {int(item.get("sample_index", -1)) + 1 for item in omitted}
    missing = sorted(excluded_set - found)
    if missing:
        joined = ", ".join(str(point) for point in missing)
        raise ValueError(f"配置中找不到要排除的原始点位: {joined}")
    return {**report, "sample_count": len(shown), "samples": shown}, omitted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按当前 user_settings 生成球心补偿训练残差空间点状图。",
    )
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exclude-points",
        default=",".join(str(point) for point in DEFAULT_EXCLUDED_POINTS),
        help="按原始一基点位编号排除，逗号分隔；默认排除 16、22、56。",
    )
    parser.add_argument(
        "--expect-count",
        type=int,
        default=53,
        help="展示点数不符时失败；设为 0 可关闭固定点数校验。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings_path = args.settings.resolve()
    compensation_path = resolve_compensation_file(settings_path)
    compensation = load_json(compensation_path)
    excluded_point_numbers = parse_point_numbers(str(args.exclude_points))
    report, omitted = report_excluding_points(
        compensation,
        excluded_point_numbers=excluded_point_numbers,
    )
    shown_count = len(report["samples"])
    if int(args.expect_count) > 0 and shown_count != int(args.expect_count):
        raise ValueError(
            f"展示点数为 {shown_count}，与期望的 {int(args.expect_count)} 不一致；"
            "请确认当前补偿配置、排除点编号或调整 --expect-count。"
        )

    quality = compensation.get("quality_report", {})
    table_span = quality.get("table_span_mm", {}) if isinstance(quality, dict) else {}
    table_width_mm = max(2540.0, float(table_span.get("width", 0.0)))
    table_height_mm = max(1270.0, float(table_span.get("height", 0.0)))
    image = render_training_residual_bubble_chart(
        table_width_mm,
        table_height_mm,
        report,
    )
    output = save_residual_view(args.output.resolve(), image)

    omitted_summary = ", ".join(
        f"#{int(item.get('sample_index', -1)) + 1}={float(item.get('error_mm', 0.0)):.3f}mm"
        for item in omitted
    )
    print(f"补偿配置: {compensation_path}")
    print(f"已生成: {output}")
    print(f"展示点数: {shown_count}; 排除点: {omitted_summary or '无'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
