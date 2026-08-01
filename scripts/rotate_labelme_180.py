from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


def rotate_labelme_document_180(document: dict[str, Any]) -> dict[str, Any]:
    """Return a Labelme document whose annotation points are rotated by 180°."""

    width = _positive_dimension(document, "imageWidth")
    height = _positive_dimension(document, "imageHeight")
    result = copy.deepcopy(document)
    shapes = result.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("Labelme document must contain a shapes list")

    for shape_index, shape in enumerate(shapes):
        if not isinstance(shape, dict) or not isinstance(shape.get("points"), list):
            raise ValueError(f"shape {shape_index} must contain a points list")
        rotated_points: list[list[float]] = []
        for point_index, point in enumerate(shape["points"]):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(
                    f"shape {shape_index} point {point_index} must be an [x, y] pair"
                )
            x, y = float(point[0]), float(point[1])
            rotated_points.append([float(width - 1) - x, float(height - 1) - y])
        shape["points"] = rotated_points
    return result


def rotate_labelme_file_180(path: Path, *, backup_suffix: str = ".pre180.bak") -> Path:
    """Rotate one file atomically and retain a one-time recovery copy."""

    path = path.resolve()
    backup = path.with_name(path.name + backup_suffix)
    if backup.exists():
        raise FileExistsError(
            f"backup already exists, refusing a possible second rotation: {backup}"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    rotated = rotate_labelme_document_180(document)

    shutil.copy2(path, backup)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(rotated, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            temp_path = Path(stream.name)
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        shutil.copy2(backup, path)
        raise
    return backup


def _positive_dimension(document: dict[str, Any], key: str) -> int:
    try:
        value = int(document[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Labelme document has invalid {key}") from exc
    if value <= 0:
        raise ValueError(f"Labelme document has invalid {key}")
    return value


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 Labelme JSON 中的全部标注点绕图像中心旋转 180°。"
    )
    parser.add_argument("paths", nargs="+", type=Path, help="一个或多个 Labelme JSON 文件")
    parser.add_argument(
        "--backup-suffix",
        default=".pre180.bak",
        help="原文件备份后缀（默认：.pre180.bak）",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    for path in args.paths:
        backup = rotate_labelme_file_180(path, backup_suffix=args.backup_suffix)
        print(f"已旋转: {path.resolve()}（原文件备份: {backup}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
