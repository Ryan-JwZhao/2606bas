from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from bas.geometry_runtime import RuntimeGeometryReloader


def _write_labelme(path: Path, label: str, points: list[list[float]]) -> None:
    path.write_text(
        json.dumps(
            {
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": [{"label": label, "points": points}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_runtime_geometry_reloader_detects_same_path_content_replacement(tmp_path) -> None:
    inline_path = tmp_path / "inline.json"
    _write_labelme(inline_path, "inline", [[10, 20], [90, 20]])

    reloader = RuntimeGeometryReloader()
    geometry, changed = reloader.refresh(None, str(inline_path), None)

    assert changed is True
    assert len(geometry.inline_norm) == 1
    np.testing.assert_allclose(geometry.inline_norm[0], np.array([[0.1, 0.2], [0.9, 0.2]], dtype=np.float32))

    geometry, changed = reloader.refresh(None, str(inline_path), None)
    assert changed is False
    np.testing.assert_allclose(geometry.inline_norm[0], np.array([[0.1, 0.2], [0.9, 0.2]], dtype=np.float32))

    _write_labelme(inline_path, "inline", [[10, 60], [90, 60], [95, 60]])

    geometry, changed = reloader.refresh(None, str(inline_path), None)
    assert changed is True
    assert len(geometry.inline_norm) == 1
    np.testing.assert_allclose(
        geometry.inline_norm[0],
        np.array([[0.1, 0.6], [0.9, 0.6], [0.95, 0.6]], dtype=np.float32),
    )


def test_runtime_geometry_reloader_hashes_content_when_size_and_mtime_are_unchanged(tmp_path) -> None:
    inline_path = tmp_path / "inline.json"
    _write_labelme(inline_path, "inline", [[10, 20], [90, 20]])
    original_stat = inline_path.stat()
    reloader = RuntimeGeometryReloader()
    reloader.refresh(None, str(inline_path), None)

    _write_labelme(inline_path, "inline", [[10, 60], [90, 60]])
    assert inline_path.stat().st_size == original_stat.st_size
    os.utime(
        inline_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    geometry, changed = reloader.refresh(None, str(inline_path), None)

    assert changed is True
    np.testing.assert_allclose(
        geometry.inline_norm[0],
        np.array([[0.1, 0.6], [0.9, 0.6]], dtype=np.float32),
    )


def test_runtime_geometry_reloader_detects_path_switch(tmp_path) -> None:
    inline_a = tmp_path / "inline_a.json"
    inline_b = tmp_path / "inline_b.json"
    _write_labelme(inline_a, "inline", [[0, 10], [100, 10]])
    _write_labelme(inline_b, "inline", [[0, 80], [100, 80]])

    reloader = RuntimeGeometryReloader()
    geometry, changed = reloader.refresh(None, str(inline_a), None)

    assert changed is True
    np.testing.assert_allclose(geometry.inline_norm[0], np.array([[0.0, 0.1], [1.0, 0.1]], dtype=np.float32))

    geometry, changed = reloader.refresh(None, str(inline_b), None)
    assert changed is True
    np.testing.assert_allclose(geometry.inline_norm[0], np.array([[0.0, 0.8], [1.0, 0.8]], dtype=np.float32))


def test_runtime_geometry_reloader_keeps_last_valid_geometry_during_partial_write(tmp_path) -> None:
    inline_path = tmp_path / "inline.json"
    _write_labelme(inline_path, "inline", [[10, 20], [90, 20]])
    reloader = RuntimeGeometryReloader()
    original, changed = reloader.refresh(None, str(inline_path), None)
    assert changed is True

    inline_path.write_text('{"imageWidth": 100, "shapes": [', encoding="utf-8")
    retained, changed = reloader.refresh(None, str(inline_path), None)

    assert changed is False
    np.testing.assert_allclose(retained.inline_norm[0], original.inline_norm[0])

    _write_labelme(inline_path, "inline", [[10, 60], [90, 60]])
    refreshed, changed = reloader.refresh(None, str(inline_path), None)

    assert changed is True
    np.testing.assert_allclose(
        refreshed.inline_norm[0],
        np.array([[0.1, 0.6], [0.9, 0.6]], dtype=np.float32),
    )


def test_runtime_geometry_reloader_reports_not_ready_on_cold_invalid_geometry(tmp_path) -> None:
    missing_inline = tmp_path / "missing-inline.json"
    reloader = RuntimeGeometryReloader()

    geometry, changed = reloader.refresh(None, str(missing_inline), None)

    assert geometry.is_empty is True
    assert changed is False
    assert reloader.is_ready is False

    _write_labelme(missing_inline, "inline", [[10, 20], [90, 20]])
    geometry, changed = reloader.refresh(None, str(missing_inline), None)

    assert changed is True
    assert reloader.is_ready is True
    assert geometry.inline_norm
