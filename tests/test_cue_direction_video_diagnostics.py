from __future__ import annotations

import json

import cv2
import numpy as np

from bas.planning.cue_aim import CueStickAimPx
from bas.video_diagnostics.cue_direction import diagnose_cue_direction_video


def test_cue_direction_video_diagnosis_writes_pass_report(tmp_path, monkeypatch) -> None:
    video = tmp_path / "cue.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 64),
    )
    assert writer.isOpened()
    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "video_basename": video.name,
                "cases": [
                    {
                        "name": "right",
                        "frame": 0,
                        "cue_center_px": [32, 32],
                        "cue_radius_px": 4,
                        "expected_direction_px": [1, 0],
                        "min_dot": 0.99,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "bas.video_diagnostics.cue_direction.CueStickAimDetector.detect",
        lambda self, **kwargs: CueStickAimPx(
            tip_px=np.asarray([30.0, 32.0], dtype=np.float32),
            tail_px=np.asarray([10.0, 32.0], dtype=np.float32),
            direction_px=np.asarray([1.0, 0.0], dtype=np.float32),
            source="test",
            score=1.0,
        ),
    )
    output = tmp_path / "report.json"
    preview = tmp_path / "preview.jpg"

    report = diagnose_cue_direction_video(
        video_path=video,
        annotations_path=labels,
        output_path=output,
        preview_path=preview,
    )

    assert report["verdict"] == "PASS"
    assert report["video"]["width"] == 64
    assert output.exists()
    assert preview.exists()
