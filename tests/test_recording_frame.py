from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bas.recording_frame import RecordingFrameCorrector


def _write_calibration(path: Path, *, k1: float) -> None:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot create calibration file: {path}")
    try:
        fs.write("image_width", 6)
        fs.write("image_height", 6)
        fs.write("camera_matrix", np.array([[5.0, 0.0, 3.0], [0.0, 5.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float64))
        fs.write("distortion_coefficients", np.array([[k1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64))
    finally:
        fs.release()


def test_recording_frame_corrector_uses_valid_fallback_calibration(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    fallback = tmp_path / "fallback.yaml"
    _write_calibration(fallback, k1=0.35)
    corrector = RecordingFrameCorrector([str(missing), str(fallback)])

    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    frame[2:4, 2:4] = 255
    corrected = corrector.corrected_frame(frame, already_corrected=False)

    assert corrected is not None
    assert corrector.has_usable_calibration() is True
    assert np.array_equal(corrected, frame) is False


def test_recording_frame_corrector_passthroughs_already_corrected_frame(tmp_path: Path) -> None:
    calibration = tmp_path / "camera.yaml"
    _write_calibration(calibration, k1=0.35)
    corrector = RecordingFrameCorrector([str(calibration)])

    frame = np.ones((6, 6, 3), dtype=np.uint8)
    corrected = corrector.corrected_frame(frame, already_corrected=True)

    assert corrected is frame


def test_recording_frame_corrector_rejects_uncorrected_frame_without_calibration(tmp_path: Path) -> None:
    corrector = RecordingFrameCorrector([str(tmp_path / "missing.yaml")])

    frame = np.ones((6, 6, 3), dtype=np.uint8)
    corrected = corrector.corrected_frame(frame, already_corrected=False)

    assert corrector.has_usable_calibration() is False
    assert corrected is None
