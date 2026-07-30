from __future__ import annotations

import cv2

from bas.capture import opencv_capture


class _FakeVideoCapture:
    def __init__(self, *, opened: bool, fourcc: int) -> None:
        self._opened = opened
        self._fourcc = fourcc
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FOURCC:
            return float(self._fourcc)
        return 0.0

    def release(self) -> None:
        self.released = True


class _ExplodingVideoCapture:
    def __call__(self, *_args):
        raise RuntimeError("backend does not support atomic parameters")


def test_open_device_negotiates_mjpg_atomically_and_rejects_yuy2(monkeypatch) -> None:
    yuy2 = _FakeVideoCapture(opened=True, fourcc=cv2.VideoWriter_fourcc(*"YUY2"))
    mjpg = _FakeVideoCapture(opened=True, fourcc=cv2.VideoWriter_fourcc(*"MJPG"))
    opened: list[tuple[int, int, list[int]]] = []

    def fake_open(index: int, api: int, params: list[int]):
        opened.append((index, api, params))
        return yuy2 if len(opened) == 1 else mjpg

    monkeypatch.setattr(opencv_capture.cv2, "VideoCapture", fake_open)
    monkeypatch.setattr(opencv_capture, "camera_api_candidates", lambda: [101, 202])

    capture = opencv_capture._open_device(3, width=1920, height=1080, fps=30)

    assert capture is mjpg
    assert yuy2.released is True
    assert opened == [
        (3, 101, opencv_capture.mjpg_open_params(1920, 1080, 30)),
        (3, 202, opencv_capture.mjpg_open_params(1920, 1080, 30)),
    ]


def test_open_device_never_returns_a_non_mjpg_stream(monkeypatch) -> None:
    opened_streams = [
        _FakeVideoCapture(opened=True, fourcc=cv2.VideoWriter_fourcc(*"YUY2")),
        _FakeVideoCapture(opened=True, fourcc=cv2.VideoWriter_fourcc(*"YUV2")),
    ]
    pending_streams = list(opened_streams)

    monkeypatch.setattr(opencv_capture, "camera_api_candidates", lambda: [101, 202])
    monkeypatch.setattr(opencv_capture.cv2, "VideoCapture", lambda *_args: pending_streams.pop(0))

    capture = opencv_capture._open_device(0, width=1920, height=1080, fps=30)

    assert capture is None
    assert all(stream.released for stream in opened_streams) is True


def test_open_device_treats_parameter_negotiation_errors_as_an_unusable_backend(monkeypatch) -> None:
    monkeypatch.setattr(opencv_capture, "camera_api_candidates", lambda: [101])
    monkeypatch.setattr(opencv_capture.cv2, "VideoCapture", _ExplodingVideoCapture())

    assert opencv_capture._open_device(0, width=1920, height=1080, fps=30) is None
