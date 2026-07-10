from __future__ import annotations

import json
import threading
import time
from urllib.request import Request, urlopen

import numpy as np

from bas.web_control import WebControlServer


def test_web_control_serves_2604_client_state_and_frame() -> None:
    bridge = WebControlServer(action_timeout_s=1.0)
    bridge.start("127.0.0.1", 0)
    stopped = threading.Event()
    seen: list[tuple[str, dict]] = []

    def handler(action: str, payload: dict) -> dict:
        seen.append((action, payload))
        if action == "state":
            return {"ok": True, "frame_idx": 7, "frame_size": {"w": 16, "h": 9}}
        return {"ok": True, "message": action, "state": {"frame_idx": 7}}

    def pump() -> None:
        while not stopped.is_set():
            bridge.process_actions(handler)
            time.sleep(0.005)

    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()
    base = f"http://127.0.0.1:{bridge.port}"
    try:
        with urlopen(f"{base}/health", timeout=2.0) as response:
            assert json.load(response)["ok"] is True
        with urlopen(f"{base}/api/state", timeout=2.0) as response:
            assert json.load(response)["frame_idx"] == 7
        with urlopen(f"{base}/", timeout=2.0) as response:
            assert "PRECISION AI Web Client" in response.read().decode("utf-8")

        request = Request(
            f"{base}/api/compute",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2.0) as response:
            assert json.load(response)["message"] == "compute"

        assert bridge.update_frame(np.zeros((9, 16, 3), dtype=np.uint8)) is True
        with urlopen(f"{base}/api/frame.jpg", timeout=2.0) as response:
            assert response.headers.get_content_type() == "image/jpeg"
            assert response.read(2) == b"\xff\xd8"
        assert ("state", {}) in seen
        assert ("compute", {}) in seen
    finally:
        stopped.set()
        bridge.stop()
        pump_thread.join(timeout=1.0)
