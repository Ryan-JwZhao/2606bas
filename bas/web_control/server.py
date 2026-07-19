from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import cv2
import numpy as np


ActionHandler = Callable[[str, Dict[str, Any]], Dict[str, Any]]


@dataclass
class _PendingAction:
    action: str
    payload: Dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[Dict[str, Any]] = None


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class WebControlServer:
    """2604-compatible LAN HTTP bridge with main-thread action dispatch."""

    POST_ACTIONS = {
        "/select": "select",
        "/selection/clear": "selection_clear",
        "/camera/refresh": "camera_refresh",
        "/camera/select_default": "camera_select_default",
        "/capture/start": "capture_start",
        "/workflow/industrial_camera_init": "camera_init",
        "/pipeline/init": "pipeline_init",
        "/workflow/system_init": "pipeline_init",
        "/projection/start": "projection_start",
        "/workflow/projection_start": "projection_start",
        "/match/start": "match_start",
        "/match/stage": "match_stage",
        "/match/switch_turn": "match_switch_turn",
        "/match/undo": "match_undo",
        "/shot_mode/toggle": "shot_mode_toggle",
        "/shot_mode/set": "shot_mode_set",
        "/runtime_mode/set": "runtime_mode_set",
        "/training/scenario/set": "training_scenario_set",
        "/training/start": "training_start",
        "/training/reset": "training_reset",
        "/shot_once/free/arm": "shot_once_free_arm",
        "/shot_once/free/clear": "shot_once_free_clear",
        "/shot_once/free/toggle": "shot_once_free_toggle",
        "/shot_once/hook/arm": "shot_once_hook_arm",
        "/shot_once/hook/clear": "shot_once_hook_clear",
        "/shot_once/hook/toggle": "shot_once_hook_toggle",
        "/shot_once/black/arm": "shot_once_black_arm",
        "/shot_once/black/clear": "shot_once_black_clear",
        "/shot_once/black/toggle": "shot_once_black_toggle",
        "/star_formula/toggle": "star_formula_toggle",
        "/star_formula/set": "star_formula_set",
        "/instant_replay/export": "instant_replay_export",
        "/compute": "compute",
        "/detect": "detect",
    }

    def __init__(self, *, jpeg_max_hz: float = 12.0, jpeg_max_side: int = 1280, action_timeout_s: float = 4.0):
        self.jpeg_max_hz = max(1.0, float(jpeg_max_hz))
        self.jpeg_max_side = max(320, int(jpeg_max_side))
        self.action_timeout_s = max(0.1, float(action_timeout_s))
        self._server: Optional[_ReusableThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._actions: queue.Queue[_PendingAction] = queue.Queue()
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._last_jpeg_encode_ts = 0.0
        self._host = "0.0.0.0"
        self._port = 17070
        asset_dir = Path(__file__).parent
        self._client_assets = {
            "/": ("text/html; charset=utf-8", (asset_dir / "mobile_web_client.html").read_bytes()),
            "/mobile_web_client.css": ("text/css; charset=utf-8", (asset_dir / "mobile_web_client.css").read_bytes()),
            "/stream_coordinates.js": ("text/javascript; charset=utf-8", (asset_dir / "stream_coordinates.js").read_bytes()),
            "/mobile_web_client.js": ("text/javascript; charset=utf-8", (asset_dir / "mobile_web_client.js").read_bytes()),
            "/manifest.webmanifest": ("application/manifest+json; charset=utf-8", (asset_dir / "manifest.webmanifest").read_bytes()),
            "/service-worker.js": ("text/javascript; charset=utf-8", (asset_dir / "service-worker.js").read_bytes()),
            "/pwa/icon-192.svg": ("image/svg+xml; charset=utf-8", (asset_dir / "pwa" / "icon-192.svg").read_bytes()),
            "/pwa/icon-512.svg": ("image/svg+xml; charset=utf-8", (asset_dir / "pwa" / "icon-512.svg").read_bytes()),
        }

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def start(self, host: str = "0.0.0.0", port: int = 17070) -> None:
        if self.is_running:
            return
        bind_host = str(host or "0.0.0.0").strip() or "0.0.0.0"
        bind_port = int(port)
        if not 0 <= bind_port <= 65535:
            raise ValueError(f"Web 控制端口无效: {bind_port}")
        handler = self._handler_class()
        server = _ReusableThreadingHTTPServer((bind_host, bind_port), handler)
        self._host = bind_host
        self._port = int(server.server_address[1])
        self._stop_event.clear()
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="bas-web-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is not None:
            try:
                server.shutdown()
            finally:
                server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._release_pending("Web 控制服务已停止")

    def update_frame(self, frame_bgr: Optional[np.ndarray]) -> bool:
        if not self.is_running or frame_bgr is None or frame_bgr.size == 0:
            return False
        now = time.perf_counter()
        if now - self._last_jpeg_encode_ts < 1.0 / self.jpeg_max_hz:
            return False
        frame = frame_bgr
        h, w = frame.shape[:2]
        long_side = max(h, w)
        if long_side > self.jpeg_max_side:
            scale = float(self.jpeg_max_side) / float(long_side)
            frame = cv2.resize(
                frame,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return False
        with self._jpeg_lock:
            self._latest_jpeg = bytes(encoded.tobytes())
        self._last_jpeg_encode_ts = now
        return True

    def process_actions(self, handler: ActionHandler, *, limit: int = 32) -> int:
        processed = 0
        for _ in range(max(1, int(limit))):
            try:
                pending = self._actions.get_nowait()
            except queue.Empty:
                break
            try:
                result = handler(pending.action, pending.payload)
                pending.result = result if isinstance(result, dict) else {"ok": False, "message": "请求结果无效"}
            except Exception as exc:
                pending.result = {"ok": False, "message": f"处理失败: {exc}"}
            finally:
                pending.event.set()
            processed += 1
        return processed

    def _request_action(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        pending = _PendingAction(action=action, payload=dict(payload))
        self._actions.put(pending)
        if not pending.event.wait(self.action_timeout_s):
            return {"ok": False, "message": "请求超时"}
        return pending.result or {"ok": False, "message": "请求结果缺失"}

    def _get_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._latest_jpeg

    def _release_pending(self, message: str) -> None:
        while True:
            try:
                pending = self._actions.get_nowait()
            except queue.Empty:
                return
            pending.result = {"ok": False, "message": message}
            pending.event.set()

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "BASWebBridge/1.0"

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

            def _send_headers(self, status: int, content_type: str, length: Optional[int] = None) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                if length is not None:
                    self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

            def _send_json(self, status: int, obj: Dict[str, Any]) -> None:
                data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self._send_headers(status, "application/json; charset=utf-8", len(data))
                self.wfile.write(data)

            def _read_json(self) -> Dict[str, Any]:
                try:
                    length = min(max(0, int(self.headers.get("Content-Length", "0"))), 1024 * 1024)
                    data = self.rfile.read(length) if length else b""
                    obj = json.loads(data.decode("utf-8")) if data else {}
                except Exception:
                    return {}
                return obj if isinstance(obj, dict) else {}

            def _serve_client_asset(self, asset_path: str) -> None:
                content_type, data = bridge._client_assets[asset_path]
                self._send_headers(200, content_type, len(data))
                self.wfile.write(data)

            def _serve_jpeg(self) -> None:
                jpg = bridge._get_jpeg()
                if jpg is None:
                    self._send_json(503, {"ok": False, "message": "尚无可用画面"})
                    return
                self._send_headers(200, "image/jpeg", len(jpg))
                self.wfile.write(jpg)

            def _serve_mjpeg(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    while not bridge._stop_event.is_set():
                        jpg = bridge._get_jpeg()
                        if jpg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                path = bridge._normalize_path(urlparse(self.path).path)
                if path in {"/", "/index.html"}:
                    self._serve_client_asset("/")
                elif path in bridge._client_assets:
                    self._serve_client_asset(path)
                elif path == "/health":
                    self._send_json(200, {"ok": True, "service": "billiards-assistance-system"})
                elif path == "/state":
                    self._send_json(200, bridge._request_action("state", {}))
                elif path == "/frame.jpg":
                    self._serve_jpeg()
                elif path == "/mjpeg":
                    self._serve_mjpeg()
                else:
                    self._send_json(404, {"ok": False, "message": f"not found: {path}"})

            def do_POST(self) -> None:  # noqa: N802
                path = bridge._normalize_path(urlparse(self.path).path)
                action = bridge.POST_ACTIONS.get(path)
                if action is None:
                    self._send_json(404, {"ok": False, "message": f"not found: {path}"})
                    return
                self._send_json(200, bridge._request_action(action, self._read_json()))

        return Handler

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = "/" + str(path or "/").lstrip("/")
        if normalized == "/api" or normalized.startswith("/api/"):
            normalized = normalized[4:] or "/"
        if len(normalized) > 1:
            normalized = normalized.rstrip("/")
        return normalized
