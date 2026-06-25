from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import CaptureInfo

NORI_OK = 0x0000
NORI_E_REPEAT = 0x8001
NORI_USB_DEVICE = 0x0001
NORI_MAX_PATH = 260
CAMERA_CONTROL_EXPOSURE = 4
CAMERA_CONTROL_FLAGS_AUTO = 1
CAMERA_CONTROL_FLAGS_MANUAL = 2

VIDEO_MEDIA_TYPE_MJPG = 0x47504A4D
VIDEO_MEDIA_TYPE_MJPG_TO_BGR24 = VIDEO_MEDIA_TYPE_MJPG + 0x01


class _NoriDeviceInfoStruct(ctypes.Structure):
    _fields_ = [
        ("idVendor", ctypes.c_ushort),
        ("idProduct", ctypes.c_ushort),
        ("bcdDevice", ctypes.c_ushort),
        ("iManufacturer", ctypes.c_ubyte * NORI_MAX_PATH),
        ("iProduct", ctypes.c_ubyte * NORI_MAX_PATH),
        ("iSerialNumber", ctypes.c_ubyte * NORI_MAX_PATH),
        ("bcdUSB", ctypes.c_ulong),
        ("portNum", ctypes.c_ushort),
        ("DevNum", ctypes.c_ushort),
        ("hubName", ctypes.c_ubyte * NORI_MAX_PATH),
        ("PDO", ctypes.c_ubyte * NORI_MAX_PATH),
        ("deviceID", ctypes.c_ubyte * NORI_MAX_PATH),
        ("friendlyName", ctypes.c_ubyte * NORI_MAX_PATH),
        ("Reserved", ctypes.c_ushort),
    ]


class _NoriVideoInfoStruct(ctypes.Structure):
    _fields_ = [
        ("u_Format", ctypes.c_uint32),
        ("u_Width", ctypes.c_uint32),
        ("u_Height", ctypes.c_uint32),
        ("f_Fps", ctypes.c_float),
    ]


class _NoriFramePixelFormatStruct(ctypes.Structure):
    _fields_ = [
        ("f_Fps", ctypes.c_float),
        ("u_Format", ctypes.c_uint32),
        ("u_Width", ctypes.c_uint32),
        ("u_Height", ctypes.c_uint32),
    ]


class _NoriFrameBufferOutStruct(ctypes.Structure):
    _fields_ = [
        ("pBufAddr", ctypes.POINTER(ctypes.c_ubyte)),
        ("u_FrameLen", ctypes.c_uint32),
        ("u_FrameNum", ctypes.c_uint64),
        ("Frame_Time", wintypes.FILETIME),
        ("PixFormat", _NoriFramePixelFormatStruct),
        ("capacity", ctypes.c_uint32),
    ]


def _decode_nori_str(buf: Any) -> str:
    raw = bytes(bytearray(buf)).split(b"\x00", maxsplit=1)[0]
    if not raw:
        return ""
    for enc in ("utf-8", "gbk", "latin1"):
        try:
            return raw.decode(enc, errors="strict")
        except Exception:
            continue
    return raw.decode("latin1", errors="ignore")


@dataclass
class NoriDeviceInfo:
    device_id: int
    product: str
    manufacturer: str
    serial_number: str

    @property
    def display_name(self) -> str:
        parts = [self.product.strip(), self.serial_number.strip()]
        return " | ".join(p for p in parts if p) or f"Nori device {self.device_id}"


@dataclass
class NoriVideoInfo:
    device_id: int
    format_code: int
    width: int
    height: int
    fps: float


class NoriProtocolController:
    def __init__(self, sdk_root: str | Path | None = None):
        self.sdk_root = Path(sdk_root).resolve() if sdk_root else None
        self.dll_path = self._resolve_dll_path()
        self._dll = None
        self._dll_dir_handle = None
        self._initialized = False
        self._active_video_devices: set[int] = set()

    def _resolve_dll_path(self) -> Optional[Path]:
        roots: List[Path] = []
        if self.sdk_root:
            roots.append(self.sdk_root)
        roots.extend(
            [
                Path.cwd() / "example" / "Nori_Xvision_Development_Kit_Ver10.00.06_Windows",
                Path("C:/CodeProject/2606BAS/example/Nori_Xvision_Development_Kit_Ver10.00.06_Windows"),
            ]
        )
        is_64 = ctypes.sizeof(ctypes.c_void_p) == 8
        for root in roots:
            if not root.exists():
                continue
            candidates = [
                root / "Libraries" / ("win64" if is_64 else "win32") / ("Nori_Xvision_API_x64.dll" if is_64 else "Nori_Xvision_API.dll"),
                root / "Samples" / "C++" / ("x64" if is_64 else "") / "Release" / ("Nori_Xvision_API_x64.dll" if is_64 else "Nori_Xvision_API.dll"),
                root / "Samples" / "C++" / ("x64" if is_64 else "") / "Debug" / ("Nori_Xvision_API_x64.dll" if is_64 else "Nori_Xvision_API.dll"),
            ]
            for path in candidates:
                if path.exists():
                    return path
        return None

    def _bind_api(self) -> None:
        assert self._dll is not None
        self._dll.Nori_Xvision_Init.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        self._dll.Nori_Xvision_Init.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_UnInit.argtypes = []
        self._dll.Nori_Xvision_UnInit.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_GetDeviceInfo.argtypes = [ctypes.c_uint32, ctypes.POINTER(_NoriDeviceInfoStruct)]
        self._dll.Nori_Xvision_GetDeviceInfo.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_GetDeviceVideoInfoSize.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        self._dll.Nori_Xvision_GetDeviceVideoInfoSize.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_GetDeviceVideoInfo.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(_NoriVideoInfoStruct)]
        self._dll.Nori_Xvision_GetDeviceVideoInfo.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_DeviceVideoInit.argtypes = [ctypes.c_uint32, _NoriVideoInfoStruct]
        self._dll.Nori_Xvision_DeviceVideoInit.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_DeviceVideoUnInit.argtypes = [ctypes.c_uint32]
        self._dll.Nori_Xvision_DeviceVideoUnInit.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_VideoStart.argtypes = [ctypes.c_uint32]
        self._dll.Nori_Xvision_VideoStart.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_VideoStop.argtypes = [ctypes.c_uint32]
        self._dll.Nori_Xvision_VideoStop.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_GetFrameBuff.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.POINTER(_NoriFrameBufferOutStruct)),
            ctypes.c_uint32,
        ]
        self._dll.Nori_Xvision_GetFrameBuff.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_FreeFrameBuff.argtypes = [ctypes.c_uint32, ctypes.POINTER(_NoriFrameBufferOutStruct)]
        self._dll.Nori_Xvision_FreeFrameBuff.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_GetCameraTerminalControl.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_long),
        ]
        self._dll.Nori_Xvision_GetCameraTerminalControl.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_SetCameraTerminalControl.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_long,
            ctypes.c_long,
        ]
        self._dll.Nori_Xvision_SetCameraTerminalControl.restype = ctypes.c_uint32
        self._dll.Nori_Xvision_SetSingleAutoExposure.argtypes = [ctypes.c_uint32]
        self._dll.Nori_Xvision_SetSingleAutoExposure.restype = ctypes.c_uint32

    def _ensure_loaded(self) -> None:
        if self._dll is not None:
            return
        if sys.platform != "win32":
            raise RuntimeError("Nori SDK capture is only supported on Windows.")
        if self.dll_path is None:
            raise RuntimeError("Nori SDK DLL not found. Set camera.nori_sdk_root in the config.")
        if hasattr(os, "add_dll_directory"):
            self._dll_dir_handle = os.add_dll_directory(str(self.dll_path.parent))
        self._dll = ctypes.WinDLL(str(self.dll_path))
        self._bind_api()

    def ensure_initialized(self) -> None:
        self._ensure_loaded()
        if self._initialized:
            return
        assert self._dll is not None
        device_num = ctypes.c_uint32(0)
        ret = int(self._dll.Nori_Xvision_Init(NORI_USB_DEVICE, ctypes.byref(device_num)))
        if ret not in (NORI_OK, NORI_E_REPEAT):
            raise RuntimeError(f"Nori_Xvision_Init failed: 0x{ret:04X}")
        self._initialized = True

    def list_devices(self, max_scan: int = 16) -> List[NoriDeviceInfo]:
        self.ensure_initialized()
        assert self._dll is not None
        devices: List[NoriDeviceInfo] = []
        for idx in range(max_scan):
            info = _NoriDeviceInfoStruct()
            ret = int(self._dll.Nori_Xvision_GetDeviceInfo(ctypes.c_uint32(idx), ctypes.byref(info)))
            if ret != NORI_OK:
                continue
            devices.append(
                NoriDeviceInfo(
                    device_id=idx,
                    product=_decode_nori_str(info.iProduct),
                    manufacturer=_decode_nori_str(info.iManufacturer),
                    serial_number=_decode_nori_str(info.iSerialNumber),
                )
            )
        return devices

    def list_video_infos(self, device_id: int) -> List[NoriVideoInfo]:
        self.ensure_initialized()
        assert self._dll is not None
        count = ctypes.c_uint32(0)
        ret = int(self._dll.Nori_Xvision_GetDeviceVideoInfoSize(ctypes.c_uint32(int(device_id)), ctypes.byref(count)))
        if ret != NORI_OK:
            raise RuntimeError(f"Nori_Xvision_GetDeviceVideoInfoSize failed: 0x{ret:04X}")
        items: List[NoriVideoInfo] = []
        for idx in range(int(count.value)):
            info = _NoriVideoInfoStruct()
            ret = int(
                self._dll.Nori_Xvision_GetDeviceVideoInfo(
                    ctypes.c_uint32(int(device_id)),
                    ctypes.c_uint32(idx),
                    ctypes.byref(info),
                )
            )
            if ret != NORI_OK:
                continue
            if int(info.u_Format) != VIDEO_MEDIA_TYPE_MJPG:
                continue
            items.append(
                NoriVideoInfo(
                    device_id=int(device_id),
                    format_code=int(info.u_Format),
                    width=int(info.u_Width),
                    height=int(info.u_Height),
                    fps=float(info.f_Fps),
                )
            )
        return items

    def select_video_info(self, device_id: int, width: int, height: int, fps: float) -> Optional[NoriVideoInfo]:
        infos = self.list_video_infos(device_id)
        if not infos:
            return None

        def score(info: NoriVideoInfo) -> Tuple[int, float, float]:
            same_resolution = 1 if info.width == int(width) and info.height == int(height) else 0
            fps_penalty = abs(float(info.fps) - float(fps))
            pixel_penalty = abs(info.width - int(width)) + abs(info.height - int(height))
            return (same_resolution, -fps_penalty, -pixel_penalty)

        return sorted(infos, key=score, reverse=True)[0]

    def open_video_stream(self, device_id: int, video_info: NoriVideoInfo) -> None:
        self.ensure_initialized()
        assert self._dll is not None
        request = _NoriVideoInfoStruct(
            u_Format=ctypes.c_uint32(VIDEO_MEDIA_TYPE_MJPG_TO_BGR24),
            u_Width=ctypes.c_uint32(int(video_info.width)),
            u_Height=ctypes.c_uint32(int(video_info.height)),
            f_Fps=ctypes.c_float(float(video_info.fps)),
        )
        self.close_video_stream(device_id, ignore_errors=True)
        ret = int(self._dll.Nori_Xvision_DeviceVideoInit(ctypes.c_uint32(int(device_id)), request))
        if ret != NORI_OK:
            raise RuntimeError(f"Nori_Xvision_DeviceVideoInit failed for MJPG: 0x{ret:04X}")
        ret = int(self._dll.Nori_Xvision_VideoStart(ctypes.c_uint32(int(device_id))))
        if ret != NORI_OK:
            try:
                self._dll.Nori_Xvision_DeviceVideoUnInit(ctypes.c_uint32(int(device_id)))
            except Exception:
                pass
            raise RuntimeError(f"Nori_Xvision_VideoStart failed: 0x{ret:04X}")
        self._active_video_devices.add(int(device_id))

    def get_frame_buffer(self, device_id: int, timeout_ms: int) -> Optional[ctypes.POINTER(_NoriFrameBufferOutStruct)]:
        self.ensure_initialized()
        assert self._dll is not None
        frame_ptr = ctypes.POINTER(_NoriFrameBufferOutStruct)()
        ret = int(
            self._dll.Nori_Xvision_GetFrameBuff(
                ctypes.c_uint32(int(device_id)),
                ctypes.byref(frame_ptr),
                ctypes.c_uint32(int(timeout_ms)),
            )
        )
        if ret != NORI_OK or not bool(frame_ptr):
            return None
        return frame_ptr

    def free_frame_buffer(self, device_id: int, frame_ptr: ctypes.POINTER(_NoriFrameBufferOutStruct)) -> None:
        if self._dll is None or not bool(frame_ptr):
            return
        try:
            self._dll.Nori_Xvision_FreeFrameBuff(ctypes.c_uint32(int(device_id)), frame_ptr)
        except Exception:
            pass

    def close_video_stream(self, device_id: int, ignore_errors: bool = False) -> None:
        if self._dll is None:
            return
        if (not ignore_errors) and int(device_id) not in self._active_video_devices:
            return
        err: Optional[Exception] = None
        try:
            ret = int(self._dll.Nori_Xvision_VideoStop(ctypes.c_uint32(int(device_id))))
            if ret != NORI_OK and not ignore_errors:
                raise RuntimeError(f"Nori_Xvision_VideoStop failed: 0x{ret:04X}")
        except Exception as exc:
            err = exc
        try:
            ret = int(self._dll.Nori_Xvision_DeviceVideoUnInit(ctypes.c_uint32(int(device_id))))
            if ret != NORI_OK and not ignore_errors:
                raise RuntimeError(f"Nori_Xvision_DeviceVideoUnInit failed: 0x{ret:04X}")
        except Exception as exc:
            err = err or exc
        self._active_video_devices.discard(int(device_id))
        if err is not None and not ignore_errors:
            raise err

    def get_exposure_control(self, device_id: int) -> Tuple[int, int, int, int, int, int, int]:
        self.ensure_initialized()
        assert self._dll is not None
        val = ctypes.c_long(0)
        flags = ctypes.c_long(0)
        step = ctypes.c_long(0)
        min_v = ctypes.c_long(0)
        max_v = ctypes.c_long(0)
        def_v = ctypes.c_long(0)
        caps = ctypes.c_long(0)
        ret = int(
            self._dll.Nori_Xvision_GetCameraTerminalControl(
                ctypes.c_uint32(int(device_id)),
                ctypes.c_int(CAMERA_CONTROL_EXPOSURE),
                ctypes.byref(val),
                ctypes.byref(flags),
                ctypes.byref(step),
                ctypes.byref(min_v),
                ctypes.byref(max_v),
                ctypes.byref(def_v),
                ctypes.byref(caps),
            )
        )
        if ret != NORI_OK:
            raise RuntimeError(f"Nori_Xvision_GetCameraTerminalControl failed: 0x{ret:04X}")
        return (int(val.value), int(flags.value), int(step.value), int(min_v.value), int(max_v.value), int(def_v.value), int(caps.value))

    def set_auto_exposure(self, device_id: int, enable: bool) -> None:
        self.ensure_initialized()
        assert self._dll is not None
        val, *_ = self.get_exposure_control(device_id)
        flags = CAMERA_CONTROL_FLAGS_AUTO if enable else CAMERA_CONTROL_FLAGS_MANUAL
        ret = int(
            self._dll.Nori_Xvision_SetCameraTerminalControl(
                ctypes.c_uint32(int(device_id)),
                ctypes.c_int(CAMERA_CONTROL_EXPOSURE),
                ctypes.c_long(val),
                ctypes.c_long(flags),
            )
        )
        if ret != NORI_OK:
            raise RuntimeError(f"Nori_Xvision_SetCameraTerminalControl failed: 0x{ret:04X}")

    def set_manual_exposure_level(self, device_id: int, level: int) -> None:
        self.ensure_initialized()
        assert self._dll is not None
        ret = int(
            self._dll.Nori_Xvision_SetCameraTerminalControl(
                ctypes.c_uint32(int(device_id)),
                ctypes.c_int(CAMERA_CONTROL_EXPOSURE),
                ctypes.c_long(max(-10, min(0, int(level)))),
                ctypes.c_long(CAMERA_CONTROL_FLAGS_MANUAL),
            )
        )
        if ret != NORI_OK:
            raise RuntimeError(f"Nori_Xvision_SetCameraTerminalControl failed: 0x{ret:04X}")

    def trigger_single_auto_exposure(self, device_id: int) -> None:
        self.ensure_initialized()
        assert self._dll is not None
        ret = int(self._dll.Nori_Xvision_SetSingleAutoExposure(ctypes.c_uint32(int(device_id))))
        if ret != NORI_OK:
            raise RuntimeError(f"Nori_Xvision_SetSingleAutoExposure failed: 0x{ret:04X}")

    def shutdown(self) -> None:
        for device_id in list(self._active_video_devices):
            self.close_video_stream(device_id, ignore_errors=True)
        if self._dll is not None and self._initialized:
            try:
                self._dll.Nori_Xvision_UnInit()
            except Exception:
                pass
        self._initialized = False
        if self._dll_dir_handle is not None:
            try:
                self._dll_dir_handle.close()
            except Exception:
                pass
            self._dll_dir_handle = None


class NoriSdkCapture:
    def __init__(
        self,
        controller: NoriProtocolController,
        device_id: int,
        video_info: NoriVideoInfo,
        camera_id: str = "nori",
        timeout_ms: int = 1000,
        owns_controller: bool = False,
    ):
        self._controller = controller
        self._device_id = int(device_id)
        self._video_info = video_info
        self._camera_id = camera_id
        self._timeout_ms = int(timeout_ms)
        self._owns_controller = owns_controller
        self._opened = False
        self._controller.open_video_stream(self._device_id, video_info)
        self._opened = True

    def is_opened(self) -> bool:
        return bool(self._opened)

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, object]]:
        if not self._opened:
            return False, None, {"backend": "nori"}
        frame_ptr = self._controller.get_frame_buffer(self._device_id, self._timeout_ms)
        if frame_ptr is None:
            return False, None, {"backend": "nori", "timeout": True}
        try:
            frame = frame_ptr.contents
            width = int(frame.PixFormat.u_Width) or int(self._video_info.width)
            height = int(frame.PixFormat.u_Height) or int(self._video_info.height)
            expected_len = width * height * 3
            if int(frame.u_FrameLen) < expected_len or not bool(frame.pBufAddr):
                return False, None, {"backend": "nori", "short_frame": True}
            raw = ctypes.string_at(frame.pBufAddr, expected_len)
            img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            img = np.ascontiguousarray(np.flipud(img))
            return True, img, {
                "backend": "nori",
                "device_id": self._device_id,
                "sdk_frame_num": int(frame.u_FrameNum),
                "fps": float(frame.PixFormat.f_Fps),
                "media_type": "MJPG",
            }
        finally:
            self._controller.free_frame_buffer(self._device_id, frame_ptr)

    def release(self) -> None:
        if self._opened:
            self._controller.close_video_stream(self._device_id, ignore_errors=True)
            self._opened = False
        if self._owns_controller:
            self._controller.shutdown()

    def info(self) -> CaptureInfo:
        return CaptureInfo(
            backend="nori",
            camera_id=self._camera_id,
            width=int(self._video_info.width),
            height=int(self._video_info.height),
            fps=float(self._video_info.fps),
            metadata={"device_id": self._device_id, "media_type": "MJPG"},
        )

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def open_nori_capture(
    device_index: int,
    width: int,
    height: int,
    fps: int,
    sdk_root: str | Path | None = None,
    nori_device_id: int | None = None,
    camera_id: str = "nori",
) -> Optional[NoriSdkCapture]:
    if sys.platform != "win32":
        return None
    controller = NoriProtocolController(sdk_root=sdk_root)
    device_id = int(nori_device_id) if nori_device_id is not None else int(device_index)
    try:
        selected = controller.select_video_info(device_id=device_id, width=width, height=height, fps=fps)
        if selected is None:
            controller.shutdown()
            return None
        timeout_ms = max(1000, int(1000.0 / max(1.0, selected.fps)) * 4)
        return NoriSdkCapture(
            controller=controller,
            device_id=device_id,
            video_info=selected,
            camera_id=camera_id,
            timeout_ms=timeout_ms,
            owns_controller=True,
        )
    except Exception:
        controller.shutdown()
        return None
