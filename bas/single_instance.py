from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .paths import PROJECT_ROOT

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    _KERNEL32.CreateMutexW.restype = wintypes.HANDLE
    _KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
    _KERNEL32.CloseHandle.restype = wintypes.BOOL
    _ERROR_ALREADY_EXISTS = 183
else:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - only relevant on non-POSIX platforms without fcntl.
        fcntl = None


@dataclass
class SingleInstanceHandle:
    name: str
    _windows_handle: int | None = None
    _lock_file: TextIO | None = None

    def release(self) -> None:
        if self._windows_handle is not None:
            handle = self._windows_handle
            self._windows_handle = None
            _KERNEL32.CloseHandle(handle)
        if self._lock_file is not None:
            lock_file = self._lock_file
            self._lock_file = None
            if fcntl is not None:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            lock_file.close()


def acquire_runtime_single_instance() -> SingleInstanceHandle | None:
    key = _runtime_instance_key(PROJECT_ROOT)
    if os.name == "nt":
        return _acquire_windows_mutex(key)
    return _acquire_posix_file_lock(key)


def _runtime_instance_key(project_root: Path) -> str:
    project_hash = hashlib.sha1(str(project_root).encode("utf-8")).hexdigest()[:12]
    return f"Local\\BAS_RUNTIME_{project_hash}"


def _acquire_windows_mutex(name: str) -> SingleInstanceHandle | None:
    handle = _KERNEL32.CreateMutexW(None, False, name)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"CreateMutexW failed for {name}")
    if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
        _KERNEL32.CloseHandle(handle)
        return None
    return SingleInstanceHandle(name=name, _windows_handle=int(handle))


def _acquire_posix_file_lock(name: str) -> SingleInstanceHandle | None:
    if fcntl is None:  # pragma: no cover - only relevant on unusual non-Windows platforms.
        return SingleInstanceHandle(name=name)
    lock_dir = PROJECT_ROOT / "local_settings"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{name.replace('\\', '_').lower()}.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        return None
    return SingleInstanceHandle(name=name, _lock_file=lock_file)
