from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .config import CaptureConfig


class FrameSource(Protocol):
    def start(self) -> None: ...
    def read(self) -> np.ndarray: ...
    def close(self) -> None: ...


@dataclass(slots=True)
class ArrayFrameSource:
    frames: list[np.ndarray]
    repeat_last: bool = True
    _index: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self._index = 0

    def start(self) -> None:
        self._index = 0

    def read(self) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("array frame source is empty")
        if self._index >= len(self.frames):
            if not self.repeat_last:
                raise EOFError("array frame source exhausted")
            frame = self.frames[-1]
        else:
            frame = self.frames[self._index]
            self._index += 1
        return np.asarray(frame).copy()

    def close(self) -> None:
        return None


class LegacyScreenSource:
    """Strict adapter around the existing WGC/dxcam implementation."""

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self._started = False

    def start(self) -> None:
        import grabscreen

        grabscreen._GAME_WINDOW_TITLE = self.config.window_title
        if self.config.backend == "wgc" and hasattr(grabscreen, "WindowsCapture"):
            grabscreen._use_wgc = True
        else:
            grabscreen._use_wgc = False
        if self.config.backend == "wgc" and not grabscreen._use_wgc:
            if not self.config.allow_dxcam_fallback:
                raise RuntimeError("WGC unavailable and dxcam fallback is disabled")
        grabscreen.init_camera(target_fps=30)
        if self.config.backend == "wgc" and not grabscreen._use_wgc:
            if not self.config.allow_dxcam_fallback:
                grabscreen.stop()
                raise RuntimeError("WGC unavailable and dxcam fallback is disabled")
        self._started = True
        self.read()

    def read(self) -> np.ndarray:
        if not self._started:
            raise RuntimeError("frame source has not been started")
        import grabscreen

        frame = grabscreen.grab_screen()
        if frame is None:
            raise RuntimeError("capture backend returned no frame")
        if not grabscreen._use_wgc:
            frame = self._crop_dxcam_to_client(frame)
        if frame.shape[:2] != (self.config.height, self.config.width):
            raise RuntimeError(
                f"capture client mismatch: expected {self.config.width}x{self.config.height}, "
                f"received {frame.shape[1]}x{frame.shape[0]}; calibration is required"
            )
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _crop_dxcam_to_client(self, frame: np.ndarray) -> np.ndarray:
        """Crop desktop duplication to the exact game client rectangle."""
        import win32gui

        hwnd = win32gui.FindWindow(None, self.config.window_title)
        if not hwnd:
            hwnd = win32gui.FindWindow("UnrealWindow", None)
        if not hwnd:
            raise RuntimeError("cannot locate the game window for dxcam client cropping")
        left, top = win32gui.ClientToScreen(hwnd, (0, 0))
        client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(hwnd)
        width = client_right - client_left
        height = client_bottom - client_top
        right = left + width
        bottom = top + height
        if left < 0 or top < 0 or right > frame.shape[1] or bottom > frame.shape[0]:
            raise RuntimeError("game client is outside the dxcam output; use the primary display")
        return frame[top:bottom, left:right]

    def close(self) -> None:
        if self._started:
            import grabscreen

            grabscreen.stop()
            self._started = False
