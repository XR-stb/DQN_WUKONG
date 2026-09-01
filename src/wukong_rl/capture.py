from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
import queue
import time
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

    def __init__(self, config: CaptureConfig, diagnostics: bool = False) -> None:
        self.config = config
        self._started = False
        self.diagnostics = diagnostics
        self.last_frame_metadata: dict = {}
        self._previous_sequence = None

    def start(self) -> None:
        import grabscreen

        self.last_frame_metadata = {}
        self._previous_sequence = None
        grabscreen._GAME_WINDOW_TITLE = self.config.window_title
        if self.config.backend == "wgc" and hasattr(grabscreen, "WindowsCapture"):
            grabscreen._use_wgc = True
        else:
            grabscreen._use_wgc = False
        if self.config.backend == "wgc" and not grabscreen._use_wgc:
            if not self.config.allow_dxcam_fallback:
                raise RuntimeError("WGC unavailable and dxcam fallback is disabled")
        grabscreen.init_camera(target_fps=30, diagnostics=self.diagnostics)
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

        if self.diagnostics:
            frame, metadata = grabscreen.grab_screen(with_metadata=True)
            sequence = metadata.get("callback_sequence")
            metadata["repeated_frame"] = sequence == self._previous_sequence if sequence is not None else None
            self._previous_sequence = sequence
            arrived = metadata.get("callback_timestamp")
            metadata["frame_age_ms"] = (time.perf_counter() - arrived) * 1000 if arrived is not None else None
            self.last_frame_metadata = metadata
        else:
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


def _capture_process_main(config: CaptureConfig, shared, metadata_lock, active_slot, reader_slot,
                          sequence, published, interval, started_at, stop_event, ready_event,
                          errors) -> None:
    """Own WGC and publish only the freshest client frame to bounded shared memory."""
    source = LegacyScreenSource(config, diagnostics=False)
    previous = None
    started_at.value = time.perf_counter()
    try:
        source.start()
        period = 1.0 / config.worker_fps
        deadline = time.perf_counter()
        slots = np.frombuffer(shared, dtype=np.uint8).reshape(3, config.height, config.width, 3)
        while not stop_event.is_set():
            frame = source.read()
            now = time.perf_counter()
            with metadata_lock:
                blocked = {int(active_slot.value), int(reader_slot.value)}
            slot = next(index for index in range(3) if index not in blocked)
            np.copyto(slots[slot], frame)
            with metadata_lock:
                active_slot.value = slot
                sequence.value += 1
                published.value = now
                interval.value = 0.0 if previous is None else (now - previous) * 1000
            previous = now
            ready_event.set()
            deadline += period
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                stop_event.wait(remaining)
            else:
                deadline = time.perf_counter()
    except BaseException as error:
        try:
            errors.put_nowait((type(error).__name__, str(error)))
        except queue.Full:
            pass
    finally:
        ready_event.set()
        source.close()


class ProcessScreenSource:
    """WGC process isolation with a single latest-frame shared-memory slot."""

    def __init__(self, config: CaptureConfig, diagnostics: bool = False) -> None:
        self.config = config
        self.diagnostics = diagnostics
        self.last_frame_metadata: dict = {}
        self._previous_sequence = None
        self._process = None
        self._context = None

    def start(self) -> None:
        if self._process is not None:
            return
        context = mp.get_context("spawn")
        size = 3 * self.config.height * self.config.width * 3
        self._shared = context.RawArray("B", size)
        self._lock = context.Lock()
        self._active_slot = context.Value("i", 0, lock=False)
        self._reader_slot = context.Value("i", -1, lock=False)
        self._sequence = context.Value("Q", 0, lock=False)
        self._published = context.Value("d", 0.0, lock=False)
        self._interval = context.Value("d", 0.0, lock=False)
        self._started_at = context.Value("d", 0.0, lock=False)
        self._stop = context.Event()
        self._ready = context.Event()
        self._errors = context.Queue(maxsize=1)
        self._process = context.Process(
            target=_capture_process_main,
            args=(self.config, self._shared, self._lock, self._active_slot, self._reader_slot,
                  self._sequence, self._published, self._interval, self._started_at,
                  self._stop, self._ready, self._errors),
            name="wukong-capture", daemon=True,
        )
        self._process.start()
        if not self._ready.wait(timeout=12) or self._sequence.value == 0:
            try:
                kind, message = self._errors.get_nowait()
                detail = f"{kind}: {message}"
            except queue.Empty:
                detail = "first frame timed out"
            self.close()
            raise RuntimeError(f"capture process failed: {detail}")
        self.read()

    def read(self) -> np.ndarray:
        if self._process is None or not self._process.is_alive():
            detail = "capture process is not running"
            try:
                kind, message = self._errors.get_nowait()
                detail = f"{kind}: {message}"
            except queue.Empty:
                pass
            raise RuntimeError(detail)
        started = time.perf_counter()
        with self._lock:
            locked = time.perf_counter()
            slot = int(self._active_slot.value)
            self._reader_slot.value = slot
            sequence = int(self._sequence.value)
            published = float(self._published.value)
            interval = float(self._interval.value)
            capture_started = float(self._started_at.value)
        try:
            frame = np.frombuffer(self._shared, dtype=np.uint8).reshape(
                3, self.config.height, self.config.width, 3
            )[slot].copy()
        finally:
            with self._lock:
                self._reader_slot.value = -1
        now = time.perf_counter()
        if self.diagnostics:
            self.last_frame_metadata = {
                "process_sequence": sequence,
                "process_publish_timestamp": published,
                "process_interval_ms": interval or None,
                "process_hz": sequence / max(published - capture_started, 1.0e-9),
                "capture_lock_wait_ms": (locked - started) * 1000,
                "capture_copy_ms": (now - locked) * 1000,
                "repeated_frame": sequence == self._previous_sequence,
                "frame_age_ms": (now - published) * 1000,
            }
        self._previous_sequence = sequence
        return frame

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._stop.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            print("[capture] isolated WGC process was terminated after shutdown timeout", flush=True)
        self._errors.cancel_join_thread()
        self._errors.close()
        self._process = None


def create_screen_source(config: CaptureConfig, diagnostics: bool = False):
    if config.process_isolation and config.backend == "wgc":
        return ProcessScreenSource(config, diagnostics=diagnostics)
    return LegacyScreenSource(config, diagnostics=diagnostics)
