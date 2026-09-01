"""Small fixed-rate scheduling helpers shared by recording and diagnostics."""
from __future__ import annotations

import sys
import time


class WindowsTimerResolution:
    """Request 1ms sleeps on Windows and always pair the system call."""

    def __init__(self, milliseconds: int = 1) -> None:
        self.milliseconds = milliseconds
        self._winmm = None
        self.active = False

    def start(self) -> None:
        if self.active or sys.platform != "win32":
            return
        import ctypes

        winmm = ctypes.WinDLL("winmm")
        if winmm.timeBeginPeriod(self.milliseconds) == 0:
            self._winmm = winmm
            self.active = True

    def close(self) -> None:
        if self.active and self._winmm is not None:
            self._winmm.timeEndPeriod(self.milliseconds)
        self.active = False
        self._winmm = None


def wait_until(deadline: float, *, clock=time.perf_counter, sleep=time.sleep,
               spin_seconds: float = 0.0015) -> None:
    """Sleep coarsely, then spin briefly to avoid a Windows scheduler quantum miss."""
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            return
        if remaining > spin_seconds + 0.001:
            sleep(remaining - spin_seconds)
        else:
            # A sub-2ms spin at 8Hz is bounded to about 1.2% of one core and
            # avoids oversleeping 10-16ms. No catch-up bursts are generated.
            pass
