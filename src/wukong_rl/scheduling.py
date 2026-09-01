"""Small fixed-rate scheduling helpers shared by recording and diagnostics."""
from __future__ import annotations

import time


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
