from wukong_rl.scheduling import wait_until, WindowsTimerResolution


def test_wait_until_uses_coarse_sleep_then_bounded_spin():
    class Clock:
        now = 0.0
        sleeps = []
        def read(self):
            self.now += 0.0001
            return self.now
        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds
    clock = Clock()
    wait_until(0.125, clock=clock.read, sleep=clock.sleep, spin_seconds=0.0015)
    assert clock.now >= 0.125
    assert len(clock.sleeps) == 1
    assert 0.12 < clock.sleeps[0] < 0.125


def test_windows_timer_resolution_is_paired_and_idempotent(monkeypatch):
    import ctypes
    import wukong_rl.scheduling as scheduling

    calls = []
    class WinMM:
        def timeBeginPeriod(self, value): calls.append(("begin", value)); return 0
        def timeEndPeriod(self, value): calls.append(("end", value)); return 0
    monkeypatch.setattr(scheduling.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda _name: WinMM(), raising=False)
    timer = WindowsTimerResolution()
    timer.start()
    timer.start()
    timer.close()
    timer.close()
    assert calls == [("begin", 1), ("end", 1)]
