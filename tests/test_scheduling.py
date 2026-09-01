from wukong_rl.scheduling import wait_until


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
