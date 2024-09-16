import window
import time


class Context(object):
    def __init__(self):
        self.boss_blood = 100
        self.next_boss_blood = 100
        self.self_blood = 100
        self.next_self_blood = 100
        self.self_energy = 100
        self.next_self_energy = 100
        self.stop = 0
        self.emergence_break = 0
        self.dodge_weight = 1
        self.attack_weight = 1
        self.init_medicine_nums = 5
        self.reward = 0
        self.done = 0
        self.paused = True
        self.begin_time = int(time.time())
        self.last_reward_time = int(time.time())

    def updateNextContext(self):
        self.next_boss_blood = window.get_boss_blood_window().blood_count()
        self.next_self_blood = window.get_self_blood_window().blood_count()
        self.next_self_energy = window.get_self_energy_window().blood_count()
        return self

    def updateContext(self):
        self.boss_blood = self.next_boss_blood
        self.self_blood = self.next_self_blood
        self.self_energy = self.next_self_energy
        return self
