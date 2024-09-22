# context.py
import window
import grabscreen
import time
from timing_decorator import timeit


class Context(object):
    def __init__(self):
        self.boss_blood = 100
        self.next_boss_blood = 100
        self.self_blood = 100
        self.next_self_blood = 100
        self.self_energy = 100
        self.next_self_energy = 100
        self.magic_num = 100
        self.next_magic_num = 100
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

    #@timeit
    def updateContext(self):
        self.boss_blood = self.next_boss_blood
        self.self_blood = self.next_self_blood
        self.self_energy = self.next_self_energy
        self.magic_num = self.next_magic_num


        # 一次性抓取屏幕帧
        frame = grabscreen.grab_screen()

        # 更新所有窗口对象
        window.get_boss_blood_window().update(frame)
        window.get_self_blood_window().update(frame)
        window.get_self_energy_window().update(frame)
        window.get_self_magic_window().update(frame)
        window.get_main_screen_window().update(frame)
        window.get_skill_1_window().update(frame)
        #window.get_skill_2_window().update(frame)

        # 更新下一帧的值
        self.next_boss_blood = window.get_boss_blood_window().blood_count()
        self.next_self_blood = window.get_self_blood_window().blood_count()
        self.next_self_energy = window.get_self_energy_window().blood_count()
        self.next_magic_num = window.get_self_magic_window().blood_count()


        return self

    def get_features(self):
        # 归一化到 0~1
        return [
            float(self.boss_blood) / 100.0,
            float(self.self_blood) / 100.0,
            float(self.self_energy) / 100.0,
            float(self.magic_num) / 100.0,
        ]