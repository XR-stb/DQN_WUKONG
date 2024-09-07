# coding=utf-8
import keys
from getkeys import key_check
import BloodCommon
import restart

import time


def precise_sleep(target_duration):
    start_time = time.perf_counter()
    while True:
        current_time = time.perf_counter()
        elapsed_time = current_time - start_time
        if elapsed_time >= target_duration:
            break
        time.sleep(target_duration - elapsed_time)  # sleep for the remaining time


keys = keys.Keys()

def pause(second=0.04):
    keys.directKey("T")
    precise_sleep(second)
    keys.directKey("T", keys.key_release)


def mouse_move(times):
    for i in range(times):
        keys.directMouse(i, 0)
        precise_sleep(0.004)


def mouse_move_v2(times):
    for i in range(times):
        keys.directMouse(i, 0)
        precise_sleep(0.01)


def lock_view(second=0.2):
    keys.directMouse(buttons=keys.mouse_mb_press)
    precise_sleep(second)
    keys.directMouse(buttons=keys.mouse_mb_release)


def run_with_dircet(second=0.5, direct="W"):
    keys.directKey(direct)
    keys.directKey("LSHIFT")
    precise_sleep(second)
    keys.directKey(direct, keys.key_release)
    keys.directKey("LSHIFT", keys.key_release)


def eat(second=0.04):
    keys.directKey("R")
    precise_sleep(second)
    keys.directKey("R", keys.key_release)


def recover():
    if BloodCommon.init_medicine_nums > 0:
        print('打药')
        eat()
        run_with_dircet(2, "S")
        BloodCommon.init_medicine_nums -= 1


def attack(second=0.2):
    keys.directMouse(buttons=keys.mouse_lb_press)
    precise_sleep(second)
    keys.directMouse(buttons=keys.mouse_lb_release)
    print("攻击")


def jump(second=0.04):
    keys.directKey("LCTRL")
    precise_sleep(second)
    keys.directKey("LCTRL", keys.key_release)


# 闪避
def dodge(second=0.04):
    keys.directKey("SPACE")
    precise_sleep(second)
    keys.directKey("SPACE", keys.key_release)
    print("闪避")


def go_forward(second=0.04):
    keys.directKey("W")
    precise_sleep(second)
    keys.directKey("W", keys.key_release)
    print("向前")


def go_back(second=0.04):
    keys.directKey("S")
    precise_sleep(second)
    keys.directKey("S", keys.key_release)


def go_left(second=0.04):
    keys.directKey("A")
    precise_sleep(second)
    keys.directKey("A", keys.key_release)


def go_right(second=0.04):
    print("向右ing")
    keys.directKey("D")
    precise_sleep(second)
    keys.directKey("D", keys.key_release)


def press_E(second=0.04):
    keys.directKey("E")
    precise_sleep(second)
    keys.directKey("E", keys.key_release)


def press_ESC(second=0.04):
    keys.directKey("ESC")
    precise_sleep(second)
    keys.directKey("ESC", keys.key_release)


def use_skill(skill_key="1", second=0.04):
    keys.directKey(skill_key)
    precise_sleep(second)
    keys.directKey(skill_key, keys.key_release)
    print('使用 技能1')


def pause_game(paused, emergence_break=0):
    keys = key_check()
    if "T" in keys:
        if paused:
            paused = False
            print("start game")
            precise_sleep(1)
        else:
            paused = True
            print("pause game")
            precise_sleep(1)
    if paused:
        print("paused")
        if emergence_break == 100:
            restart.restart()

        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if "T" in keys:
                if paused:
                    paused = False
                    print("start game")
                    precise_sleep(1)
                    break
                else:
                    paused = True
                    precise_sleep(1)
    return paused

def is_skill_1_in_cd():
   return BloodCommon.get_skill_1_window().blood_count() < 50

# 减少选择,加速收敛
def take_action(action, self_blood, dodge_weight):
    if action == 0:
        print("不动")

    elif action == 1:
        attack()

    elif action == 2:
        if BloodCommon.get_self_magic_window().blood_count() > 10:
            dodge()
            dodge_weight += 5 # 防止收敛于一直闪避的状态

    elif action == 3:
        magic_num = BloodCommon.get_self_magic_window().blood_count()
        energy_num = BloodCommon.get_self_energy_window().blood_count()
        self_blood = BloodCommon.get_self_blood_window().blood_count()
        print("magic:%d, energy:%d, self_blood:%d" % (magic_num, energy_num, self_blood))
        print("skill_1 cd :%d" % is_skill_1_in_cd())
        if magic_num > 20 and not is_skill_1_in_cd():
            use_skill("1")
       
        elif energy_num < 50 and self_blood < 50:
            # 回体力和打药
            recover()

    # elif action == 4:
    #     # 往后拉, 回体力
    #     go_back(2)

    return dodge_weight
