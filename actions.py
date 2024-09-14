# coding=utf-8
import keys
from getkeys import key_check
import window
import restart

import time
from context import Context
from log import log

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
    log("打药")
    eat()
    run_with_dircet(2, "S")


def attack(second=0.2):
    keys.directMouse(buttons=keys.mouse_lb_press)
    precise_sleep(second)
    keys.directMouse(buttons=keys.mouse_lb_release)
    log("攻击")


def jump(second=0.04):
    keys.directKey("LCTRL")
    precise_sleep(second)
    keys.directKey("LCTRL", keys.key_release)


# 闪避
def dodge(second=0.04):
    keys.directKey("SPACE")
    precise_sleep(second)
    keys.directKey("SPACE", keys.key_release)
    log("闪避")


def go_forward(second=0.04):
    keys.directKey("W")
    precise_sleep(second)
    keys.directKey("W", keys.key_release)
    log("向前")


def go_back(second=0.04):
    keys.directKey("S")
    precise_sleep(second)
    keys.directKey("S", keys.key_release)


def go_left(second=0.04):
    keys.directKey("A")
    precise_sleep(second)
    keys.directKey("A", keys.key_release)


def go_right(second=0.04):
    log("向右ing")
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

def press_Enter(second=0.04):
    keys.directKey("ENTER")
    precise_sleep(second)
    keys.directKey("ENTER", keys.key_release)
    log("按下 Enter")

def press_up(second=0.04):
    keys.directKey("UP")
    precise_sleep(second)
    keys.directKey("UP", keys.key_release)
    log("按下 上箭头")

def press_down(second=0.04):
    keys.directKey("DOWN")
    precise_sleep(second)
    keys.directKey("DOWN", keys.key_release)
    log("按下 下箭头")

def press_left(second=0.04):
    keys.directKey("LEFT")
    precise_sleep(second)
    keys.directKey("LEFT", keys.key_release)
    log("按下 左箭头")

def press_right(second=0.04):
    keys.directKey("RIGHT")
    precise_sleep(second)
    keys.directKey("RIGHT", keys.key_release)
    log("按下 右箭头")

def use_skill(skill_key="1", second=0.04):
    keys.directKey(skill_key)
    precise_sleep(second)
    keys.directKey(skill_key, keys.key_release)
    log("使用 技能1")


def pause_game(context: Context, emergence_break=0):
    keys = key_check()
    if "T" in keys:
        if context.paused:
            context.paused = False
            log("start game")
            precise_sleep(1)
        else:
            context.paused = True
            log("pause game, press T to start")
            precise_sleep(1)
    if context.paused:
        log("paused")
        if emergence_break == 100:
            restart.restart()

        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if "T" in keys:
                if context.paused:
                    context.paused = False
                    context.begin_time = int(time.time())
                    log("start game")
                    precise_sleep(1)
                    break
                else:
                    context.paused = True
                    precise_sleep(1)
    return context


def is_skill_1_in_cd():
    return window.get_skill_1_window().blood_count() < 50


def take_action(action, context: Context) -> Context:
    # 更新当前状态
    context.magic_num = window.get_self_magic_window().blood_count()
    context.self_energy = window.get_self_energy_window().blood_count()
    context.self_blood = window.get_self_blood_window().blood_count()

    if action == 0:  # 无操作，等待
        run_with_dircet(0.5, 'W')
        context.dodge_weight = 1

    elif action == 1:  # 攻击
        attack()
        context.dodge_weight = 1

    elif action == 2:  # 闪避
        if context.magic_num > 10:
            dodge()
            context.dodge_weight += 10  # 防止收敛于一直闪避的状态

    elif action == 3:  # 使用技能1
        log(
            "magic:%d, energy:%d, self_blood:%d" % (context.magic_num, context.self_energy, context.self_blood)
        )
        log("skill_1 cd :%d" % is_skill_1_in_cd())
        if context.magic_num > 20 and not is_skill_1_in_cd():
            use_skill("1")
            attack()
            context.dodge_weight = 1

    elif action == 4:  # 恢复
        if context.self_blood < 60 and context.init_medicine_nums > 0:
            # 回体力和打药
            recover()
            context.init_medicine_nums -= 1
            context.dodge_weight = 1

    return context

