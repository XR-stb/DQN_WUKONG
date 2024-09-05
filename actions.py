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

init_medicine_nums = BloodCommon.init_medicine_nums
init_self_blood = BloodCommon.init_self_blood

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

def recover(self_blood):
    if init_medicine_nums > 0:
        if init_self_blood > 0 and self_blood / init_self_blood < 0.5:
            run_with_dircet("S")
            # 拉开一段距离然后打药
            eat()

def attack(second=0.2):
    keys.directMouse(buttons=keys.mouse_lb_press)
    precise_sleep(second)
    keys.directMouse(buttons=keys.mouse_lb_release)
    print('攻击')
    
def jump(second=0.04):
    keys.directKey("LCTRL")
    precise_sleep(second)
    keys.directKey("LCTRL", keys.key_release)

 # 闪避
def dodge(second=0.04):
    keys.directKey("SPACE")
    precise_sleep(second)
    keys.directKey("SPACE", keys.key_release)
    print('闪避')

def go_forward(second=0.04):
    keys.directKey("W")
    precise_sleep(second)
    keys.directKey("W", keys.key_release)
    print('向前')
    
def go_back(second=0.04):
    keys.directKey("S")
    precise_sleep(second)
    keys.directKey("S", keys.key_release)
    
def go_left(second=0.04):
    keys.directKey("A")
    precise_sleep(second)
    keys.directKey("A", keys.key_release)
    
def go_right(second=0.04):
    print('向右ing')
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

def pause_game(paused, emergence_break=0):
    keys = key_check()
    if 'T' in keys:
        if paused:
            paused = False
            print('start game')
            precise_sleep(1)
        else:
            paused = True
            print('pause game')
            precise_sleep(1)
    if paused:
        print('paused')
        if emergence_break == 100:
            restart.restart()

        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if 'T' in keys:
                if paused:
                    paused = False
                    print('start game')
                    precise_sleep(1)
                    break
                else:
                    paused = True
                    precise_sleep(1)
    return paused

# 减少选择,加速收敛
def take_action(action, self_blood):
    if action == 0:
        pass # 不动
    elif action == 1: 
        attack()
    elif action == 2: 
        dodge()
    elif action == 3: 
        go_forward(0.5)
        attack()
       

if __name__ == '__main__':
    pass
    # TODO: 死了以后让Ai自动走到boss那里