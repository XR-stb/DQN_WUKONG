# coding=utf-8
from time import sleep 
import keys
from getkeys import key_check
import BloodCommon
import restart

keys = keys.Keys()

init_medicine_nums = BloodCommon.init_medicine_nums
init_self_blood = BloodCommon.init_self_blood

def pause(second=0.04):
    keys.directKey("T")
    sleep(second)
    keys.directKey("T", keys.key_release)

def lock_view(second=0.2):
    keys.directMouse(buttons=keys.mouse_mb_press)
    sleep(second)
    keys.directMouse(buttons=keys.mouse_mb_release)

def run_with_dircet(second=0.5, direct="W"):
    keys.directKey(direct)
    keys.directKey("LSHIFT")
    sleep(second)
    keys.directKey(direct, keys.key_release)
    keys.directKey("LSHIFT", keys.key_release)
     
def eat(second=0.04):
    keys.directKey("R")
    sleep(second)
    keys.directKey("R", keys.key_release)

def recover(self_blood):
    if init_medicine_nums > 0:
        if init_self_blood > 0 and self_blood / init_self_blood < 0.5:
            run_with_dircet("S")
            # 拉开一段距离然后打药
            eat()

def attack(second=0.2):
    keys.directMouse(buttons=keys.mouse_lb_press)
    sleep(second)
    keys.directMouse(buttons=keys.mouse_lb_release)
    
def jump(second=0.04):
    keys.directKey("LCTRL")
    sleep(second)
    keys.directKey("LCTRL", keys.key_release)

 # 闪避
def dodge(second=0.04):
    keys.directKey("SPACE")
    sleep(second)
    keys.directKey("SPACE", keys.key_release)

def go_forward(second=0.04):
    print('向前ing')
    keys.directKey("W")
    sleep(second)
    keys.directKey("W", keys.key_release)
    
def go_back(second=0.04):
    keys.directKey("S")
    sleep(second)
    keys.directKey("S", keys.key_release)
    
def go_left(second=0.04):
    keys.directKey("A")
    sleep(second)
    keys.directKey("A", keys.key_release)
    
def go_right(second=0.04):
    print('向右ing')
    keys.directKey("D")
    sleep(second)
    keys.directKey("D", keys.key_release)

def pause_game(paused, emergence_break=0):
    keys = key_check()
    if 'T' in keys:
        if paused:
            paused = False
            print('start game')
            sleep(1)
        else:
            paused = True
            print('pause game')
            sleep(1)
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
                    sleep(1)
                    break
                else:
                    paused = True
                    sleep(1)
    return paused

# 减少选择,加速收敛
def take_action(action, self_blood):
    if action == 0:
        pass # 不动
    elif action == 1: 
        attack()
    elif action == 2: 
        dodge()
       

if __name__ == '__main__':
    pass
    # TODO: 死了以后让Ai自动走到boss那里