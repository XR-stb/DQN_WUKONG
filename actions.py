# -*- coding: utf-8 -*-
from time import sleep 
import keys
from getkeys import key_check

keys = keys.Keys()

# 防御暂时也让它闪
def defense():
    keys.directKey("SPACE")
    sleep(0.04)
    keys.directKey("SPACE", keys.key_release)
    
def attack():
    keys.directMouse(buttons=keys.mouse_lb_press)
    sleep(0.2)
    keys.directMouse(buttons=keys.mouse_lb_release)
    
def jump():
    keys.directKey("LCTRL")
    sleep(0.04)
    keys.directKey("LCTRL", keys.key_release)

 # 闪避
def dodge():
    keys.directKey("SPACE")
    sleep(0.04)
    keys.directKey("SPACE", keys.key_release)

def go_forward():
    keys.directKey("W")
    sleep(0.04)
    keys.directKey("W", keys.key_release)
    
def go_back():
    keys.directKey("S")
    sleep(0.04)
    keys.directKey("S", keys.key_release)
    
def go_left():
    keys.directKey("A")
    sleep(0.04)
    keys.directKey("A", keys.key_release)
    
def go_right():
    keys.directKey("D")
    sleep(0.04)
    keys.directKey("D", keys.key_release)

def pause_game(paused):
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

def take_action(action):
    if action == 0:     # n_choose
        pass
    elif action == 1:   # j
        attack()
    elif action == 2:   # k
        # actions.jump()
        # actions.dodge()
       attack()
    elif action == 3:   # m
        # actions.defense()
        # actions.dodge()
        pass
    elif action == 4:   # r
       dodge()

if __name__ == '__main__':
    pass
    # TODO: 死了以后让Ai自动走到boss那里