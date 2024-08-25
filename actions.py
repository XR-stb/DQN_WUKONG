# -*- coding: utf-8 -*-
from time import sleep 
import keys

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

if __name__ == '__main__':
    pass
    # TODO: 死了以后让Ai自动走到boss那里