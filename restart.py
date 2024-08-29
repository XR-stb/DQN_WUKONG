# coding=utf-8

import actions
import time

from getkeys import key_check

# 黑风大王处寻路
def heifeng_restart():
    # actions.pause()
    print("已死亡,自动寻路训练开始")

    # 等复活到土地庙先
    wait = 30
    for i in range(wait, 0, -1):
        print('%d秒后复活' % i)
        time.sleep(1)
    
    print('向左ing')
    actions.run_with_dircet(0.7, 'A')
    time.sleep(1) # 必须等一会儿，不然按键会没响应
    print('向前ing')
    actions.run_with_dircet(9, 'W')

    actions.lock_view()
    actions.pause()

def huxianfeng_restart():
    # return
    print("已死亡,自动寻路训练开始")

    # 等复活到土地庙先
    wait = 28
    for i in range(wait, 0, -1):
        keys = key_check()
        if 'B' in keys:
            return
        print('%d秒后复活' % i)
        time.sleep(1)

    actions.go_right(2)
    time.sleep(1) # 必须等一会儿，不然按键会没响应

    actions.go_forward(3)
    time.sleep(1)

    actions.go_left(1)
    time.sleep(1)

    actions.run_with_dircet(4, 'W')
    time.sleep(1)

    actions.go_left(1.7)
    time.sleep(1)

    actions.run_with_dircet(8, 'W')
    actions.lock_view()
    actions.run_with_dircet(1.5, 'W')

    time.sleep(1)

    actions.pause()

def restart():
    
    print("已死亡,自动寻路训练开始")

    # 等复活到土地庙先
    wait = 25
    for i in range(wait, 0, -1):
        keys = key_check()
        if 'B' in keys:
            return
        print('%d秒后复活' % i)
        time.sleep(1)
    
    # 用土地庙锁定初始视角
    actions.press_E()
    time.sleep(3)
    actions.press_ESC()
    time.sleep(3)

    actions.go_back(2.5)
    time.sleep(1) # 必须等一会儿，不然按键会没响应

    actions.go_right(2)
    time.sleep(1)

    times = 4
    while times > 0:
        times -= 1
        actions.run_with_dircet(2, 'W')
        time.sleep(1)

        if times == 0:
            actions.mouse_move(50)
            actions.lock_view()
            time.sleep(1)
            break

        actions.go_right(3)
        time.sleep(1)
    actions.run_with_dircet(2, 'W')
    actions.pause()

if __name__ == "__main__":  
    restart()