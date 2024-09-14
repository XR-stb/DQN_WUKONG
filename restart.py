# coding=utf-8

import actions
from log import log
from getkeys import key_check

# 黑风大王处寻路
def heifeng_restart():
    # actions.pause()
    log("已死亡,自动寻路训练开始")

    # 等复活到土地庙先
    wait = 30
    for i in range(wait, 0, -1):
        log('%d秒后复活' % i)
        actions.precise_sleep(1)
    
    log('向左ing')
    actions.run_with_direct(0.7, 'A')
    actions.precise_sleep(1) # 必须等一会儿，不然按键会没响应
    log('向前ing')
    actions.run_with_direct(9, 'W')

    actions.lock_view()
    actions.pause()

def huxianfeng_restart_v1():
    log("已死亡,自动寻路训练开始")

    # 等复活到土地庙先
    wait = 25
    for i in range(wait, 0, -1):
        keys = key_check()
        if 'B' in keys:
            return
        log('%d秒后复活' % i)
        actions.precise_sleep(1)
    
    # 用土地庙锁定初始视角
    actions.press_E()
    actions.precise_sleep(3)
    actions.press_ESC()
    actions.precise_sleep(3)

    actions.go_back(2.5)
    actions.precise_sleep(1) # 必须等一会儿，不然按键会没响应

    actions.go_right(2)
    actions.precise_sleep(1)

    times = 4
    while times > 0:
        times -= 1
        actions.run_with_direct(2, 'W')
        actions.precise_sleep(1)

        if times == 0:
            actions.mouse_move(35)
            actions.lock_view()
            actions.precise_sleep(1)
            break

        actions.go_right(3)
        actions.precise_sleep(1)
    actions.run_with_direct(2, 'W')
    actions.pause()

def huxianfeng_restart_v2():
    log("已死亡,自动寻路训练开始")

    # 等复活到土地庙先
    wait = 25
    for i in range(wait, 0, -1):
        keys = key_check()
        if 'B' in keys:
            return
        log('%d秒后复活' % i)
        actions.precise_sleep(1)
    
    # 用土地庙锁定初始视角
    actions.press_E()
    actions.precise_sleep(4)
    actions.press_ESC()
    actions.precise_sleep(3)

    actions.go_back(2.5)
    actions.precise_sleep(1) # 必须等一会儿，不然按键会没响应

    actions.go_right(1.6)
    actions.precise_sleep(1)

    actions.mouse_move(35)
    actions.precise_sleep(1)

    actions.run_with_direct(11, 'W')
    actions.precise_sleep(1)

    actions.lock_view()
    actions.pause()
    actions.run_with_direct(1, 'W')



def yinhu_restart():
    log("已死亡,自动寻路训练开始")

    # 等复活到土地庙先
    wait = 18
    for i in range(wait, 0, -1):
        keys = key_check()
        if 'B' in keys:
            return
        log('%d秒后复活' % i)
        actions.precise_sleep(1)

    actions.go_forward(1.5)
    actions.precise_sleep(3) # 必须等一会儿，不然按键会没响应

    actions.press_E()
    actions.precise_sleep(5)

    # 菜单会停留在上次选的那个项
    # actions.press_down()
    # actions.precise_sleep(1)
    # actions.press_down()
    # actions.precise_sleep(1)

    actions.dodge()
    actions.precise_sleep(5)

    log('长按e 跳过动画！')
    actions.press_E(3)
    actions.precise_sleep(10)
    
    log('ready!')
    #开了自动锁定了
    #actions.lock_view()
    actions.run_with_direct(2, 'W')
    actions.pause()


def restart():
    yinhu_restart()

if __name__ == "__main__":  
    restart()