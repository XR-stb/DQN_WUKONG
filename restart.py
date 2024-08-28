# coding=utf-8

import actions
import time

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

def restart():
    return
    # actions.pause()
    print("已死亡,自动寻路训练开始")

    # 等复活到土地庙先
    wait = 30
    for i in range(wait, 0, -1):
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

    actions.go_left(1.5)
    time.sleep(1)

    actions.run_with_dircet(6, 'W')
    actions.lock_view()
    actions.run_with_dircet(3, 'W')

    time.sleep(1)

    actions.pause()

if __name__ == "__main__":  
    restart()