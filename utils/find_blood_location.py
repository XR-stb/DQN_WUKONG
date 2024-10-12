# coding=utf-8
from ast import main
import cv2
import time
import window
from log import log
import grabscreen

def main():
    wait_time = 5
    update_content = True  # 新增一个标志变量，控制是否更新内容

    while wait_time > 0:
        log.debug(f'{wait_time} 秒后开始执行')
        wait_time -= 1
        time.sleep(1)

    last_time = time.time()
    
    # 初始化显示窗口
    main_screen_window = None
    self_blood_window = None
    magic_blood_window = None
    boss_blood_window = None

    while True:
        if update_content:

            # 一次性抓取屏幕帧
            frame = grabscreen.grab_screen()

            # 更新所有窗口对象
            window.get_boss_blood_window().update(frame)
            window.get_self_blood_window().update(frame)
            window.get_self_energy_window().update(frame)
            window.get_self_magic_window().update(frame)
            window.get_main_screen_window().update(frame)


            main_screen_window = window.get_main_screen_window()

            self_blood_window = window.get_self_blood_window()
            magic_blood_window = window.get_self_magic_window()

            boss_blood_window = window.get_boss_blood_window()

            self_blood = self_blood_window.blood_count()
            boss_blood = boss_blood_window.blood_count()
            magic_blood = magic_blood_window.blood_count()

            log.debug('self_blood: %d, boss_blood:%d' % (self_blood, boss_blood))

        # 显示当前的画面，无论是否更新内容
        if main_screen_window:
            # cv2.imshow('main_screen', main_screen_window.gray)
            cv2.imshow('self_screen_gray', self_blood_window.gray)
            # cv2.imshow('boss_screen_gray', boss_blood_window.gray)
            # cv2.imshow('magic_blood_gray', magic_blood_window.gray)
            # cv2.imshow('energy_window', window.get_self_energy_window().gray)

        

        # 耗时
        # log.debug('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            update_content = not update_content  # 切换内容更新状态

    cv2.waitKey()  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
