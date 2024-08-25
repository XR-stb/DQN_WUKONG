# coding=utf-8
from ast import main
import cv2
import time
import BloodCommon

def main():
    wait_time = 5
    while wait_time > 0:
        print(f'{wait_time} 秒后开始执行')
        wait_time -= 1
        time.sleep(1)

    last_time = time.time()
    while(True):
        self_blood_window = BloodCommon.get_self_blood_window()
        boss_blood_window = BloodCommon.get_boss_blood_window()
        main_screen_window = BloodCommon.get_main_screen_window()
        self_blood = self_blood_window.blood_count()
        boss_blood = boss_blood_window.blood_count()
        # print('self_blood: %d, boss_blood:%d' % (self_blood, boss_blood))

        # cv2.imshow('self_screen_gray', self_blood_window.gray)
        # cv2.imshow('boss_screen_gray', boss_blood_window.gray)
        # cv2.imshow('main_screen', main_screen_window.gray)
        cv2.imshow('energy_window', BloodCommon.get_self_energy_window().gray)
        
        # 耗时
        # print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.waitKey() # 视频结束后，按任意键退出
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()