# coding=utf-8
import cv2
import time
import numpy as np
import grabscreen
import window

def main():
    time.sleep(5)

    # 抓取屏幕灰度图
    np.savetxt('gray_image.txt', window.get_self_magic_window().gray, fmt='%d')

    # log(gray_image.shape)

if __name__ == "__main__":
    main()