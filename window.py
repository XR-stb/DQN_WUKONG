# coding=utf-8
import cv2
import grabscreen
import numpy as np

class Window(object):
    def __init__(self, sx, sy, ex, ey):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        self.mistake = 3 # 图像尺寸误差
        self.color = grabscreen.grab_screen(self)
        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

    def get_tuple(self):
        return (self.sx, self.sy, self.ex, self.ey)
    
    def __iter__(self):
        return iter((self.sx, self.sy, self.ex, self.ey))

    def __repr__(self):
        return f"Window(sx={self.sx}, sy={self.sy}, ex={self.ex}, ey={self.ey})"

class Bloodwindow(Window):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        # 剩余血量的灰度值范围, 即血条白色部分的灰度值
        self.blood_gray_max = 255
        self.blood_gray_min = 100
    def blood_count(self) -> int:
        # 直接获取图像中间一行像素
        middle_row = self.gray[self.gray.shape[0] // 2, :]

        # 使用 np.clip 限制在 blood_gray_min 和 blood_gray_max 之间的值
        clipped = np.clip(middle_row, self.blood_gray_min, self.blood_gray_max)

        # 计算符合血量区间的像素数量
        current_health_length = np.count_nonzero(clipped == middle_row)

        # 计算血量百分比
        total_length = len(middle_row)
        health_percentage = (current_health_length / total_length) * 100

        return health_percentage

    
class Energywindow(Bloodwindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        # 剩余体力的灰度值范围
        self.blood_gray_max = 165
        self.blood_gray_min = 135

class Magicwindow(Bloodwindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        # 剩余体力的灰度值范围
        self.blood_gray_max = 120
        self.blood_gray_min = 70

def get_self_blood_window():
    return Bloodwindow(778, 684, 989, 697)

def get_skill_1_window():
    return Bloodwindow(1749, 601, 1759, 611)

def get_skill_2_window():
    return Bloodwindow(1786, 601, 1797, 611)

# 体力值
def get_self_energy_window():
    return Energywindow(780, 709, 995, 713)

# 蓝条
def get_self_magic_window():
    return Magicwindow(780, 701, 956, 705)

def get_boss_blood_window():
    return Bloodwindow(1151, 639, 1417, 648)

def get_main_screen_window():
    # 主要窗口, 不收集全屏数据, 只关心能看到自己和boss这部分画面, 减少训练量
    #return Window(640, 30, 1920, 750)
    return Window(1050, 100, 1600, 700)


# 这句话很重要 程序启动的时候 初始化好 截屏器    
init_self_blood = get_self_blood_window().blood_count()