# coding=utf-8
import cv2
import grabscreen

class GrayWindow(object):
    def __init__(self, sx, sy, ex, ey):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        self.mistake = 3 # 图像尺寸误差
        self.gray = cv2.cvtColor(grabscreen.grab_screen(self), cv2.COLOR_BGR2GRAY)

    def get_tuple(self):
        return (self.sx, self.sy, self.ex, self.ey)
    
    def __iter__(self):
        return iter((self.sx, self.sy, self.ex, self.ey))

    def __repr__(self):
        return f"BloodWindow(sx={self.sx}, sy={self.sy}, ex={self.ex}, ey={self.ey})"

class BloodWindow(GrayWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        # 剩余血量的灰度值范围, 即血条白色部分的灰度值
        self.blood_gray_max = 220
        self.blood_gray_min = 180
        
        self.blood_gray_min = 80

    def blood_count(self) -> int:
        self_blood = 0
        for self_bd_num in self.gray[(self.gray.shape[0]) // 2]: # 取中间这一行值作为血量计算
            if self.blood_gray_min <= self_bd_num <= self.blood_gray_max:
                self_blood += 1
        return self_blood
    
class EnergyWindow(BloodWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        # 剩余体力的灰度值范围
        self.blood_gray_max = 165
        self.blood_gray_min = 135

def get_self_blood_window():
    # return BloodWindow(210, 980, 572, 995)
    return BloodWindow(210, 980, 360, 995)

# 体力值
def get_self_energy_window():
    return EnergyWindow(210, 1015, 350, 1023)

def get_boss_blood_window():
    return BloodWindow(765, 910, 1160, 924)

def get_main_screen_window():
    # 主要窗口, 不收集全屏数据, 只关心能看到自己和boss这部分画面, 减少训练量
    return GrayWindow(600, 270, 1300, 900)

init_self_blood = get_self_blood_window().blood_count()
init_medicine_nums = 4 