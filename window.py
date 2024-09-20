# window.py
import cv2
import numpy as np


class Window:
    def __init__(self, sx, sy, ex, ey):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        self.color = None  # 初始化为 None

    def extract_region(self, frame):
        if frame is None:
            print("No frame received.")
            return None
        return frame[self.sy:self.ey + 1, self.sx:self.ex + 1]

    def update(self, frame):
        self.color = self.extract_region(frame)

    def __repr__(self):
        return f"Window(sx={self.sx}, sy={self.sy}, ex={self.ex}, ey={self.ey})"

class GrayWindow(Window):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.gray = None  # 初始化为 None

    def update(self, frame):
        super().update(frame)
        if self.color is not None:
            self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = None

class BloodWindow(GrayWindow):
    def __init__(self, sx, sy, ex, ey, blood_gray_min=100, blood_gray_max=255):
        super().__init__(sx, sy, ex, ey)
        self.blood_gray_min = blood_gray_min
        self.blood_gray_max = blood_gray_max
        self.health_percentage = 0  # 初始化血量百分比

    def update(self, frame):
        super().update(frame)
        if self.gray is not None:
            middle_row = self.gray[self.gray.shape[0] // 2, :]
            clipped = np.clip(middle_row, self.blood_gray_min, self.blood_gray_max)
            count = np.count_nonzero(clipped == middle_row)
            total_length = len(middle_row)
            self.health_percentage = (count / total_length) * 100

    def blood_count(self) -> int:
        return self.health_percentage

class EnergyWindow(BloodWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, blood_gray_min=135, blood_gray_max=165)

class MagicWindow(BloodWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, blood_gray_min=70, blood_gray_max=120)

# 预实例化所有窗口对象
self_blood_window = BloodWindow(778, 684, 989, 697)
skill_1_window = BloodWindow(1749, 601, 1759, 611)
skill_2_window = BloodWindow(1786, 601, 1797, 611)
self_energy_window = EnergyWindow(780, 709, 995, 713)
self_magic_window = MagicWindow(780, 701, 956, 705)
boss_blood_window = BloodWindow(1151, 639, 1417, 648)
main_screen_window = Window(1050, 100, 1600, 700)

# 提供访问函数
def get_self_blood_window():
    return self_blood_window

def get_skill_1_window():
    return skill_1_window

def get_skill_2_window():
    return skill_2_window

def get_self_energy_window():
    return self_energy_window

def get_self_magic_window():
    return self_magic_window

def get_boss_blood_window():
    return boss_blood_window

def get_main_screen_window():
    return main_screen_window
