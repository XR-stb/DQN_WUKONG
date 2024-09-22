# window.py
import cv2
import numpy as np

# 创建一个基类，封装静态 offset 和 frame
class BaseWindow:
    offset_x = 0  # 静态变量，所有实例共享
    offset_y = 0  # 静态变量，所有实例共享
    frame = None  # 静态变量，共享的frame

    # 用于存储所有窗口实例的列表
    all_windows = []

    def __init__(self, sx, sy, ex, ey):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        self.color = None  # 每个实例独有的窗口区域
        # 将每个实例添加到 all_windows 列表中
        BaseWindow.all_windows.append(self)

    @staticmethod
    def set_offset(offset_x, offset_y):
        # 设置所有窗口共享的偏移
        BaseWindow.offset_x = offset_x
        BaseWindow.offset_y = offset_y

    @staticmethod
    def set_frame(frame):
        # 设置共享的 frame
        BaseWindow.frame = frame

    def extract_region(self):
        if BaseWindow.frame is None:
            print("No frame received.")
            return None
        # 使用共享的偏移量来计算实际区域
        return BaseWindow.frame[self.sy + BaseWindow.offset_y:self.ey + 1 + BaseWindow.offset_y, 
                                self.sx + BaseWindow.offset_x:self.ex + 1 + BaseWindow.offset_x]

    def update(self):
        self.color = self.extract_region()

    @staticmethod
    def update_all():
        # 更新所有窗口的状态
        for window in BaseWindow.all_windows:
            window.update()

    def __repr__(self):
        return f"BaseWindow(sx={self.sx}, sy={self.sy}, ex={self.ex}, ey={self.ey}, offset_x={BaseWindow.offset_x}, offset_y={BaseWindow.offset_y})"

# 继承 BaseWindow 的子类
class GrayWindow(BaseWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.gray = None

    def update(self):
        super().update()
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

    def update(self):
        super().update()
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



# 查找logo位置的函数
def find_game_window_logo(frame, template_path, threshold):
    # 读取模板图像
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Failed to load template image from {template_path}")
        return None

    # 将frame转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 模板匹配
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

    # 查找匹配区域
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 如果匹配结果足够好，则返回logo的左上角位置
    if max_val >= threshold:
        return max_loc
    else:
        return None

# 设置窗口的相对坐标偏移
def set_windows_offset(frame):
    # 查找logo的初始位置
    logo_position = find_game_window_logo(frame, "./images/title_logo.png", 0.8)

    if logo_position is not None:
        offset_x, offset_y = logo_position

        #根据logo图片再title bar的位置修正
        offset_x += -3
        offset_y += 27

        # 设置偏移量给所有窗口对象
        BaseWindow.set_offset(offset_x, offset_y)
        BaseWindow.set_frame(frame)

        print(f"All windows offset by ({offset_x}, {offset_y})")
        return True
    else:
        print("Failed to find the game logo, offsets not set.")
        return False


# 预实例化所有窗口对象
game_window = BaseWindow(0, 0, 1280, 720)
self_blood_window = BloodWindow(0, 0, 100, 100)
# skill_1_window = BloodWindow(1749, 601, 1759, 611)
# skill_2_window = BloodWindow(1786, 601, 1797, 611)
# self_energy_window = EnergyWindow(780, 709, 995, 713)
# self_magic_window = MagicWindow(780, 701, 956, 705)
# boss_blood_window = BloodWindow(1151, 639, 1417, 648)
# battle_roi_window = BaseWindow(1050, 100, 1600, 700)