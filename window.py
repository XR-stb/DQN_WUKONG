# window.py
import cv2
import numpy as np

# 基类，封装静态 offset 和 frame
class BaseWindow:
    offset_x = 0
    offset_y = 0
    frame = None
    all_windows = []

    def __init__(self, sx, sy, ex, ey):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        self.color = None
        BaseWindow.all_windows.append(self)

    @staticmethod
    def set_offset(offset_x, offset_y):
        BaseWindow.offset_x = offset_x
        BaseWindow.offset_y = offset_y

    @staticmethod
    def set_frame(frame):
        BaseWindow.frame = frame

    def extract_region(self):
        if BaseWindow.frame is None:
            print("No frame received.")
            return None
        return BaseWindow.frame[self.sy + BaseWindow.offset_y:self.ey + 1 + BaseWindow.offset_y, 
                                self.sx + BaseWindow.offset_x:self.ex + 1 + BaseWindow.offset_x]

    def update(self):
        self.color = self.extract_region()

    @staticmethod
    def update_all():
        for window in BaseWindow.all_windows:
            window.update()

    def __repr__(self):
        return f"BaseWindow(sx={self.sx}, sy={self.sy}, ex={self.ex}, ey={self.ey}, offset_x={BaseWindow.offset_x}, offset_y={BaseWindow.offset_y})"

# 新的基类，统一返回状态（0/1 或 百分比）
class StatusWindow(BaseWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.status = 0  # 初始化为0

    def update(self):
        super().update()
        if self.color is not None:
            self.process_color()
        else:
            self.status = 0

    def process_color(self):
        # 子类需要实现具体的处理逻辑
        pass

    def get_status(self):
        return self.status

# 灰度窗口的基类
class GrayWindow(StatusWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.gray = None

    def process_color(self):
        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY) if self.color is not None else None

# 血量窗口
class BloodWindow(GrayWindow):
    def __init__(self, sx, sy, ex, ey, blood_gray_min=100, blood_gray_max=255):
        super().__init__(sx, sy, ex, ey)
        self.blood_gray_min = blood_gray_min
        self.blood_gray_max = blood_gray_max

    def process_color(self):
        super().process_color()
        if self.gray is not None:
            middle_row = self.gray[self.gray.shape[0] // 2, :]
            clipped = np.clip(middle_row, self.blood_gray_min, self.blood_gray_max)
            count = np.count_nonzero(clipped == middle_row)
            total_length = len(middle_row)
            self.status = (count / total_length) * 100

# 技能窗口
class SkillWindow(GrayWindow):
    def __init__(self, sx, sy, ex, ey, skill_gray_min=100, skill_gray_max=150):
        super().__init__(sx, sy, ex, ey)
        self.skill_gray_min = skill_gray_min
        self.skill_gray_max = skill_gray_max

    def process_color(self):
        super().process_color()
        if self.gray is not None:
            avg_gray = np.mean(self.gray)
            self.status = 1 if self.skill_gray_min <= avg_gray <= self.skill_gray_max else 0

# 其他窗口可继承 BloodWindow 或 SkillWindow
class MagicWindow(BloodWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, blood_gray_min=70, blood_gray_max=120)

class EnergyWindow(BloodWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, blood_gray_min=135, blood_gray_max=165)

class SkillTSWindow(SkillWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, skill_gray_min=150, skill_gray_max=190)

class SkillFBWindow(SkillWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, skill_gray_min=150, skill_gray_max=190)

class GunShiWindow(SkillWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, skill_gray_min=210, skill_gray_max=255)

class HuluWindow(GrayWindow):
    def __init__(self, sx, sy, ex, ey, hulu_gray_min=120, hulu_gray_max=255):
        super().__init__(sx, sy, ex, ey)
        self.hulu_gray_min = hulu_gray_min
        self.hulu_gray_max = hulu_gray_max

    def process_color(self):
        super().process_color()
        if self.gray is not None:
            middle_col_1 = self.gray[:, self.gray.shape[1] // 3]
            middle_col_2 = self.gray[:, 2 * self.gray.shape[1] // 3]
            clipped_col_1 = np.clip(middle_col_1, self.hulu_gray_min, self.hulu_gray_max)
            clipped_col_2 = np.clip(middle_col_2, self.hulu_gray_min, self.hulu_gray_max)
            count_1 = np.count_nonzero(clipped_col_1 == middle_col_1)
            count_2 = np.count_nonzero(clipped_col_2 == middle_col_2)
            total_length = len(middle_col_1)
            self.status = ((count_1 + count_2) / (2 * total_length)) * 100


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
        BaseWindow.update_all()

        print(f"All windows offset by ({offset_x}, {offset_y})")
        return True
    else:
        print("Failed to find the game logo, offsets not set.")
        return False


# 预实例化所有窗口对象
game_window = BaseWindow(0, 0, 1280, 720)

self_blood_window = BloodWindow(140, 656, 345, 665)
self_magic_window = MagicWindow(141, 669, 366, 675)
self_energy_window = EnergyWindow(140, 678, 352, 682)

skill_1_window = SkillWindow(1110, 571, 1120, 580)
skill_2_window = SkillWindow(1147, 571, 1156, 580)
skill_3_window = SkillWindow(1184, 571, 1193, 580)
skill_4_window = SkillWindow(1221, 571, 1230, 580)

skill_ts_window = SkillTSWindow(995, 694, 1005, 703)
skill_fb_window = SkillFBWindow(1061, 694, 1071, 703)

gunshi1_window = GunShiWindow(1191, 691, 1198, 697)
gunshi2_window = GunShiWindow(1205, 681, 1211, 686)
gunshi3_window = GunShiWindow(1211, 663, 1219, 671)

hulu_window = HuluWindow(82,645,88,679)

boss_blood_window = BloodWindow(512, 609, 776, 616)

battle_roi_window = BaseWindow(411, 70, 961, 670)


