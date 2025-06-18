# window.py
import cv2
import numpy as np
import utils
from utils.change_window import check_window_resolution_same


# 基类，封装静态 offset 和 frame
class BaseWindow:
    offset_x = 0
    offset_y = 0
    frame = None
    all_windows = []
    cached_templates = {}

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
        return BaseWindow.frame[
            self.sy + BaseWindow.offset_y : self.ey + 1 + BaseWindow.offset_y,
            self.sx + BaseWindow.offset_x : self.ex + 1 + BaseWindow.offset_x,
        ]

    def update(self):
        self.color = self.extract_region()

    @staticmethod
    def update_all():
        for window in BaseWindow.all_windows:
            window.update()

    @staticmethod
    def load_template_once(template_image_path):
        """
        加载模板图像并缓存，如果已经加载过则直接使用缓存。
        """
        if template_image_path not in BaseWindow.cached_templates:
            template = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise FileNotFoundError(
                    f"Failed to load template image from {template_image_path}"
                )
            BaseWindow.cached_templates[template_image_path] = template
        return BaseWindow.cached_templates[template_image_path]

    def check_similarity(self, template_image_path, threshold=0.8):
        """
        检查窗口区域内是否包含指定图像，并返回相似度。

        参数:
        - template_image_path: 模板图像的路径
        - threshold: 相似度阈值，默认为0.8

        返回:
        - 如果相似度超过阈值，返回True，否则返回False
        - 匹配的相似度值
        """
        if self.gray is None:
            print("No grayscale data to compare.")
            return False, 0.0

        # 加载模板图像，仅加载一次
        template_image = BaseWindow.load_template_once(template_image_path)

        # 进行模板匹配
        result = cv2.matchTemplate(self.gray, template_image, cv2.TM_CCOEFF_NORMED)

        # 查找匹配结果
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # 返回匹配结果
        return max_val >= threshold, max_val

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
        self.gray = (
            cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)
            if self.color is not None
            else None
        )


# 亮度窗口的基类
class HLSWindow(StatusWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.gray = None

    def process_color(self):
        # 如果 self.color 为空，直接返回
        if self.color is None:
            self.gray = None
            return

        # 将 BGR 图像转换为 HLS 模型
        hls_image = cv2.cvtColor(self.color, cv2.COLOR_BGR2HLS)
        # 提取 L 通道 (亮度) 作为灰度图
        self.gray = hls_image[:, :, 1]  # L 通道为 HLS 的第二个通道 (索引 1)
        self.hls = hls_image


# 血量窗口
class BloodWindow(GrayWindow):
    def __init__(self, sx, sy, ex, ey, blood_gray_min=117, blood_gray_max=255):
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


# 血量窗口 挨打会闪烁红色
class BloodWindowV2(HLSWindow):
    def __init__(self, sx, sy, ex, ey, blood_gray_min=117, blood_gray_max=255):
        super().__init__(sx, sy, ex, ey)
        self.blood_gray_min = blood_gray_min
        self.blood_gray_max = blood_gray_max
        self.last_status = None  # 上一次的血量状态

    def is_color_red(self):
        """通过中间一行的 H 通道检测是否偏红"""
        if self.hls is None:
            return False

        # 采样 hls 图像的中间一行
        middle_row_hls = self.hls[self.hls.shape[0] // 2, :, :]

        # 提取 H 通道
        h_channel = middle_row_hls[:, 0]

        # 红色在 H 通道中的范围：0-15 和 165-180
        red_mask_1 = (h_channel >= 0) & (h_channel <= 15)
        red_mask_2 = (h_channel >= 165) & (h_channel <= 180)
        red_mask = red_mask_1 | red_mask_2

        # 计算红色像素的比例
        red_percentage = np.sum(red_mask) / len(h_channel)

        # 如果中间一行有超过 40% 的像素是红色，则判断为偏红
        return red_percentage > 0.4

    def process_color(self):
        super().process_color()
        if self.gray is not None:
            # 检测是否整体偏红
            is_red = self.is_color_red()

            # 采样中间一行用于血量检测
            middle_row = self.gray[self.gray.shape[0] // 2, :]
            clipped = np.clip(middle_row, self.blood_gray_min, self.blood_gray_max)
            count = np.count_nonzero(clipped == middle_row)
            total_length = len(middle_row)
            new_status = (count / total_length) * 100  # 计算新的血量百分比

            if is_red:
                # 如果整体偏红，只允许血量下降
                if self.last_status is None or new_status < self.last_status:
                    self.status = new_status
                else:
                    self.status = self.last_status  # 保持上一次的状态
            else:
                # 正常更新状态
                self.status = new_status

            # 更新 last_status
            self.last_status = self.status


# 技能窗口
class SkillWindow(HLSWindow):
    def __init__(self, sx, sy, ex, ey, skill_gray_min=100, skill_gray_max=160):
        super().__init__(sx, sy, ex, ey)
        self.skill_gray_min = skill_gray_min
        self.skill_gray_max = skill_gray_max

    def process_color(self):
        super().process_color()
        if self.gray is not None:
            avg_gray = np.mean(self.gray)
            self.status = (
                1 if self.skill_gray_min <= avg_gray <= self.skill_gray_max else 0
            )


# 其他窗口可继承 BloodWindow 或 SkillWindow
class MagicWindow(BloodWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, blood_gray_min=80, blood_gray_max=120)


class EnergyWindow(BloodWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, blood_gray_min=135, blood_gray_max=165)


class SkillTSWindow(SkillWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, skill_gray_min=120, skill_gray_max=160)


class SkillFBWindow(SkillWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey, skill_gray_min=120, skill_gray_max=160)


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
            clipped_col_1 = np.clip(
                middle_col_1, self.hulu_gray_min, self.hulu_gray_max
            )
            clipped_col_2 = np.clip(
                middle_col_2, self.hulu_gray_min, self.hulu_gray_max
            )
            count_1 = np.count_nonzero(clipped_col_1 == middle_col_1)
            count_2 = np.count_nonzero(clipped_col_2 == middle_col_2)
            total_length = len(middle_col_1)
            self.status = ((count_1 + count_2) / (2 * total_length)) * 100


# 查找logo位置的函数
def find_game_window_logo(frame, template_path, threshold):
    return (0, 0)
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

        # 根据logo图片再title bar的位置修正
        # offset_x += 10 # 不需要，已经校正窗口位置了
        offset_y += 30

        # 设置偏移量给所有窗口对象
        BaseWindow.set_offset(offset_x, offset_y)
        BaseWindow.set_frame(frame)
        BaseWindow.update_all()

        print(f"All windows offset by ({offset_x}, {offset_y})")
        return True
    else:
        print("Failed to find the game logo, offsets not set.")
        return False


# 实际游戏窗口大小
game_width = 3840  # NOTE: 替换成你游戏的宽度和分辨率
game_height = 2160

# 基准窗口大小: 勿动，否则需要连同下方所有数值一起做修改
base_width = 1280 # 勿动
base_height = 720 # 勿动

# 计算缩放因子
width_scale = game_width / base_width
height_scale = game_height / base_height


# 坐标转换函数
def convert_coordinates(x1, y1, x2, y2):
    # return x1, y1, x2, y2
    new_x1 = round(x1 * width_scale)
    new_y1 = round(y1 * height_scale)
    new_x2 = round(x2 * width_scale)
    new_y2 = round(y2 * height_scale)
    return new_x1, new_y1, new_x2, new_y2


game_window = BaseWindow(0, 0, game_width, game_height)
# 转换后的窗口坐标
self_blood_window = BloodWindow(*convert_coordinates(138, 646, 320, 655))
self_magic_window = MagicWindow(*convert_coordinates(140, 658, 365, 664))
self_energy_window = EnergyWindow(*convert_coordinates(140, 669, 280, 673))

skill_1_window = SkillWindow(*convert_coordinates(1107, 562, 1117, 571))
skill_2_window = SkillWindow(*convert_coordinates(1144, 562, 1153, 571))
skill_3_window = SkillWindow(*convert_coordinates(1181, 562, 1190, 571))
skill_4_window = SkillWindow(*convert_coordinates(1218, 562, 1227, 571))

skill_ts_window = SkillTSWindow(*convert_coordinates(992, 682, 1002, 691))
skill_fb_window = SkillFBWindow(*convert_coordinates(1058, 682, 1068, 691))

gunshi1_window = GunShiWindow(*convert_coordinates(1191, 682, 1198, 688))
gunshi2_window = GunShiWindow(*convert_coordinates(1205, 672, 1211, 677))
gunshi3_window = GunShiWindow(*convert_coordinates(1211, 654, 1219, 662))

hulu_window = HuluWindow(*convert_coordinates(80, 632, 85, 668))

q_window = SkillWindow(*convert_coordinates(185, 533, 195, 542))

# boss_blood_window = BloodWindow(*convert_coordinates(512, 609, 776, 616)) # 寅虎
# boss_blood_window = BloodWindow(*convert_coordinates(460, 599, 836, 606)) # 虎先锋
boss_blood_window = BloodWindow(*convert_coordinates(510, 599, 776, 606)) # 广谋
# boss_blood_window = BloodWindow(*convert_coordinates(455, 609, 836, 616)) # 青背龙

roi_x_size = 300  # ROI的宽度和高度（以游戏窗口中心为中心的矩形）
roi_y_size = 400  # ROI的宽度和高度（以游戏窗口中心为中心的矩形）
start_xy = (
    game_width // 2 - roi_x_size // 2,
    game_height // 2 - roi_y_size // 2,
)  # ROI的起始坐标 (x, y)


battle_roi_window = BaseWindow(
    start_xy[0], start_xy[1], start_xy[0] + roi_x_size, start_xy[1] + roi_y_size
)
