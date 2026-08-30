"""
🎮 AI训练实时OSD叠加层 (Overlay HUD)

在游戏画面上实时叠加显示AI训练数据，类似外挂HUD效果。
特性：
    - 透明背景、始终置顶、鼠标穿透（不影响游戏操作）
    - 实时显示：回合/步数、当前动作、血量条、奖励、胜率等
    - 在游戏画面对应位置绘制 UI 检测区域矩形框（血条、技能、Boss血条等）
    - 从 realtime_state.json 读取训练进程写入的实时数据

使用方法：
    python utils/overlay_hud.py

按 ESC 退出。
"""

import sys
import os
import json
import time
import yaml

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ctypes
import ctypes.wintypes
import pygame
import win32gui
import win32con
import win32api
import pygetwindow as gw

# ============================================================
# 配置加载
# ============================================================

def _load_yaml(relative_path):
    config_path = os.path.join(os.path.dirname(__file__), '..', relative_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

_game_conf = _load_yaml('config/game_conf.yaml')
_dash_conf = _load_yaml('config/dashboard_conf.yaml')

# 游戏窗口尺寸
GAME_WIDTH = _game_conf['game_window']['width']
GAME_HEIGHT = _game_conf['game_window']['height']

# 基准分辨率
BASE_WIDTH = _game_conf['base_resolution']['width']
BASE_HEIGHT = _game_conf['base_resolution']['height']

# 缩放因子
SCALE_X = GAME_WIDTH / BASE_WIDTH
SCALE_Y = GAME_HEIGHT / BASE_HEIGHT

# ROI 区域
ROI_X_SIZE = _game_conf['roi']['x_size']
ROI_Y_SIZE = _game_conf['roi']['y_size']

# 实时状态文件路径
REALTIME_JSON = os.path.join(
    os.path.dirname(__file__), '..',
    _dash_conf['data']['train_data_dir'],
    'realtime_state.json'
)

# OSD 刷新帧率
FPS = 15

# 底部监控面板高度
DASHBOARD_HEIGHT = 220

# OSD 窗口总高度（游戏区域 + 底部监控面板）
OSD_TOTAL_HEIGHT = GAME_HEIGHT + DASHBOARD_HEIGHT

# 透明色键（用于实现窗口透明）
COLORKEY = (1, 1, 1)

# 游戏窗口标题（与 change_window.py 保持一致）
GAME_WINDOW_TITLE = "b1  "


# ============================================================
# UI 检测区域定义
# ============================================================

def _convert_coords(base_coords):
    """将基准分辨率坐标转换为实际游戏窗口坐标"""
    sx, sy, ex, ey = base_coords
    return (
        round(sx * SCALE_X),
        round(sy * SCALE_Y),
        round(ex * SCALE_X),
        round(ey * SCALE_Y),
    )


def _build_ui_regions():
    """从 game_conf.yaml 构建所有 UI 检测区域"""
    regions = []
    ui = _game_conf['ui_coordinates']

    # --- 玩家状态 ---
    # 颜色分类：血量=红色系，法力=蓝色系，精力=黄色系，技能=绿色系，葫芦=紫色系
    region_defs = [
        # (配置键, 显示名称, 颜色, 数据key, 值类型)
        ('self_blood',  'HP',      (0, 255, 100),   'self_blood',  'percent'),
        ('self_magic',  'MP',      (80, 160, 255),  'self_magic',  'percent'),
        ('self_energy', 'Energy',  (255, 220, 50),  'self_energy', 'percent'),
        ('hulu',        'Hulu',    (200, 100, 255), 'hulu',        'percent'),
        ('skill_1',     'Skill1',  (0, 220, 180),   'skill_1',     'bool'),
        ('skill_2',     'Skill2',  (0, 220, 180),   'skill_2',     'bool'),
        ('skill_3',     'Skill3',  (0, 220, 180),   'skill_3',     'bool'),
        ('skill_4',     'Skill4',  (0, 220, 180),   'skill_4',     'bool'),
        ('skill_ts',    'TS',      (255, 160, 40),  'skill_ts',    'bool'),
        ('skill_fb',    'FB',      (255, 160, 40),  'skill_fb',    'bool'),
        ('gunshi1',     'GS1',     (255, 80, 80),   'gunshi1',     'bool'),
        ('gunshi2',     'GS2',     (255, 80, 80),   'gunshi2',     'bool'),
        ('gunshi3',     'GS3',     (255, 80, 80),   'gunshi3',     'bool'),
        ('q_window',    'Q',       (255, 255, 100), None,          'none'),
    ]

    for conf_key, name, color, data_key, val_type in region_defs:
        if conf_key in ui:
            coords = _convert_coords(ui[conf_key])
            regions.append({
                'name': name,
                'coords': coords,  # (sx, sy, ex, ey) 实际像素坐标
                'color': color,
                'data_key': data_key,
                'val_type': val_type,
            })

    # --- Boss 血条 ---
    active_boss = _game_conf['active_boss']
    boss_coords_base = _game_conf['boss_blood_presets'].get(active_boss)
    if boss_coords_base:
        coords = _convert_coords(boss_coords_base)
        regions.append({
            'name': f'Boss({active_boss})',
            'coords': coords,
            'color': (255, 50, 50),
            'data_key': 'boss_blood',
            'val_type': 'percent',
        })

    return regions


UI_REGIONS = _build_ui_regions()


# ============================================================
# 颜色定义
# ============================================================
class Colors:
    TRANSPARENT = COLORKEY
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 100)
    RED = (255, 60, 60)
    YELLOW = (255, 220, 50)
    CYAN = (80, 220, 255)
    ORANGE = (255, 160, 40)
    PURPLE = (180, 100, 255)
    GRAY = (160, 160, 160)
    DARK_GRAY = (80, 80, 80)
    PANEL_BG = (15, 15, 30)
    DASHBOARD_BG = (10, 12, 25)
    BAR_BG = (40, 40, 60)
    HP_GREEN = (50, 205, 50)
    HP_RED = (220, 50, 50)
    BOSS_ORANGE = (220, 120, 20)
    BOSS_RED = (200, 30, 30)
    SKILL_ON = (0, 255, 120)
    SKILL_OFF = (120, 120, 120)
    CHART_GRID = (30, 30, 50)
    CHART_REWARD = (0, 200, 255)
    CHART_SELF_HP = (50, 255, 50)
    CHART_BOSS_HP = (255, 80, 80)
    BAR_ATTACK = (255, 140, 40)
    BAR_DODGE = (80, 200, 255)
    BAR_SKILL = (200, 100, 255)
    BAR_MOVE = (100, 200, 100)
    BAR_OTHER = (160, 160, 160)


# ============================================================
# 游戏窗口定位
# ============================================================

def get_game_window_client_pos():
    """获取游戏窗口客户区左上角在屏幕上的坐标
    
    Returns:
        (x, y) 客户区左上角的屏幕坐标，找不到窗口返回 (0, 0)
    """
    hwnd = win32gui.FindWindow(None, GAME_WINDOW_TITLE)
    if not hwnd:
        # 也尝试通过类名找
        hwnd = win32gui.FindWindow("UnrealWindow", None)
    if not hwnd:
        return (0, 0)
    
    # 获取客户区左上角 (0,0) 在屏幕坐标系中的位置
    pt = ctypes.wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return (pt.x, pt.y)


# ============================================================
# Win32 透明窗口工具函数
# ============================================================

def make_window_transparent(hwnd):
    """设置窗口为透明、置顶、鼠标穿透"""
    ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    win32gui.SetWindowLong(
        hwnd, win32con.GWL_EXSTYLE,
        ex_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOOLWINDOW
    )
    win32gui.SetLayeredWindowAttributes(
        hwnd, win32api.RGB(*COLORKEY), 0, win32con.LWA_COLORKEY
    )
    win32gui.SetWindowPos(
        hwnd, win32con.HWND_TOPMOST,
        0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
    )


# ============================================================
# OSD 绘制器
# ============================================================

class OverlayHUD:
    """透明置顶 OSD 叠加层"""

    def __init__(self, running_event=None, visible_event=None):
        """
        Args:
            running_event: multiprocessing.Event，控制进程生命周期（clear时退出）。
                          为 None 时作为独立脚本运行（ESC退出）。
            visible_event: multiprocessing.Event，控制OSD显示/隐藏（F1切换）。
                          set=显示，clear=隐藏。为 None 时默认显示。
        """
        self.running_event = running_event
        self.visible_event = visible_event
        # 先获取游戏窗口客户区位置
        self.game_x, self.game_y = get_game_window_client_pos()
        print(f"[OSD] 游戏窗口客户区位置: ({self.game_x}, {self.game_y})")

        os.environ['SDL_VIDEO_WINDOW_POS'] = f'{self.game_x},{self.game_y}'
        pygame.init()

        self.screen = pygame.display.set_mode(
            (GAME_WIDTH, OSD_TOTAL_HEIGHT),
            pygame.NOFRAME
        )
        pygame.display.set_caption("DQN_WUKONG OSD")

        self.hwnd = pygame.display.get_wm_info()['window']
        make_window_transparent(self.hwnd)

        self._init_fonts()
        self.clock = pygame.time.Clock()

        # 缓存的状态数据
        self.state = {}
        self.last_read_time = 0
        self.read_interval = 0.05

        # 奖励历史
        self.reward_history = []
        self.max_history = 60

        # 动画计数器
        self.frame_count = 0

        # 窗口跟踪间隔
        self.last_pos_check = 0
        self.pos_check_interval = 2.0  # 每2秒重新检查游戏窗口位置

        # OSD 绘制开关：WGC 按窗口句柄捕获，不会截到 OSD 叠加层，因此默认开启所有绘制
        # 由 visible_event 控制显示/隐藏，或者独立运行时由 F1 本地切换
        self.debug_overlay = True
        # 上一帧的可见状态，用于检测变化并打印日志
        self._prev_visible = True

    def _init_fonts(self):
        """初始化字体"""
        font_names = ['Microsoft YaHei', 'SimHei', 'Arial']
        self.font_path = None
        for name in font_names:
            try:
                font = pygame.font.SysFont(name, 16)
                if font:
                    self.font_path = name
                    break
            except Exception:
                continue

        self.font_xs = pygame.font.SysFont(self.font_path, 11)
        self.font_sm = pygame.font.SysFont(self.font_path, 13)
        self.font_md = pygame.font.SysFont(self.font_path, 16)
        self.font_lg = pygame.font.SysFont(self.font_path, 20)
        self.font_title = pygame.font.SysFont(self.font_path, 14, bold=True)

    def _sync_visible_event(self):
        """同步 visible_event 到 debug_overlay（由主进程F1控制）"""
        if self.visible_event is not None:
            new_visible = self.visible_event.is_set()
            if new_visible != self._prev_visible:
                self.debug_overlay = new_visible
                self._prev_visible = new_visible
                mode_str = "显示" if new_visible else "隐藏"
                print(f"[OSD] 画面覆盖层: {mode_str}")

    def read_state(self):
        """从 JSON 文件读取实时状态"""
        now = time.time()
        if now - self.last_read_time < self.read_interval:
            return

        self.last_read_time = now
        try:
            if os.path.exists(REALTIME_JSON):
                with open(REALTIME_JSON, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data and data.get('timestamp', 0) > 0:
                    self.state = data
                    reward = data.get('reward', 0)
                    self.reward_history.append(reward)
                    if len(self.reward_history) > self.max_history:
                        self.reward_history.pop(0)
        except (json.JSONDecodeError, IOError, OSError):
            pass

        # 定期重新检查游戏窗口位置，确保 OSD 跟随游戏窗口
        if now - self.last_pos_check > self.pos_check_interval:
            self.last_pos_check = now
            new_x, new_y = get_game_window_client_pos()
            if (new_x, new_y) != (self.game_x, self.game_y) and (new_x, new_y) != (0, 0):
                self.game_x, self.game_y = new_x, new_y
                win32gui.SetWindowPos(
                    self.hwnd, win32con.HWND_TOPMOST,
                    self.game_x, self.game_y,
                    GAME_WIDTH, OSD_TOTAL_HEIGHT,
                    win32con.SWP_NOACTIVATE
                )

    # ============================================================
    # 主绘制入口
    # ============================================================

    def draw(self):
        """绘制 OSD 叠加层
        
        WGC 按窗口句柄捕获游戏画面，不会截到 OSD 叠加层，
        因此默认显示所有 UI 检测框和数据面板。
        按 F1 可切换显示/隐藏游戏画面区域内的绘制内容。
        """
        self.screen.fill(COLORKEY)

        # --- 游戏画面区域内的绘制：受 F1 开关控制 ---
        if self.debug_overlay:
            self._draw_ui_regions()
            self._draw_roi_region()

            if not self.state:
                self._draw_waiting()
            else:
                self._draw_top_bar()
                # 不再绘制左下角血量面板，因为UI检测框已经显示了HP/Boss血量
                self._draw_action_panel()
                self._draw_reward_panel()
                self._draw_mini_chart()
                self._draw_stats_panel()

        # --- 底部训练监控面板（始终绘制，位于游戏画面下方，不影响截屏） ---
        self._draw_dashboard()

        # 始终显示 F1 提示（左上角小标签）
        self._draw_toggle_hint()

        self.frame_count += 1

    # ============================================================
    # UI 检测区域矩形框绘制
    # ============================================================

    def _draw_ui_regions(self):
        """在游戏画面对应位置绘制所有 UI 检测区域的矩形框 + 标签 + 数值
        
        标签放置策略（避免重叠）：
        - 血量/法力/精力(HP/MP/Energy): 标签放在框的右侧
        - 葫芦(Hulu): 标签放在框的左侧
        - 技能类(Skill1-4): 水平排列，统一放在框的上方一行
        - TS/FB: 水平排列，统一放在框的下方一行
        - 棍势(GS1-3): 标签放在框的左侧
        - Boss血条: 标签放在框的右侧
        - Q窗口: 标签放在框右侧
        """
        has_data = bool(self.state)
        ui_data = self.state.get('ui', {})

        # 收集技能组和TS/FB组，用于分组排列标签
        skill_group = []  # Skill1-4
        tsfb_group = []   # TS, FB
        gs_group = []     # GS1-3

        for region in UI_REGIONS:
            sx, sy, ex, ey = region['coords']
            name = region['name']
            color = region['color']
            data_key = region['data_key']
            val_type = region['val_type']
            w = ex - sx
            h = ey - sy

            # 获取实时数值（仅在有训练数据时才读取，避免误显示 0%）
            value = None
            if has_data:
                if data_key == 'self_blood':
                    value = self.state.get('self_blood', 0)
                elif data_key == 'boss_blood':
                    value = self.state.get('boss_blood', 0)
                elif data_key and data_key in ui_data:
                    value = ui_data[data_key]

            # 根据类型决定边框样式
            if val_type == 'bool' and value is not None:
                if value:
                    draw_color = Colors.SKILL_ON
                    border_w = 2
                else:
                    draw_color = Colors.SKILL_OFF
                    border_w = 1
            else:
                draw_color = color
                border_w = 1

            # 绘制矩形框
            pygame.draw.rect(self.screen, draw_color, (sx, sy, w, h), border_w)

            # 对于血量类区域，在框内绘制一个填充比例条
            if val_type == 'percent' and value is not None and h >= 4:
                fill_w = max(0, int(w * value / 100.0))
                fill_color = (*draw_color[:3],)
                fill_surf = pygame.Surface((fill_w, max(h - 2, 1)))
                fill_surf.fill(fill_color)
                fill_surf.set_alpha(80)
                self.screen.blit(fill_surf, (sx + 1, sy + 1))

            # 收集分组信息（延迟绘制标签，避免重叠）
            region_info = {
                'region': region, 'value': value, 'draw_color': draw_color
            }
            if name.startswith('Skill'):
                skill_group.append(region_info)
                continue
            elif name in ('TS', 'FB'):
                tsfb_group.append(region_info)
                continue
            elif name.startswith('GS'):
                gs_group.append(region_info)
                continue

            # --- 单独绘制非分组标签 ---
            label_text = name
            if has_data and val_type == 'percent' and value is not None:
                label_text = f"{name}: {value:.0f}%"

            text_surf = self.font_xs.render(label_text, True, draw_color)

            # 标签位置策略
            if name == 'Hulu':
                # 葫芦：放在框左侧
                label_x = sx - text_surf.get_width() - 4
                label_y = sy
            elif name.startswith('Boss'):
                # Boss血条：放在框右侧
                label_x = ex + 4
                label_y = sy - 2
            elif name == 'Q':
                # Q窗口：放在框右侧
                label_x = ex + 3
                label_y = sy
            else:
                # HP/MP/Energy：放在框右侧
                label_x = ex + 4
                label_y = sy - 2

            # 安全边界检查
            if label_x < 0:
                label_x = ex + 3
            if label_x + text_surf.get_width() > GAME_WIDTH:
                label_x = sx - text_surf.get_width() - 4

            self._draw_label_with_bg(text_surf, label_x, label_y)

        # --- 技能组标签（Skill1-4 水平排列在同一行，放在最左边技能框上方） ---
        if skill_group:
            # 找到技能组最上方的Y坐标和最左边的X坐标
            min_sy = min(r['region']['coords'][1] for r in skill_group)
            label_y = min_sy - 16  # 统一放在框上方
            label_x = min(r['region']['coords'][0] for r in skill_group)

            for info in skill_group:
                r = info['region']
                value = info['value']
                draw_color = info['draw_color']
                short = r['name'][-1]  # 取数字部分: 1,2,3,4
                if has_data and value is not None:
                    status = "✓" if value else "✗"
                else:
                    status = "?"
                label_text = f"S{short}:{status}"
                text_surf = self.font_xs.render(label_text, True, draw_color)
                self._draw_label_with_bg(text_surf, label_x, label_y)
                label_x += text_surf.get_width() + 6

        # --- TS/FB组标签（水平排列在同一行，放在框上方） ---
        if tsfb_group:
            min_sy = min(r['region']['coords'][1] for r in tsfb_group)
            label_y = min_sy - 16
            label_x = min(r['region']['coords'][0] for r in tsfb_group)

            for info in tsfb_group:
                r = info['region']
                value = info['value']
                draw_color = info['draw_color']
                if has_data and value is not None:
                    status = "ON" if value else "OFF"
                else:
                    status = "?"
                label_text = f"{r['name']}:{status}"
                text_surf = self.font_xs.render(label_text, True, draw_color)
                self._draw_label_with_bg(text_surf, label_x, label_y)
                label_x += text_surf.get_width() + 6

        # --- 棍势组标签（GS1-3 垂直排列在最左边框的左侧） ---
        if gs_group:
            min_sx = min(r['region']['coords'][0] for r in gs_group)
            label_x_base = min_sx - 50  # 放在框左侧

            for i, info in enumerate(gs_group):
                r = info['region']
                value = info['value']
                draw_color = info['draw_color']
                sy = r['coords'][1]
                if has_data and value is not None:
                    status = "ON" if value else "OFF"
                else:
                    status = "?"
                label_text = f"{r['name']}:{status}"
                text_surf = self.font_xs.render(label_text, True, draw_color)
                self._draw_label_with_bg(text_surf, label_x_base, sy - 2)

    def _draw_label_with_bg(self, text_surf, x, y):
        """绘制带半透明背景的标签文字"""
        bg_surf = pygame.Surface((text_surf.get_width() + 4, text_surf.get_height() + 2))
        bg_surf.fill(Colors.PANEL_BG)
        bg_surf.set_alpha(180)
        self.screen.blit(bg_surf, (x - 2, y - 1))
        self.screen.blit(text_surf, (x, y))

    def _draw_roi_region(self):
        """绘制 ROI 战斗截取区域（虚线矩形）"""
        roi_sx = GAME_WIDTH // 2 - ROI_X_SIZE // 2
        roi_sy = GAME_HEIGHT // 2 - ROI_Y_SIZE // 2
        roi_ex = roi_sx + ROI_X_SIZE
        roi_ey = roi_sy + ROI_Y_SIZE

        # 用虚线风格绘制（交替绘制短线段）
        dash_len = 8
        gap_len = 5
        roi_color = (100, 200, 255)

        # 上边和下边
        for side_y in [roi_sy, roi_ey]:
            x = roi_sx
            while x < roi_ex:
                end_x = min(x + dash_len, roi_ex)
                pygame.draw.line(self.screen, roi_color, (x, side_y), (end_x, side_y), 1)
                x += dash_len + gap_len

        # 左边和右边
        for side_x in [roi_sx, roi_ex]:
            y = roi_sy
            while y < roi_ey:
                end_y = min(y + dash_len, roi_ey)
                pygame.draw.line(self.screen, roi_color, (side_x, y), (side_x, end_y), 1)
                y += dash_len + gap_len

        # ROI 标签
        text = self.font_xs.render("ROI", True, roi_color)
        self.screen.blit(text, (roi_sx + 3, roi_sy + 3))

    # ============================================================
    # 数据面板绘制（保留之前的所有面板）
    # ============================================================

    def _draw_waiting(self):
        """无训练数据时不额外显示面板，只靠 _draw_toggle_hint 和 UI 区域框即可"""
        pass

    def _draw_toggle_hint(self):
        """左上角始终显示 OSD 模式提示"""
        if self.debug_overlay:
            hint = "🎮 OSD ON · F1 Hide"
            color = Colors.CYAN
        else:
            hint = "🎮 OSD OFF · F1 Show"
            color = Colors.DARK_GRAY
        text_surf = self.font_xs.render(hint, True, color)
        bg = pygame.Surface((text_surf.get_width() + 8, text_surf.get_height() + 4))
        bg.fill(Colors.PANEL_BG)
        bg.set_alpha(160)
        self.screen.blit(bg, (8, 6))
        self.screen.blit(text_surf, (12, 8))

    def _draw_panel_bg(self, x, y, w, h, alpha_color=None):
        """绘制面板背景"""
        bg_color = alpha_color or Colors.PANEL_BG
        panel = pygame.Surface((w, h))
        panel.fill(bg_color)
        panel.set_alpha(200)
        self.screen.blit(panel, (x, y))
        pygame.draw.rect(self.screen, (60, 60, 90), (x, y, w, h), 1)

    def _draw_top_bar(self):
        """顶部信息条"""
        bar_h = 28
        bar_w = 520
        x = (GAME_WIDTH - bar_w) // 2
        y = 5

        self._draw_panel_bg(x, y, bar_w, bar_h)

        episode = self.state.get('episode', 0)
        step = self.state.get('step', 0)
        duration = self.state.get('duration', 0)
        epsilon = self.state.get('epsilon', 0)
        total_ep = self.state.get('total_episodes', 0)

        items = [
            (f"EP: {episode}", Colors.CYAN),
            (f"Step: {step}", Colors.WHITE),
            (f"Time: {duration:.1f}s", Colors.YELLOW),
            (f"Eps: {epsilon:.3f}", Colors.ORANGE),
            (f"Total: {total_ep}", Colors.GRAY),
        ]

        cx = x + 12
        for text_str, color in items:
            text = self.font_title.render(text_str, True, color)
            self.screen.blit(text, (cx, y + 7))
            cx += text.get_width() + 20

    def _draw_blood_bars(self):
        """血量条面板"""
        panel_w = 220
        panel_h = 58
        x = 10
        y = GAME_HEIGHT - panel_h - 80

        self._draw_panel_bg(x, y, panel_w, panel_h)

        self_blood = self.state.get('self_blood', 0)
        boss_blood = self.state.get('boss_blood', 0)

        self._draw_bar(
            x + 10, y + 8, 200, 14,
            self_blood / 100.0,
            f"HP: {self_blood:.0f}%",
            Colors.HP_GREEN if self_blood > 30 else Colors.HP_RED
        )
        self._draw_bar(
            x + 10, y + 32, 200, 14,
            boss_blood / 100.0,
            f"BOSS: {boss_blood:.0f}%",
            Colors.BOSS_ORANGE if boss_blood > 30 else Colors.BOSS_RED
        )

    def _draw_bar(self, x, y, w, h, ratio, label, color):
        """绘制进度条"""
        ratio = max(0, min(1, ratio))
        pygame.draw.rect(self.screen, Colors.BAR_BG, (x, y, w, h))
        if ratio > 0:
            pygame.draw.rect(self.screen, color, (x, y, int(w * ratio), h))
        pygame.draw.rect(self.screen, (80, 80, 100), (x, y, w, h), 1)
        text = self.font_sm.render(label, True, Colors.WHITE)
        self.screen.blit(text, (x + 4, y + 1))

    def _draw_action_panel(self):
        """当前动作面板"""
        panel_w = 200
        panel_h = 50
        x = GAME_WIDTH - panel_w - 10
        y = 40

        self._draw_panel_bg(x, y, panel_w, panel_h)

        action = self.state.get('action', '---')
        reward = self.state.get('reward', 0)

        action_color = Colors.GREEN if reward > 0 else (Colors.RED if reward < 0 else Colors.WHITE)
        text = self.font_md.render(f"ACT: {action}", True, action_color)
        self.screen.blit(text, (x + 8, y + 6))

        reward_color = Colors.GREEN if reward > 0 else (Colors.RED if reward < 0 else Colors.GRAY)
        sign = '+' if reward > 0 else ''
        text = self.font_sm.render(f"Reward: {sign}{reward:.1f}", True, reward_color)
        self.screen.blit(text, (x + 8, y + 28))

    def _draw_reward_panel(self):
        """回合累计奖励面板"""
        panel_w = 200
        panel_h = 50
        x = GAME_WIDTH - panel_w - 10
        y = 96

        self._draw_panel_bg(x, y, panel_w, panel_h)

        ep_reward = self.state.get('episode_reward', 0)
        best_reward = self.state.get('best_reward', 0)
        injured = self.state.get('injured_count', 0)

        color = Colors.GREEN if ep_reward > 0 else (Colors.RED if ep_reward < -100 else Colors.YELLOW)
        sign = '+' if ep_reward > 0 else ''
        text = self.font_md.render(f"EP Reward: {sign}{ep_reward:.0f}", True, color)
        self.screen.blit(text, (x + 8, y + 6))

        text = self.font_sm.render(
            f"Hurt: {injured}x  |  Best: {best_reward:.0f}", True, Colors.GRAY
        )
        self.screen.blit(text, (x + 8, y + 30))

    def _draw_mini_chart(self):
        """迷你奖励曲线"""
        if len(self.reward_history) < 3:
            return

        chart_w = 200
        chart_h = 50
        x = GAME_WIDTH - chart_w - 10
        y = 152

        self._draw_panel_bg(x, y, chart_w, chart_h)

        text = self.font_sm.render("Reward Trend", True, Colors.GRAY)
        self.screen.blit(text, (x + 4, y + 2))

        data = self.reward_history
        n = len(data)
        if n < 2:
            return

        max_val = max(abs(v) for v in data) if data else 1
        if max_val == 0:
            max_val = 1

        chart_area_y = y + 16
        chart_area_h = chart_h - 20
        mid_y = chart_area_y + chart_area_h // 2

        pygame.draw.line(self.screen, (50, 50, 70), (x + 2, mid_y), (x + chart_w - 2, mid_y), 1)

        points = []
        for i, val in enumerate(data):
            px = x + 4 + int((chart_w - 8) * i / (n - 1))
            py = mid_y - int((val / max_val) * (chart_area_h // 2 - 2))
            py = max(chart_area_y + 2, min(chart_area_y + chart_area_h - 2, py))
            points.append((px, py))

        if len(points) >= 2:
            line_color = Colors.GREEN if data[-1] >= 0 else Colors.RED
            pygame.draw.lines(self.screen, line_color, False, points, 1)

    def _draw_stats_panel(self):
        """统计面板"""
        panel_w = 200
        panel_h = 36
        x = GAME_WIDTH - panel_w - 10
        y = 208

        self._draw_panel_bg(x, y, panel_w, panel_h)

        wr10 = self.state.get('win_rate_10', 0) * 100
        avg10 = self.state.get('avg_reward_10', 0)

        wr_color = Colors.GREEN if wr10 > 50 else (Colors.YELLOW if wr10 > 20 else Colors.RED)
        text = self.font_sm.render(f"WR(10): {wr10:.0f}%", True, wr_color)
        self.screen.blit(text, (x + 8, y + 10))

        sign = '+' if avg10 > 0 else ''
        avg_color = Colors.GREEN if avg10 > 0 else Colors.RED
        text = self.font_sm.render(f"AvgR(10): {sign}{avg10:.0f}", True, avg_color)
        self.screen.blit(text, (x + 110, y + 10))

    # ============================================================
    # 底部训练监控面板
    # ============================================================

    def _draw_dashboard(self):
        """绘制底部训练监控面板（游戏画面下方，全屏宽度）"""
        ep_stats = self.state.get('episode_stats', {})

        # 面板起始位置
        dash_y = GAME_HEIGHT + 2
        dash_w = GAME_WIDTH
        dash_h = DASHBOARD_HEIGHT - 4

        # 绘制面板大背景
        bg = pygame.Surface((dash_w, dash_h))
        bg.fill(Colors.DASHBOARD_BG)
        bg.set_alpha(230)
        self.screen.blit(bg, (0, dash_y))
        pygame.draw.line(self.screen, Colors.CYAN, (0, dash_y), (dash_w, dash_y), 1)

        # ---- 布局规划 ----
        # | 奖励曲线 (35%) | 血量曲线 (35%) | 动作分布+技能+指标 (30%) |
        margin = 6
        chart1_w = int(dash_w * 0.35) - margin * 2
        chart2_w = int(dash_w * 0.35) - margin * 2
        right_w = dash_w - int(dash_w * 0.70) - margin * 2

        chart1_x = margin
        chart2_x = int(dash_w * 0.35) + margin
        right_x = int(dash_w * 0.70) + margin

        chart_h = dash_h - 30
        chart_top = dash_y + 22

        # ---- 标题栏 ----
        title = self.font_title.render("📊 TRAINING DASHBOARD", True, Colors.CYAN)
        self.screen.blit(title, (margin + 4, dash_y + 4))

        # 实时血量信息（从主状态读取，不依赖 episode_stats）
        self_blood = self.state.get('self_blood', 0)
        boss_blood = self.state.get('boss_blood', 0)
        hp_color = Colors.HP_GREEN if self_blood > 30 else Colors.HP_RED
        boss_color = Colors.BOSS_ORANGE if boss_blood > 30 else Colors.BOSS_RED
        ep_num = self.state.get('episode', 0)
        step_num = self.state.get('step', 0)
        ep_reward_val = self.state.get('episode_reward', 0)

        # 在标题右侧显示实时状态
        rt_x = 200  # 默认起始位置
        if self.state:
            rt_items = [
                (f"EP:{ep_num}", Colors.CYAN),
                (f"Step:{step_num}", Colors.WHITE),
                (f"HP:{self_blood:.0f}%", hp_color),
                (f"Boss:{boss_blood:.0f}%", boss_color),
                (f"R:{ep_reward_val:+.0f}", Colors.GREEN if ep_reward_val > 0 else Colors.RED),
            ]
            for text_str, color in rt_items:
                surf = self.font_xs.render(text_str, True, color)
                self.screen.blit(surf, (rt_x, dash_y + 6))
                rt_x += surf.get_width() + 10

        # 如果没有统计数据，显示等待提示
        if not ep_stats:
            hint = self.font_md.render("⏳ Waiting for training data...", True, Colors.DARK_GRAY)
            self.screen.blit(hint, (dash_w // 2 - hint.get_width() // 2, dash_y + dash_h // 2 - 10))
            return

        # 显示关键指标在标题栏右侧（紧接实时状态后面）
        dps = ep_stats.get('dps', 0)
        avg_rps = ep_stats.get('avg_reward_per_step', 0)
        atk = ep_stats.get('attack_count', 0)
        boss_hit = ep_stats.get('boss_hit_count', 0)
        dmg_dealt = ep_stats.get('blood_damage_dealt', 0)
        dmg_taken = ep_stats.get('blood_damage_taken', 0)
        dodge_rate = ep_stats.get('dodge_success_rate', 0) * 100

        kpi_items = [
            (f"DPS: {dps:.1f}", Colors.ORANGE),
            (f"AvgR/s: {avg_rps:+.1f}", Colors.CHART_REWARD),
            (f"ATK: {atk}", Colors.YELLOW),
            (f"HIT: {boss_hit}", Colors.RED),
            (f"DMG↑: {dmg_dealt:.0f}", Colors.GREEN),
            (f"DMG↓: {dmg_taken:.0f}", Colors.HP_RED),
            (f"Dodge: {dodge_rate:.0f}%", Colors.BAR_DODGE),
        ]
        # KPI 指标紧接在实时状态之后
        kpi_x = rt_x + 20
        for text_str, color in kpi_items:
            surf = self.font_xs.render(text_str, True, color)
            self.screen.blit(surf, (kpi_x, dash_y + 6))
            kpi_x += surf.get_width() + 14

        # ---- 1. 奖励曲线 ----
        self._draw_chart_line(
            chart1_x, chart_top, chart1_w, chart_h,
            ep_stats.get('reward_history', []),
            title="Step Reward",
            line_color=Colors.CHART_REWARD,
            zero_line=True
        )

        # ---- 2. 血量曲线 (双线) ----
        self._draw_blood_chart(
            chart2_x, chart_top, chart2_w, chart_h,
            ep_stats.get('self_blood_history', []),
            ep_stats.get('boss_blood_history', [])
        )

        # ---- 3. 右侧区域：动作分布 + 技能使用 ----
        half_h = (chart_h - 4) // 2
        self._draw_action_distribution(
            right_x, chart_top, right_w, half_h,
            ep_stats.get('action_counts', {})
        )
        self._draw_skill_usage(
            right_x, chart_top + half_h + 4, right_w, half_h,
            ep_stats.get('skill_usage', {}),
            ep_stats.get('dodge_count', 0),
            ep_stats.get('dodge_total', 0),
        )

    def _draw_chart_line(self, x, y, w, h, data, title="", line_color=Colors.CYAN, zero_line=False):
        """绘制折线图"""
        # 面板背景
        self._draw_panel_bg(x, y, w, h)

        # 标题
        if title:
            t = self.font_xs.render(title, True, Colors.GRAY)
            self.screen.blit(t, (x + 4, y + 2))

        if not data or len(data) < 2:
            t = self.font_xs.render("Waiting...", True, Colors.DARK_GRAY)
            self.screen.blit(t, (x + w // 2 - 20, y + h // 2 - 5))
            return

        # 绘图区域
        pad_top = 16
        pad_bottom = 14
        pad_lr = 30
        chart_x = x + pad_lr
        chart_y = y + pad_top
        chart_w = w - pad_lr - 4
        chart_h = h - pad_top - pad_bottom

        n = len(data)
        max_val = max(abs(v) for v in data) if data else 1
        if max_val == 0:
            max_val = 1

        # 绘制网格线
        for i in range(5):
            gy = chart_y + int(chart_h * i / 4)
            pygame.draw.line(self.screen, Colors.CHART_GRID, (chart_x, gy), (chart_x + chart_w, gy), 1)

        # 绘制零线
        if zero_line:
            min_val = min(data)
            val_range = max_val - min_val if max_val != min_val else 1
            zero_y = chart_y + int(chart_h * (max_val / (max_val - min_val if max_val != min_val else 1)))
            zero_y = max(chart_y, min(chart_y + chart_h, zero_y))
            pygame.draw.line(self.screen, (60, 60, 80), (chart_x, zero_y), (chart_x + chart_w, zero_y), 1)

        # 绘制曲线
        min_val = min(data)
        val_range = max_val - min_val if max_val != min_val else 1
        points = []
        for i, val in enumerate(data):
            px = chart_x + int(chart_w * i / (n - 1))
            py = chart_y + chart_h - int(chart_h * (val - min_val) / val_range)
            py = max(chart_y, min(chart_y + chart_h, py))
            points.append((px, py))

        if len(points) >= 2:
            pygame.draw.lines(self.screen, line_color, False, points, 2)

        # Y 轴刻度
        max_label = self.font_xs.render(f"{max_val:.0f}", True, Colors.GRAY)
        min_label = self.font_xs.render(f"{min_val:.0f}", True, Colors.GRAY)
        self.screen.blit(max_label, (x + 2, chart_y - 2))
        self.screen.blit(min_label, (x + 2, chart_y + chart_h - 8))

        # 当前值标注
        cur_val = data[-1]
        cur_color = Colors.GREEN if cur_val >= 0 else Colors.RED
        cur_label = self.font_xs.render(f"{cur_val:+.1f}", True, cur_color)
        self.screen.blit(cur_label, (chart_x + chart_w - cur_label.get_width(), y + h - pad_bottom + 1))

    def _draw_blood_chart(self, x, y, w, h, self_hp_data, boss_hp_data):
        """绘制血量双线折线图"""
        self._draw_panel_bg(x, y, w, h)

        t = self.font_xs.render("Blood Curve", True, Colors.GRAY)
        self.screen.blit(t, (x + 4, y + 2))

        # 图例
        legend_x = x + 80
        pygame.draw.line(self.screen, Colors.CHART_SELF_HP, (legend_x, y + 7), (legend_x + 12, y + 7), 2)
        t = self.font_xs.render("Self", True, Colors.CHART_SELF_HP)
        self.screen.blit(t, (legend_x + 15, y + 2))
        legend_x += 50
        pygame.draw.line(self.screen, Colors.CHART_BOSS_HP, (legend_x, y + 7), (legend_x + 12, y + 7), 2)
        t = self.font_xs.render("Boss", True, Colors.CHART_BOSS_HP)
        self.screen.blit(t, (legend_x + 15, y + 2))

        has_self = self_hp_data and len(self_hp_data) >= 2
        has_boss = boss_hp_data and len(boss_hp_data) >= 2
        if not has_self and not has_boss:
            t = self.font_xs.render("Waiting...", True, Colors.DARK_GRAY)
            self.screen.blit(t, (x + w // 2 - 20, y + h // 2 - 5))
            return

        pad_top = 16
        pad_bottom = 14
        pad_lr = 30
        chart_x = x + pad_lr
        chart_y = y + pad_top
        chart_w = w - pad_lr - 4
        chart_h = h - pad_top - pad_bottom

        # 网格
        for i in range(5):
            gy = chart_y + int(chart_h * i / 4)
            pygame.draw.line(self.screen, Colors.CHART_GRID, (chart_x, gy), (chart_x + chart_w, gy), 1)
            # 刻度 100/75/50/25/0
            val = 100 - i * 25
            label = self.font_xs.render(f"{val}", True, Colors.DARK_GRAY)
            self.screen.blit(label, (x + 2, gy - 5))

        def draw_hp_line(data, color):
            if not data or len(data) < 2:
                return
            n = len(data)
            points = []
            for i, val in enumerate(data):
                px = chart_x + int(chart_w * i / (n - 1))
                py = chart_y + chart_h - int(chart_h * max(0, min(100, val)) / 100)
                points.append((px, py))
            if len(points) >= 2:
                pygame.draw.lines(self.screen, color, False, points, 2)

        draw_hp_line(self_hp_data, Colors.CHART_SELF_HP)
        draw_hp_line(boss_hp_data, Colors.CHART_BOSS_HP)

        # 当前值标注
        if has_self:
            v = self_hp_data[-1]
            t = self.font_xs.render(f"{v:.0f}%", True, Colors.CHART_SELF_HP)
            self.screen.blit(t, (chart_x + chart_w - t.get_width() - 30, y + h - pad_bottom + 1))
        if has_boss:
            v = boss_hp_data[-1]
            t = self.font_xs.render(f"{v:.0f}%", True, Colors.CHART_BOSS_HP)
            self.screen.blit(t, (chart_x + chart_w - t.get_width(), y + h - pad_bottom + 1))

    def _draw_action_distribution(self, x, y, w, h, action_counts):
        """绘制动作分布水平柱状图"""
        self._draw_panel_bg(x, y, w, h)

        t = self.font_xs.render("Action Distribution", True, Colors.GRAY)
        self.screen.blit(t, (x + 4, y + 2))

        if not action_counts:
            return

        # 动作分类和颜色映射
        ATTACK_SET = {'LIGHT_ATTACK', 'HEAVY_ATTACK', 'ATTACK_DODGE',
                      'FIVE_HIT_COMBO', 'QIESHOU', 'STEALTH_CHARGE'}
        DODGE_SET = {'DODGE', 'DODGE_TWO', 'DODGE_THREE'}
        SKILL_SET = {'SKILL_1', 'SKILL_2', 'SKILL_3', 'SKILL_4', 'TISHEN', 'FABAO'}
        MOVE_SET = {'GO_FORWARD', 'GO_BACK'}

        def get_color(name):
            if name in ATTACK_SET:
                return Colors.BAR_ATTACK
            elif name in DODGE_SET:
                return Colors.BAR_DODGE
            elif name in SKILL_SET:
                return Colors.BAR_SKILL
            elif name in MOVE_SET:
                return Colors.BAR_MOVE
            return Colors.BAR_OTHER

        # 按使用次数排序，取前8名
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        if not sorted_actions:
            return

        max_count = sorted_actions[0][1] if sorted_actions else 1
        bar_start_y = y + 16
        bar_h = max(8, min(14, (h - 20) // len(sorted_actions) - 2))

        for i, (name, count) in enumerate(sorted_actions):
            by = bar_start_y + i * (bar_h + 2)
            if by + bar_h > y + h:
                break

            # 动作名缩写
            short_name = name[:8]
            label = self.font_xs.render(short_name, True, Colors.WHITE)
            self.screen.blit(label, (x + 4, by))

            # 柱状条
            bar_x = x + 70
            bar_w_max = w - 100
            bar_fill = int(bar_w_max * count / max_count) if max_count > 0 else 0
            bar_color = get_color(name)

            pygame.draw.rect(self.screen, Colors.BAR_BG, (bar_x, by + 1, bar_w_max, bar_h - 2))
            if bar_fill > 0:
                pygame.draw.rect(self.screen, bar_color, (bar_x, by + 1, bar_fill, bar_h - 2))

            # 数量
            count_label = self.font_xs.render(str(count), True, Colors.WHITE)
            self.screen.blit(count_label, (bar_x + bar_w_max + 4, by))

    def _draw_skill_usage(self, x, y, w, h, skill_usage, dodge_count, dodge_total):
        """绘制技能使用统计和闪避统计"""
        self._draw_panel_bg(x, y, w, h)

        t = self.font_xs.render("Skill & Dodge Stats", True, Colors.GRAY)
        self.screen.blit(t, (x + 4, y + 2))

        # 技能使用（紧凑显示）
        SKILL_NAMES = ['SKILL_1', 'SKILL_3', 'SKILL_4', 'TISHEN', 'FABAO', 'STEALTH_CHARGE']
        SHORT_NAMES = ['Sk1', 'Sk3', 'Sk4', 'TS', 'FB', 'SC']
        SKILL_COLORS = [
            Colors.BAR_SKILL, Colors.BAR_SKILL, Colors.BAR_SKILL,
            Colors.ORANGE, Colors.ORANGE, Colors.CYAN
        ]

        col_w = w // 3
        row_h = 14
        sy = y + 16

        for i, (full_name, short, color) in enumerate(zip(SKILL_NAMES, SHORT_NAMES, SKILL_COLORS)):
            count = skill_usage.get(full_name, 0)
            col = i % 3
            row = i // 3
            tx = x + 6 + col * col_w
            ty = sy + row * row_h

            t = self.font_xs.render(f"{short}: {count}", True, color if count > 0 else Colors.DARK_GRAY)
            self.screen.blit(t, (tx, ty))

        # 闪避统计
        dodge_y = sy + 2 * row_h + 4
        dodge_rate = (dodge_count / dodge_total * 100) if dodge_total > 0 else 0
        dodge_color = Colors.GREEN if dodge_rate > 60 else (Colors.YELLOW if dodge_rate > 30 else Colors.RED)

        t = self.font_xs.render(f"Dodge: {dodge_count}/{dodge_total}", True, Colors.BAR_DODGE)
        self.screen.blit(t, (x + 6, dodge_y))

        # 闪避成功率条
        bar_x = x + 90
        bar_w_avail = w - 100
        bar_h_dodge = 8
        pygame.draw.rect(self.screen, Colors.BAR_BG, (bar_x, dodge_y + 2, bar_w_avail, bar_h_dodge))
        if dodge_total > 0:
            fill = int(bar_w_avail * dodge_rate / 100)
            pygame.draw.rect(self.screen, dodge_color, (bar_x, dodge_y + 2, fill, bar_h_dodge))
        rate_label = self.font_xs.render(f"{dodge_rate:.0f}%", True, dodge_color)
        self.screen.blit(rate_label, (bar_x + bar_w_avail + 4, dodge_y))

    # ============================================================
    # 主循环
    # ============================================================

    def run(self):
        """主循环"""
        print("=" * 50)
        print("[OSD] AI训练实时OSD叠加层已启动")
        print(f"[OSD] 游戏窗口: {GAME_WIDTH}x{GAME_HEIGHT}")
        print(f"[OSD] 状态文件: {REALTIME_JSON}")
        print(f"[OSD] 刷新帧率: {FPS} FPS")
        print(f"[OSD] 已加载 {len(UI_REGIONS)} 个UI检测区域")
        print(f"   ROI区域: {ROI_X_SIZE}x{ROI_Y_SIZE}")
        for r in UI_REGIONS:
            sx, sy, ex, ey = r['coords']
            print(f"   [{r['name']}] ({sx},{sy})-({ex},{ey})")
        if self.running_event:
            print("由主程序管理，进程常驻，F1 即时切换显隐")
        else:
            print("按 ESC 退出，F1 切换显示/隐藏")
        print("=" * 50)

        running = True
        try:
            while running:
                # 如果是由主程序通过 Event 控制，检查 Event 状态
                if self.running_event is not None:
                    if not self.running_event.is_set():
                        # Event 被清除，退出 OSD
                        running = False
                        break

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE and self.running_event is None:
                            # 仅独立运行时 ESC 退出
                            running = False
                        elif event.key == pygame.K_F1 and self.visible_event is None:
                            # 仅独立运行时，F1 本地切换显隐
                            # 由主程序管理时，F1 通过 visible_event 控制（process_handler 中处理）
                            self.debug_overlay = not self.debug_overlay
                            mode_str = "显示" if self.debug_overlay else "隐藏"
                            print(f"[OSD] 画面覆盖层: {mode_str}")

                # 同步外部 visible_event 控制的显隐状态
                self._sync_visible_event()
                self.read_state()
                self.draw()
                pygame.display.flip()
                self.clock.tick(FPS)
        except KeyboardInterrupt:
            print("OSD: 收到中断信号，正在退出...")
        except Exception as e:
            print(f"OSD: 发生异常: {e}")
        finally:
            try:
                pygame.quit()
            except Exception:
                pass
            print("OSD 叠加层已退出。")


def run_overlay_process(running_event=None, visible_event=None):
    """作为独立进程运行 OSD 叠加层的入口函数
    
    Args:
        running_event: multiprocessing.Event，控制进程生命周期（clear时退出）。
                      为 None 时作为独立脚本运行。
        visible_event: multiprocessing.Event，控制OSD显示/隐藏（F1即时切换）。
                      为 None 时默认显示，由本地F1控制。
    """
    try:
        hud = OverlayHUD(running_event=running_event, visible_event=visible_event)
        hud.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"OSD 进程异常退出: {e}")


if __name__ == '__main__':
    hud = OverlayHUD()
    hud.run()
