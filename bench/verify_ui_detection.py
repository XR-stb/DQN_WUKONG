"""
🔍 UI 检测实时验证叠加层
==========================
独立运行的透明叠加窗口，直接覆盖在游戏画面上显示：
  - 所有 UI 检测区域的矩形框（血条/技能/棍势/Boss等）
  - 每个区域的实时检测数值（百分比 or ON/OFF）
  - ROI 战斗截取区域虚线框
  - 右上角检测结果汇总面板

使用 WGC 截屏 + window.py 的实际检测逻辑，
与训练时的检测方式完全一致。

使用方法：
  python bench/verify_ui_detection.py

  ESC 退出。F1 切换显示/隐藏。
"""

import sys
import os
import time
import yaml
import ctypes
import ctypes.wintypes

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pygame
import win32gui
import win32con
import win32api

import grabscreen
import window

# ============================================================
# 配置加载
# ============================================================

def _load_yaml(relative_path):
    config_path = os.path.join(os.path.dirname(__file__), '..', relative_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

_game_conf = _load_yaml('config/game_conf.yaml')

GAME_WIDTH = _game_conf['game_window']['width']
GAME_HEIGHT = _game_conf['game_window']['height']
BASE_WIDTH = _game_conf['base_resolution']['width']
BASE_HEIGHT = _game_conf['base_resolution']['height']
SCALE_X = GAME_WIDTH / BASE_WIDTH
SCALE_Y = GAME_HEIGHT / BASE_HEIGHT

ROI_X_SIZE = _game_conf['roi']['x_size']
ROI_Y_SIZE = _game_conf['roi']['y_size']

FPS = 15
COLORKEY = (1, 1, 1)
GAME_WINDOW_TITLE = "b1  "

# ============================================================
# UI 检测区域定义（与 overlay_hud.py 一致）
# ============================================================

def _convert_coords(base_coords):
    sx, sy, ex, ey = base_coords
    return (round(sx * SCALE_X), round(sy * SCALE_Y),
            round(ex * SCALE_X), round(ey * SCALE_Y))


def _build_ui_regions():
    regions = []
    ui = _game_conf['ui_coordinates']
    region_defs = [
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
        ('q_window',    'Q',       (255, 255, 100), 'q_window',    'none'),
    ]
    for conf_key, name, color, data_key, val_type in region_defs:
        if conf_key in ui:
            coords = _convert_coords(ui[conf_key])
            regions.append({
                'name': name, 'coords': coords, 'color': color,
                'data_key': data_key, 'val_type': val_type,
            })

    active_boss = _game_conf['active_boss']
    boss_coords_base = _game_conf['boss_blood_presets'].get(active_boss)
    if boss_coords_base:
        coords = _convert_coords(boss_coords_base)
        regions.append({
            'name': f'Boss({active_boss})', 'coords': coords,
            'color': (255, 50, 50), 'data_key': 'boss_blood', 'val_type': 'percent',
        })
    return regions


UI_REGIONS = _build_ui_regions()


# ============================================================
# 颜色
# ============================================================
class Colors:
    TRANSPARENT = COLORKEY
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 100)
    RED = (255, 60, 60)
    YELLOW = (255, 220, 50)
    CYAN = (80, 220, 255)
    GRAY = (160, 160, 160)
    DARK_GRAY = (80, 80, 80)
    PANEL_BG = (15, 15, 30)
    SKILL_ON = (0, 255, 120)
    SKILL_OFF = (120, 120, 120)


# ============================================================
# 游戏窗口定位
# ============================================================

def get_game_window_client_pos():
    hwnd = win32gui.FindWindow(None, GAME_WINDOW_TITLE)
    if not hwnd:
        hwnd = win32gui.FindWindow("UnrealWindow", None)
    if not hwnd:
        return (0, 0)
    pt = ctypes.wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return (pt.x, pt.y)


def make_window_transparent(hwnd):
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
# window.py 中各窗口对象与区域的映射
# ============================================================
WINDOW_MAP = {
    'self_blood':  window.self_blood_window,
    'self_magic':  window.self_magic_window,
    'self_energy': window.self_energy_window,
    'hulu':        window.hulu_window,
    'boss_blood':  window.boss_blood_window,
    'skill_1':     window.skill_1_window,
    'skill_2':     window.skill_2_window,
    'skill_3':     window.skill_3_window,
    'skill_4':     window.skill_4_window,
    'skill_ts':    window.skill_ts_window,
    'skill_fb':    window.skill_fb_window,
    'gunshi1':     window.gunshi1_window,
    'gunshi2':     window.gunshi2_window,
    'gunshi3':     window.gunshi3_window,
    'q_window':    window.q_window,
}


# ============================================================
# 主叠加层类
# ============================================================

class VerifyOverlay:
    """实时验证叠加层：截屏 → 检测 → 在游戏画面上显示检测框和数值"""

    def __init__(self):
        # 初始化截屏
        print("⏳ 初始化 WGC 截屏引擎...")
        grabscreen.init_camera(target_fps=30)
        time.sleep(1)
        print("✅ 截屏引擎就绪")

        # 游戏窗口位置
        self.game_x, self.game_y = get_game_window_client_pos()
        print(f"🎯 游戏窗口客户区位置: ({self.game_x}, {self.game_y})")

        # 初始化 pygame 透明窗口
        os.environ['SDL_VIDEO_WINDOW_POS'] = f'{self.game_x},{self.game_y}'
        pygame.init()
        self.screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT), pygame.NOFRAME)
        pygame.display.set_caption("UI Verify Overlay")

        self.hwnd = pygame.display.get_wm_info()['window']
        make_window_transparent(self.hwnd)

        self._init_fonts()
        self.clock = pygame.time.Clock()

        self.show_overlay = True
        self.status = {}  # 最新检测结果
        self.detect_fps = 0  # 检测帧率统计
        self._fps_counter = 0
        self._fps_timer = time.time()

        # 窗口跟踪
        self.last_pos_check = 0

    def _init_fonts(self):
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

    # ============================================================
    # 截屏 + 检测
    # ============================================================

    def detect(self):
        """截屏并用 window.py 实际检测所有 UI 区域"""
        frame = grabscreen.grab_screen()
        if frame is None:
            return

        # 直接调用底层方法，避免 set_windows_offset 每帧打印日志
        if grabscreen._use_wgc:
            window.BaseWindow.set_offset(0, 0)
        else:
            window.BaseWindow.set_offset(0, 30)
        window.BaseWindow.set_frame(frame)
        window.BaseWindow.update_all()

        # 读取所有状态
        self.status = {}
        for key, win_obj in WINDOW_MAP.items():
            self.status[key] = win_obj.get_status()

        # FPS 统计
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self.detect_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer = now

    # ============================================================
    # 绘制
    # ============================================================

    def draw(self):
        self.screen.fill(COLORKEY)

        if not self.show_overlay:
            self._draw_toggle_hint()
            return

        self._draw_ui_regions()
        self._draw_roi_region()
        self._draw_summary_panel()
        self._draw_toggle_hint()

    def _draw_label_with_bg(self, text_surf, x, y):
        bg_surf = pygame.Surface((text_surf.get_width() + 4, text_surf.get_height() + 2))
        bg_surf.fill(Colors.PANEL_BG)
        bg_surf.set_alpha(200)
        self.screen.blit(bg_surf, (x - 2, y - 1))
        self.screen.blit(text_surf, (x, y))

    def _draw_ui_regions(self):
        """绘制所有 UI 检测区域矩形框 + 标签 + 实时数值"""
        skill_group = []
        tsfb_group = []
        gs_group = []

        for region in UI_REGIONS:
            sx, sy, ex, ey = region['coords']
            name = region['name']
            color = region['color']
            data_key = region['data_key']
            val_type = region['val_type']
            w = ex - sx
            h = ey - sy

            value = self.status.get(data_key)

            # 边框样式
            if val_type == 'bool' and value is not None:
                draw_color = Colors.SKILL_ON if value else Colors.SKILL_OFF
                border_w = 2 if value else 1
            else:
                draw_color = color
                border_w = 1

            # 矩形框（对于太小的区域，扩大绘制范围使其可见）
            draw_w = max(w, 12)
            draw_h = max(h, 12)
            draw_sx = sx - (draw_w - w) // 2
            draw_sy = sy - (draw_h - h) // 2
            pygame.draw.rect(self.screen, draw_color, (draw_sx, draw_sy, draw_w, draw_h), border_w)

            # 百分比条 - 在框内绘制填充
            if val_type == 'percent' and value is not None and h >= 4:
                fill_w = max(0, int(w * value / 100.0))
                fill_surf = pygame.Surface((fill_w, max(h - 2, 1)))
                fill_surf.fill(draw_color)
                fill_surf.set_alpha(80)
                self.screen.blit(fill_surf, (sx + 1, sy + 1))

            # 分组收集（延迟绘制标签）
            region_info = {'region': region, 'value': value, 'draw_color': draw_color}
            if name.startswith('Skill'):
                skill_group.append(region_info)
                continue
            elif name in ('TS', 'FB'):
                tsfb_group.append(region_info)
                continue
            elif name.startswith('GS'):
                gs_group.append(region_info)
                continue

            # --- 单独标签 ---
            if val_type == 'percent' and value is not None:
                label_text = f"{name}: {value:.0f}%"
            elif val_type == 'none':
                label_text = name
            else:
                label_text = name

            text_surf = self.font_xs.render(label_text, True, draw_color)

            if name == 'Hulu':
                label_x = sx - text_surf.get_width() - 4
                label_y = sy
            elif name.startswith('Boss'):
                label_x = ex + 4
                label_y = sy - 2
            elif name == 'Q':
                label_x = ex + 3
                label_y = sy
            else:
                label_x = ex + 4
                label_y = sy - 2

            if label_x < 0:
                label_x = ex + 3
            if label_x + text_surf.get_width() > GAME_WIDTH:
                label_x = sx - text_surf.get_width() - 4

            self._draw_label_with_bg(text_surf, label_x, label_y)

        # --- 技能组（每个框上方单独标注） ---
        for info in skill_group:
            r = info['region']
            sx, sy, ex, ey = r['coords']
            value = info['value']
            draw_color = info['draw_color']
            short = r['name'][-1]
            status = "✓" if value else "✗" if value is not None else "?"
            text_surf = self.font_xs.render(f"S{short}:{status}", True, draw_color)
            label_x = sx + (ex - sx) // 2 - text_surf.get_width() // 2
            label_y = sy - 16
            self._draw_label_with_bg(text_surf, label_x, label_y)

        # --- TS/FB 组（每个框上方单独标注） ---
        for info in tsfb_group:
            r = info['region']
            sx, sy, ex, ey = r['coords']
            value = info['value']
            draw_color = info['draw_color']
            status = "ON" if value else "OFF" if value is not None else "?"
            text_surf = self.font_xs.render(f"{r['name']}:{status}", True, draw_color)
            label_x = sx + (ex - sx) // 2 - text_surf.get_width() // 2
            label_y = sy - 16
            self._draw_label_with_bg(text_surf, label_x, label_y)

        # --- 棍势组（每个框左侧单独标注） ---
        for info in gs_group:
            r = info['region']
            sx, sy, ex, ey = r['coords']
            value = info['value']
            draw_color = info['draw_color']
            status = "ON" if value else "OFF" if value is not None else "?"
            text_surf = self.font_xs.render(f"{r['name']}:{status}", True, draw_color)
            label_x = sx - text_surf.get_width() - 4
            label_y = sy + (ey - sy) // 2 - text_surf.get_height() // 2
            self._draw_label_with_bg(text_surf, label_x, label_y)

    def _draw_roi_region(self):
        """绘制 ROI 区域虚线框"""
        roi_sx = GAME_WIDTH // 2 - ROI_X_SIZE // 2
        roi_sy = GAME_HEIGHT // 2 - ROI_Y_SIZE // 2
        roi_ex = roi_sx + ROI_X_SIZE
        roi_ey = roi_sy + ROI_Y_SIZE

        dash_len = 8
        gap_len = 5
        roi_color = (100, 200, 255)

        for side_y in [roi_sy, roi_ey]:
            x = roi_sx
            while x < roi_ex:
                end_x = min(x + dash_len, roi_ex)
                pygame.draw.line(self.screen, roi_color, (x, side_y), (end_x, side_y), 1)
                x += dash_len + gap_len

        for side_x in [roi_sx, roi_ex]:
            y = roi_sy
            while y < roi_ey:
                end_y = min(y + dash_len, roi_ey)
                pygame.draw.line(self.screen, roi_color, (side_x, y), (side_x, end_y), 1)
                y += dash_len + gap_len

        text = self.font_xs.render("ROI", True, roi_color)
        self.screen.blit(text, (roi_sx + 3, roi_sy + 3))

    def _draw_summary_panel(self):
        """右上角显示检测结果汇总面板"""
        panel_w = 210
        panel_h = 200
        x = GAME_WIDTH - panel_w - 8
        y = 8

        # 半透明背景
        bg = pygame.Surface((panel_w, panel_h))
        bg.fill(Colors.PANEL_BG)
        bg.set_alpha(200)
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, Colors.CYAN, (x, y, panel_w, panel_h), 1)

        # 标题
        title = self.font_title.render("🔍 UI Verify", True, Colors.CYAN)
        self.screen.blit(title, (x + 6, y + 4))

        fps_text = self.font_xs.render(f"Detect FPS: {self.detect_fps:.1f}", True, Colors.GRAY)
        self.screen.blit(fps_text, (x + panel_w - fps_text.get_width() - 6, y + 6))

        cy = y + 24

        # 百分比类
        percent_items = [
            ('HP',     'self_blood',  Colors.GREEN),
            ('MP',     'self_magic',  (80, 160, 255)),
            ('Energy', 'self_energy', Colors.YELLOW),
            ('Hulu',   'hulu',        (200, 100, 255)),
            ('Boss',   'boss_blood',  Colors.RED),
        ]
        for label, key, color in percent_items:
            val = self.status.get(key, 0)
            # 进度条
            bar_x = x + 50
            bar_w = 110
            bar_h = 10
            pygame.draw.rect(self.screen, (40, 40, 60), (bar_x, cy + 2, bar_w, bar_h))
            fill = max(0, min(bar_w, int(bar_w * val / 100.0)))
            if fill > 0:
                pygame.draw.rect(self.screen, color, (bar_x, cy + 2, fill, bar_h))
            pygame.draw.rect(self.screen, (80, 80, 100), (bar_x, cy + 2, bar_w, bar_h), 1)

            # 标签
            label_surf = self.font_xs.render(f"{label}:", True, color)
            self.screen.blit(label_surf, (x + 6, cy))

            # 数值
            val_surf = self.font_xs.render(f"{val:.0f}%", True, Colors.WHITE)
            self.screen.blit(val_surf, (bar_x + bar_w + 4, cy))

            cy += 18

        cy += 4

        # 技能状态 (紧凑2列)
        bool_items = [
            ('S1', 'skill_1'), ('S2', 'skill_2'), ('S3', 'skill_3'), ('S4', 'skill_4'),
            ('TS', 'skill_ts'), ('FB', 'skill_fb'),
            ('G1', 'gunshi1'), ('G2', 'gunshi2'), ('G3', 'gunshi3'),
        ]
        col = 0
        row_y = cy
        for label, key in bool_items:
            val = self.status.get(key, 0)
            icon = "●" if val else "○"
            color = Colors.SKILL_ON if val else Colors.SKILL_OFF
            text_surf = self.font_xs.render(f"{icon} {label}", True, color)
            tx = x + 6 + col * 68
            self.screen.blit(text_surf, (tx, row_y))
            col += 1
            if col >= 3:
                col = 0
                row_y += 16

        # Q 检测
        row_y += 20
        q_val = self.status.get('q_window', 0)
        q_color = Colors.GREEN if q_val else Colors.DARK_GRAY
        q_text = self.font_xs.render(f"Q: {'Detected' if q_val else 'None'}", True, q_color)
        self.screen.blit(q_text, (x + 6, row_y))

    def _draw_toggle_hint(self):
        """左上角提示"""
        if self.show_overlay:
            hint = "🔍 VERIFY ON · F1 Hide · ESC Quit"
            color = Colors.CYAN
        else:
            hint = "🔍 VERIFY OFF · F1 Show · ESC Quit"
            color = Colors.DARK_GRAY
        text_surf = self.font_xs.render(hint, True, color)
        bg = pygame.Surface((text_surf.get_width() + 8, text_surf.get_height() + 4))
        bg.fill(Colors.PANEL_BG)
        bg.set_alpha(160)
        self.screen.blit(bg, (8, 6))
        self.screen.blit(text_surf, (12, 8))

    # ============================================================
    # 窗口跟踪
    # ============================================================

    def _track_game_window(self):
        """跟踪游戏窗口位置"""
        now = time.time()
        if now - self.last_pos_check < 2.0:
            return
        self.last_pos_check = now
        new_x, new_y = get_game_window_client_pos()
        if (new_x, new_y) != (self.game_x, self.game_y) and (new_x, new_y) != (0, 0):
            self.game_x, self.game_y = new_x, new_y
            win32gui.SetWindowPos(
                self.hwnd, win32con.HWND_TOPMOST,
                self.game_x, self.game_y,
                GAME_WIDTH, GAME_HEIGHT,
                win32con.SWP_NOACTIVATE
            )

    # ============================================================
    # 主循环
    # ============================================================

    def run(self):
        print("=" * 50)
        print("🔍 UI 检测实时验证叠加层")
        print(f"📐 游戏窗口: {GAME_WIDTH}x{GAME_HEIGHT}")
        print(f"📦 检测区域: {len(UI_REGIONS)} 个")
        for r in UI_REGIONS:
            sx, sy, ex, ey = r['coords']
            print(f"   [{r['name']}] ({sx},{sy})-({ex},{ey})")
        print(f"🔄 帧率: {FPS} FPS")
        print("按 ESC 退出，F1 切换显示/隐藏")
        print("=" * 50)

        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_F1:
                            self.show_overlay = not self.show_overlay
                            print(f"🔄 叠加层: {'显示' if self.show_overlay else '隐藏'}")

                self._track_game_window()
                self.detect()
                self.draw()
                pygame.display.flip()
                self.clock.tick(FPS)

        except KeyboardInterrupt:
            print("\n收到中断信号")
        finally:
            grabscreen.stop()
            try:
                pygame.quit()
            except Exception:
                pass
            print("✅ 验证叠加层已退出")


if __name__ == '__main__':
    overlay = VerifyOverlay()
    overlay.run()
