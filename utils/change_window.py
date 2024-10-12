import pygetwindow as gw
import ctypes
import win32gui
from log import log

WUKONG_TITLE = "b1  "
WUKONG_CLASS_NAME = "UnrealWindow"  # 也可以使用类名去寻找窗口 getWindowsWithClass


def get_window_position(title):
    try:
        window = gw.getWindowsWithTitle(title)[0]  # 获取第一个匹配的窗口
        return window.topleft, window.bottomright
    except IndexError:
        print(f"No window with title '{title}' found.")
        return None, None


def move_window(title, x, y):
    try:
        window = gw.getWindowsWithTitle(title)[0]  # 获取第一个匹配的窗口
        window.moveTo(x, y)
    except IndexError:
        print(f"No window with title '{title}' found.")


# 移动窗口到左上角
def set_window_topleft():
    move_window(WUKONG_TITLE, -8, 0)


def is_window_visible(window_title):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        return window.visible  # 检查窗口是否可见
    except IndexError:
        return False  # 没有找到窗口


def is_window_active(window_title):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        return window.isActive  # 检查窗口是否为活动窗口
    except IndexError:
        return False  # 没有找到窗口


def restore_window(window_title):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        if window.isMinimized:  # 如果窗口最小化
            window.restore()  # 恢复窗口
    except IndexError:
        print(f"Window titled '{window_title}' not found.")


# 校正窗口位置
def correction_window():
    if not is_window_visible(WUKONG_TITLE):
        print(f"{WUKONG_TITLE} is not visible.")
        restore_window(WUKONG_TITLE)  # 尝试恢复窗口
        gw.getWindowsWithTitle(WUKONG_TITLE)[0].activate()  # 激活窗口
        set_window_topleft()

    elif not is_window_active(WUKONG_TITLE):
        print(f"{WUKONG_TITLE} is in the background.")
        restore_window(WUKONG_TITLE)  # 尝试恢复窗口
        gw.getWindowsWithTitle(WUKONG_TITLE)[0].activate()  # 激活窗口
        set_window_topleft()

    else:
        print(f"{WUKONG_TITLE} is visible and active.")


def get_window_resolution(window_title):
    # 获取窗口句柄
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd:
        # 使用 GetClientRect 获取窗口的客户区域
        rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        return width, height
    else:
        print(f"未找到标题为 '{window_title}' 的窗口。")
        return None


# 检查游戏窗口分辨率是否和配置一致
def check_window_resolution_same(weight, height):
    resolution = get_window_resolution(WUKONG_TITLE)
    log.debug(f"实际分辨率：{resolution}")
    return resolution[0] == weight and resolution[1] == height
