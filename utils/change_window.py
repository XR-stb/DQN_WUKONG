import pygetwindow as gw

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


# # 使用你的窗口标题
# topleft, bottom_right = get_window_position('b1')
# print(topleft, bottom_right)

def set_window_topleft():
    # 移动窗口到左上角
    move_window('b1', 0, 0)