import cv2
import time
import window
import grabscreen
import signal
import sys

# 标志位，表示是否继续运行
running = True

# 处理 Ctrl+C 的函数
def signal_handler(sig, frame):
    global running
    print("\nGracefully exiting...")
    running = False

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

# 等待游戏窗口出现的函数
def wait_for_game_window():
    while running:
        frame = grabscreen.grab_screen()

        if frame is not None:
            if window.set_windows_offset(frame):
                print("Game window detected and offsets set!")
                return frame

            print("Failed to find the game logo, offsets not set.")

        time.sleep(1)

# 展示游戏窗口的GUI，只显示game_window的区域，并等待用户按 'q' 键退出
def display_game_window_with_offsets():
    # 从 frame 中提取 game_window 的区域
    game_window_frame = window.BaseWindow.frame[
        window.game_window.sy + window.BaseWindow.offset_y : window.game_window.ey + window.BaseWindow.offset_y + 1,
        window.game_window.sx + window.BaseWindow.offset_x : window.game_window.ex + window.BaseWindow.offset_x + 1
    ]

    # 绘制self_blood_window的矩形框
    cv2.rectangle(game_window_frame, 
                  (window.self_blood_window.sx, window.self_blood_window.sy),
                  (window.self_blood_window.ex, window.self_blood_window.ey),
                  (0, 0, 255), 3)  # 红色的矩形框

    # 创建一个窗口，并设置其为置顶窗口
    cv2.namedWindow("Game Window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Game Window", cv2.WND_PROP_TOPMOST, 1)

    # 显示只包含 game_window 的区域
    cv2.imshow("Game Window", game_window_frame)

    # 等待用户按 'q' 键退出
    while True:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # 关闭窗口
    cv2.destroyAllWindows()

# 主程序循环，显示玩家的血条数值，并支持优雅退出
def main_loop():
    frame = wait_for_game_window()  # 等待游戏窗口出现

    if frame is not None:
        display_game_window_with_offsets()

        # 进入主循环
        while running:
            window.BaseWindow.update_all()

            # 获取玩家的血条值
            blood_percentage = window.self_blood_window.blood_count()
            print(f"Player's health: {blood_percentage:.2f}%")

            time.sleep(1)

if __name__ == "__main__":
    print("start main_loop")
    main_loop()
    print("Program has exited cleanly.")
