import cv2
import time
from window import *
import grabscreen
import signal
import sys
import hashlib
import colorsys
import tkinter as tk

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
            if set_windows_offset(frame):
                print("Game window detected and offsets set!")                
                return True

            print("Failed to find the game logo, offsets not set.")

        time.sleep(1)


def display_gui_elements():
    # Ensure that game_window has been updated
    if game_window.color is None:
        print("Game window frame is not available.")
        return
    
    # Create a copy to draw rectangles on
    game_window_frame = game_window.color.copy()
    
    # Iterate through all window instances and draw rectangles
    for win in BaseWindow.all_windows:
        # Get the class name of the window instance
        class_name = win.__class__.__name__.replace('Window', '')
        

        # Define top-left and bottom-right points
        top_left = (win.sx, win.sy)
        bottom_right = (win.ex, win.ey)
        
        # Draw the rectangle on the game_window_frame
        cv2.rectangle(game_window_frame, top_left, bottom_right, (0,0,255), 1)
        
        text_position = (win.ex + 1, win.sy + 6) 
        cv2.putText(game_window_frame, class_name, text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,255,128), 1, cv2.LINE_AA)
        
    
    # Create a window and set it to be always on top
    cv2.namedWindow("Game Window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Game Window", cv2.WND_PROP_TOPMOST, 1)
    
    # Display the frame with all rectangles
    cv2.imshow("Game Window", game_window_frame)

    print("Press q to comfirm.")
    
    # Wait until the user presses 'q' to exit
    while True:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

# 创建 Tkinter 窗口并显示血量、魔法值等状态
class GameStatusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Game Status")

        # 标签显示玩家状态
        self.health_label = tk.Label(root, text="Player's health: 0.00%")
        self.health_label.pack(pady=1)

        self.magic_label = tk.Label(root, text="Magic: 0.00%")
        self.magic_label.pack(pady=1)

        self.energy_label = tk.Label(root, text="Energy: 0.00%" )
        self.energy_label.pack(pady=1)

        self.skill1_label = tk.Label(root, text="Skill_1: 0" )
        self.skill1_label.pack(pady=1)

        # 添加一个退出按钮
        self.quit_button = tk.Button(root, text="Quit", command=self.exit_program)
        self.quit_button.pack(pady=10)

    def update_status(self, blood_percentage, magic_value, energy_value, skill_1):
        # 更新玩家状态信息
        self.health_label.config(text=f"Player's health: {blood_percentage:.2f}%")
        self.magic_label.config(text=f"Magic: {magic_value:.2f}%")
        self.energy_label.config(text=f"Energy: {energy_value:.2f}%")
        self.skill1_label.config(text=f"Skill_1: {int(skill_1)}")

    def exit_program(self):
        global running
        running = False
        self.root.quit()  # 退出 Tkinter 主循环

# 主程序循环，显示玩家的血条数值，并支持优雅退出
def main_loop():
    root = tk.Tk()
    app = GameStatusApp(root)

    if wait_for_game_window():
        display_gui_elements()

        # 进入主循环
        while running:
            frame = grabscreen.grab_screen()
            BaseWindow.set_frame(frame)
            BaseWindow.update_all()

            is_similar, similarity_score = q_window.check_similarity("./images/q.png", threshold=0.9)

            if is_similar:
                # 获取玩家的血条、魔法值、能量值、技能状态
                self_blood = self_blood_window.get_status()
                self_magic = self_magic_window.get_status()
                self_energy = self_energy_window.get_status()
                skill_1 = skill_1_window.get_status()

                # 更新 Tkinter 界面上的状态
                app.update_status(self_blood, self_magic, self_energy, skill_1)

            # 更新 Tkinter 窗口
            root.update_idletasks()
            root.update()

    print("Program has exited cleanly.")

if __name__ == "__main__":
    print("start main_loop")
    main_loop()
    print("Program has exited cleanly.")
