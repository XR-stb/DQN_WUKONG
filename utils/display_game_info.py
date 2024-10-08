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

    print("Press q to confirm.")
    
    # Wait until the user presses 'q' to exit
    while True:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

class GameStatusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Game Status")

        # 创建左右两个框架
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # 存储变量及其对应的标签
        self.variables = {}


    def add_variable(self, var_name, var_type='float', column='left'):
        """
        添加一个新的追踪变量到GUI。

        :param var_name: 变量的名称，用于显示和更新
        :param var_type: 变量的类型，'float' 或 'bool'
        :param column: 'left' 或 'right'，决定标签显示在哪一栏
        """
        frame = self.left_frame if column == 'left' else self.right_frame

        # 创建标签
        label = tk.Label(frame, text=f"{var_name}: 0.00%")
        label.pack(anchor='w', pady=2)

        # 存储变量信息
        self.variables[var_name] = {
            'type': var_type,
            'label': label
        }

    def update_status(self, **kwargs):
        """
        更新多个变量的状态。

        :param kwargs: 以变量名为键，变量值为值的键值对
        """
        for var_name, value in kwargs.items():
            if var_name in self.variables:
                var_info = self.variables[var_name]
                var_type = var_info['type']
                label = var_info['label']

                if var_type == 'float':
                    label.config(text=f"{var_name}: {value:.2f}%")
                elif var_type == 'bool':
                    text = "Active" if value else "Inactive"
                    label.config(text=f"{var_name}: {text}")
                else:
                    label.config(text=f"{var_name}: {value}")
            else:
                print(f"Warning: Variable '{var_name}' not found in GUI.")



# 主程序循环，显示玩家的血条数值，并支持优雅退出
def main_loop():
    root = tk.Tk()
    app = GameStatusApp(root)


    # 添加初始变量（示例）
    app.add_variable("self_blood", var_type='float', column='left')
    app.add_variable("self_magic", var_type='float', column='left')
    app.add_variable("self_energy", var_type='float', column='left')
    app.add_variable("hulu", var_type='float', column='left')
    app.add_variable("boss_blood", var_type='float', column='left')
    app.add_variable("skill_1", var_type='bool', column='right')
    app.add_variable("skill_2", var_type='bool', column='right')
    app.add_variable("skill_3", var_type='bool', column='right')
    app.add_variable("skill_4", var_type='bool', column='right')

    app.add_variable("skill_ts", var_type='bool', column='right')
    app.add_variable("skill_fb", var_type='bool', column='right')

    app.add_variable("gunshi1", var_type='bool', column='right')
    app.add_variable("gunshi2", var_type='bool', column='right')
    app.add_variable("gunshi3", var_type='bool', column='right')

    app.add_variable("q_found", var_type='bool', column='right')



    if wait_for_game_window():
        display_gui_elements()

        # 进入主循环
        while running:
            frame = grabscreen.grab_screen()
            BaseWindow.set_frame(frame)
            BaseWindow.update_all()

            is_similar, similarity_score = q_window.check_similarity("./images/q.png", threshold=0.8)

            # 更新 Tkinter 界面上的状态
            app.update_status(
                **{
                    "self_blood": self_blood_window.get_status(),
                    "self_magic": self_magic_window.get_status(),
                    "self_energy": self_energy_window.get_status(),
                    "hulu": hulu_window.get_status(),
                    "boss_blood": boss_blood_window.get_status(),
                    "skill_1": skill_1_window.get_status(),
                    "skill_2": skill_2_window.get_status(),
                    "skill_3": skill_3_window.get_status(),
                    "skill_4": skill_4_window.get_status(),
                    "skill_ts": skill_ts_window.get_status(),
                    "skill_fb": skill_fb_window.get_status(),
                    "gunshi1": gunshi1_window.get_status(),
                    "gunshi2": gunshi2_window.get_status(),
                    "gunshi3": gunshi3_window.get_status(),

                    "q_found": is_similar,
                }
            )

            # 更新 Tkinter 窗口
            root.update_idletasks()
            root.update()


if __name__ == "__main__":
    print("start main_loop")
    main_loop()
    print("Program has exited cleanly.")
