# grabscreen.py
import mss
import numpy as np

camera = None

def init_camera(target_fps):
    global camera
    if camera is None:
        print("Initializing MSS Screen Capture")
        camera = mss.mss()  # 创建mss实例

def grab_screen():
    try:
        # 捕捉主显示器
        monitor = camera.monitors[1]  # 主显示器通常是索引1
        sct_img = camera.grab(monitor)
        
        # 转换为numpy数组并处理通道
        frame = np.array(sct_img)
        return frame
    except Exception as e:
        print(f"Screen capture error: {e}")
        return None