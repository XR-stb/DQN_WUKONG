# grabscreen.py
import dxcam_cpp as dxcam
import numpy as np
import atexit
from log import log
import time

camera = None

def init_camera(target_fps):
    global camera
    if camera is None:
        log.debug("Initializing Screen Capture Camera.\n")
        camera = dxcam.create(output_idx=0, output_color="BGRA")
        camera.start(target_fps=target_fps, video_mode=True)
        atexit.register(camera.stop)
        time.sleep(2) # 需要等待两秒彻底初始完

def grab_screen():
    
    frame = camera.get_latest_frame()
    if frame is None:
        print("No frame received.")
    return frame

