# grabscreen.py
import dxcam_cpp as dxcam
import numpy as np
import atexit
from log import log

camera = None

def init_camera(target_fps):
    global camera
    if camera is None:
        log("Initializing Screen Capture Camera.\n")
        camera = dxcam.create(output_idx=0, output_color="BGRA")
        camera.start(target_fps=target_fps, video_mode=True)
        atexit.register(camera.stop)

def grab_screen():
    
    frame = camera.get_latest_frame()
    if frame is None:
        print("No frame received.")
    return frame

