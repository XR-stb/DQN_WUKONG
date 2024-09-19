# coding=utf-8
import dxcam_cpp as dxcam  # 使用 dxcam_cpp 替代 win32 来抓取屏幕
import numpy as np
import atexit

camera = None


def init_camera(target_fps=60):
    global camera
    if camera is None:
        camera = dxcam.create(output_idx=0, output_color="BGRA")
        camera.start(target_fps=target_fps, video_mode=True)

        # 注册程序退出时的清理操作
        atexit.register(camera.stop)


def grab_screen(region=None):
    # 初始化摄像头
    init_camera(target_fps=60)

    # 获取最新的帧
    frame = camera.get_latest_frame()

    if frame is None:
        print("No frame received.")
        return None

    # 如果有指定 region 则裁剪图像
    if region:
        left, top, x2, y2 = region
        frame = frame[top : (y2 + 1), left : (x2 + 1)]  # 裁剪 frame

    return frame

