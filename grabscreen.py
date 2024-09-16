# coding=utf-8
import dxcam_cpp as dxcam  # 使用 dxcam_cpp 替代 win32 来抓取屏幕
import numpy as np
import atexit

import win32gui, win32ui, win32con, win32api


def grab_screen_win32(region=None):
    hwin = win32gui.GetDesktopWindow()
    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype="uint8")
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img


camera = None


def init_camera(target_fps=60):
    global camera
    if camera is None:
        camera = dxcam.create(output_idx=0, output_color="BGRA")
        camera.start(target_fps=target_fps, video_mode=True)

        # 注册程序退出时的清理操作
        atexit.register(camera.stop)


def grab_screen_dxcam(region=None):
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


def grab_screen(region=None):
    # dxcam 效率更高, 如果你的平台支持dx的话
    return grab_screen_dxcam(region)
    # return grab_screen_win32(region)
