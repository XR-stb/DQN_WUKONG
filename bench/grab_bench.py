# coding=utf-8
import cv2
import numpy as np
import time
import win32gui, win32ui, win32con, win32api
import dxcam_cpp as dxcam  # 使用 dxcam_cpp 替代 win32 来抓取屏幕
import matplotlib.pyplot as plt

# Win32 screen capture function
def win32_grab_screen(region=None):
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
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img

# DXCam screen capture function
camera = None

def init_camera(target_fps=60):
    global camera
    if camera is None:
        camera = dxcam.create(output_idx=0, output_color="BGRA")
        camera.start(target_fps=target_fps, video_mode=True)

def dxcam_grab_screen(region=None):
    init_camera(target_fps=60)
    frame = camera.get_latest_frame()

    if frame is None:
        print("No frame received.")
        return None

    if region:
        left, top, x2, y2 = region
        frame = frame[top:(y2+1), left:(x2+1)]

    return frame

# Performance testing
def test_performance(grab_func, num_frames=100, region=None):
    times = []
    for _ in range(num_frames):
        start_time = time.time()
        grab_func(region)  # Call the screen capture function
        end_time = time.time()
        times.append(end_time - start_time)
    return times

# Plotting function
def plot_performance(win32_times, dxcam_times, num_frames):
    plt.figure(figsize=(10, 6))
    plt.plot(num_frames, win32_times, label="Win32 Screen Capture", marker='o')
    plt.plot(num_frames, dxcam_times, label="DXCam Screen Capture", marker='o')
    plt.title("Performance Comparison: Win32 vs DXCam")
    plt.xlabel("Number of Frames")
    plt.ylabel("Average Time per Frame (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to run the tests
def main():
    num_tests = [10, 50, 100, 200, 500]  # Varying number of frames for testing
    region = None  # Full screen capture

    win32_times = []
    dxcam_times = []

    for num in num_tests:
        print(f"Testing with {num} frames...")

        # Test win32 capture
        win32_time = test_performance(win32_grab_screen, num_frames=num, region=region)
        avg_win32_time = np.mean(win32_time)
        win32_times.append(avg_win32_time)

        # Test dxcam capture
        dxcam_time = test_performance(dxcam_grab_screen, num_frames=num, region=region)
        avg_dxcam_time = np.mean(dxcam_time)
        dxcam_times.append(avg_dxcam_time)

    # Plot performance results
    plot_performance(win32_times, dxcam_times, num_tests)

if __name__ == "__main__":
    main()
