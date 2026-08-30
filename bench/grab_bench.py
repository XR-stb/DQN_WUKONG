# coding=utf-8
"""截屏方案性能对比测试: Win32 (GDI) vs dxcam vs WGC (Windows Graphics Capture)"""
import cv2
import numpy as np
import time
import sys
import os

# 添加项目根目录到 path，以便导入 grabscreen 等模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import win32gui, win32ui, win32con, win32api
import matplotlib.pyplot as plt


# ============================================================
#  Win32 GDI 截屏 (整屏)
# ============================================================
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


# ============================================================
#  dxcam 截屏 (整屏, Desktop Duplication API)
# ============================================================
_dxcam_camera = None

def init_dxcam(target_fps=60):
    global _dxcam_camera
    if _dxcam_camera is None:
        try:
            import dxcam_cpp as dxcam
            _dxcam_camera = dxcam.create(output_idx=0, output_color="BGRA")
            _dxcam_camera.start(target_fps=target_fps, video_mode=True)
            time.sleep(2)
        except ImportError:
            print("dxcam_cpp 未安装，跳过该后端测试")
            return False
    return True

def dxcam_grab_screen(region=None):
    if _dxcam_camera is None:
        return None
    frame = _dxcam_camera.get_latest_frame()
    if frame is not None and region:
        left, top, x2, y2 = region
        frame = frame[top:(y2+1), left:(x2+1)]
    return frame


# ============================================================
#  WGC 截屏 (按窗口句柄, Windows Graphics Capture API)
# ============================================================
_wgc_initialized = False

def init_wgc():
    """初始化 WGC 截屏（通过 grabscreen 模块）"""
    global _wgc_initialized
    try:
        import grabscreen
        if not grabscreen._use_wgc:
            print("windows-capture 未安装，跳过 WGC 后端测试")
            return False
        grabscreen.init_camera(target_fps=30)
        _wgc_initialized = True
        return True
    except Exception as e:
        print(f"WGC 初始化失败: {e}")
        return False

def wgc_grab_screen(region=None):
    """使用 WGC 截取游戏窗口"""
    import grabscreen
    frame = grabscreen.grab_screen()
    if frame is not None and region:
        left, top, x2, y2 = region
        frame = frame[top:(y2+1), left:(x2+1)]
    return frame


# ============================================================
#  性能测试
# ============================================================
def test_performance(grab_func, num_frames=100, region=None):
    times = []
    for _ in range(num_frames):
        start_time = time.time()
        grab_func(region)
        end_time = time.time()
        times.append(end_time - start_time)
    return times


def plot_performance(results, num_frames_list):
    """绘制性能对比图"""
    plt.figure(figsize=(12, 6))
    markers = ['o', 's', '^']
    for i, (name, avg_times) in enumerate(results.items()):
        plt.plot(num_frames_list, avg_times, label=name, marker=markers[i % len(markers)])
    plt.title("截屏方案性能对比: Win32 vs DXcam vs WGC")
    plt.xlabel("测试帧数")
    plt.ylabel("平均每帧耗时 (秒)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    num_tests = [10, 50, 100, 200, 500]
    region = None  # 全屏截取

    results = {}

    # --- Win32 测试 ---
    print("=" * 50)
    print("测试 Win32 GDI 截屏...")
    win32_avg_times = []
    for num in num_tests:
        print(f"  {num} 帧...")
        times = test_performance(win32_grab_screen, num_frames=num, region=region)
        win32_avg_times.append(np.mean(times))
    results["Win32 (GDI)"] = win32_avg_times
    print(f"  平均帧耗时: {[f'{t*1000:.2f}ms' for t in win32_avg_times]}")

    # --- dxcam 测试 ---
    if init_dxcam(target_fps=60):
        print("=" * 50)
        print("测试 DXcam 截屏 (Desktop Duplication API)...")
        dxcam_avg_times = []
        for num in num_tests:
            print(f"  {num} 帧...")
            times = test_performance(dxcam_grab_screen, num_frames=num, region=region)
            dxcam_avg_times.append(np.mean(times))
        results["DXcam (DDUP)"] = dxcam_avg_times
        print(f"  平均帧耗时: {[f'{t*1000:.2f}ms' for t in dxcam_avg_times]}")

    # --- WGC 测试 ---
    if init_wgc():
        print("=" * 50)
        print("测试 WGC 截屏 (Windows Graphics Capture - 按窗口)...")
        wgc_avg_times = []
        for num in num_tests:
            print(f"  {num} 帧...")
            times = test_performance(wgc_grab_screen, num_frames=num, region=region)
            wgc_avg_times.append(np.mean(times))
        results["WGC (按窗口)"] = wgc_avg_times
        print(f"  平均帧耗时: {[f'{t*1000:.2f}ms' for t in wgc_avg_times]}")

    # --- 绘图 ---
    print("=" * 50)
    print("绘制对比图...")
    plot_performance(results, num_tests)


if __name__ == "__main__":
    main()
