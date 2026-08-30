# main.py

import multiprocessing as mp
import subprocess
import time
import sys
import signal
import cv2

from context import Context
from utils import change_window
import window
import grabscreen
from log import log
from process_handler import process
import os

# Event to control running state
running_event = mp.Event()

# Event to control OSD overlay process lifecycle (set=running, clear=exit)
osd_event = mp.Event()

# Event to control OSD visibility (set=visible, clear=hidden) — F1 即时切换
osd_visible_event = mp.Event()
osd_visible_event.set()  # 默认显示

# 仪表盘子进程引用
_dashboard_proc = None

def _start_dashboard():
    """启动实时训练仪表盘（独立子进程，不阻塞主程序）"""
    global _dashboard_proc
    if _dashboard_proc is not None and _dashboard_proc.poll() is None:
        log.info("📊 实时仪表盘已在运行中")
        return
    dashboard_script = os.path.join(os.path.dirname(__file__), 'train_data', 'live_dashboard.py')
    _dashboard_proc = subprocess.Popen(
        [sys.executable, dashboard_script],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    log.info("📊 实时训练仪表盘已启动 (F2 开关)")

def _stop_dashboard():
    """停止实时训练仪表盘"""
    global _dashboard_proc
    if _dashboard_proc is not None and _dashboard_proc.poll() is None:
        _dashboard_proc.terminate()
        _dashboard_proc.wait(timeout=3)
        log.info("📊 实时训练仪表盘已关闭")
    _dashboard_proc = None

def _toggle_dashboard():
    """切换仪表盘开关状态"""
    global _dashboard_proc
    if _dashboard_proc is not None and _dashboard_proc.poll() is None:
        _stop_dashboard()
    else:
        _start_dashboard()

# 记录是否已收到过一次中断信号
_interrupted_once = False

def signal_handler(sig, frame):
    global _interrupted_once
    if _interrupted_once:
        # 第二次 Ctrl+C：强制退出（防止卡死）
        log.debug("强制退出！")
        os._exit(1)
    _interrupted_once = True
    log.debug("Gracefully exiting... (再按一次 Ctrl+C 强制退出)")
    # 先清除 osd_event，防止退出期间主循环又重新启动 OSD 进程
    osd_event.clear()
    running_event.clear()

def wait_for_game_window(running_event):
    while running_event.is_set():
        frame = grabscreen.grab_screen()
        if frame is not None and window.set_windows_offset(frame):
            log.debug("Game window detected and offsets set!")
            return True
        time.sleep(1)
    return False

def main():
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize camera
    grabscreen.init_camera(target_fps=30)

    change_window.correction_window()

    if change_window.check_window_resolution_same(window.game_width, window.game_height) == False:
        raise ValueError(
            f"游戏分辨率和配置game_width({window.game_width}), game_height({window.game_height})不一致，请到window.py中修改"
        )
    
    running_event.set()

    # Wait for game window
    if not wait_for_game_window(running_event):
        log.debug("Failed to detect game window.")
        return

    # Create and initialize Context
    context = Context()

    # Start child process (传入 osd_visible_event 以便通过 F1 即时切换 OSD 显隐)
    p_brain = mp.Process(target=process, args=(context, running_event, osd_visible_event))
    p_brain.start()

    # 🚀 启动 OSD 叠加层进程（常驻运行，F1 只切换显隐，不销毁/重建进程）
    from utils.overlay_hud import run_overlay_process
    osd_event.set()
    p_osd = mp.Process(
        target=run_overlay_process,
        args=(osd_event, osd_visible_event),
        daemon=True
    )
    p_osd.start()
    log.info("OSD 叠加层已启动（常驻，F1 即时切换显隐）")

    # 🚀 自动启动实时训练仪表盘
    _start_dashboard()

    try:
        while running_event.is_set():
            context.update_status()

            # 检查 OSD 进程是否意外退出，自动重启
            if osd_event.is_set() and (p_osd is None or not p_osd.is_alive()):
                p_osd = mp.Process(
                    target=run_overlay_process,
                    args=(osd_event, osd_visible_event),
                    daemon=True
                )
                p_osd.start()
                log.info("OSD 叠加层进程意外退出，已自动重启")

    except KeyboardInterrupt:
        log.debug("Main process: 收到中断信号...")
    except Exception as e:
        log.error(f"Main process: 发生错误: {e}")
    finally:
        log.debug("Main process: 正在清理资源...")
        running_event.clear()
        osd_event.clear()
        osd_visible_event.clear()

        # 清理实时仪表盘进程
        _stop_dashboard()

        # 清理 OSD 叠加层进程（先等再杀）
        if p_osd is not None and p_osd.is_alive():
            p_osd.join(timeout=2)
            if p_osd.is_alive():
                p_osd.terminate()
                p_osd.join(timeout=1)
            if p_osd.is_alive():
                p_osd.kill()

        # 等待训练子进程正常退出，超时后强制终止
        if p_brain.is_alive():
            p_brain.join(timeout=5)
            if p_brain.is_alive():
                log.debug("Main process: 子进程未响应，强制终止...")
                p_brain.terminate()
                p_brain.join(timeout=2)
            if p_brain.is_alive():
                log.debug("Main process: 子进程仍未退出，强制 kill...")
                p_brain.kill()

        # 停止 WGC 截屏后台线程
        try:
            grabscreen.stop()
        except Exception:
            pass

        # 清理共享内存（主进程负责unlink）
        try:
            context.close()
        except Exception:
            pass

        cv2.destroyAllWindows()
        log.debug("Main process: 退出完成.")

if __name__ == '__main__':
    main()
