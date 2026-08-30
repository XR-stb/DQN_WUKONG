# grabscreen.py
# 使用 Windows Graphics Capture (WGC) API 按游戏窗口句柄截屏
# 优势：只截取游戏窗口本身画面，OSD 叠加层、其他窗口均不会被截入
# 依赖：pip install windows-capture
# 降级：若 windows-capture 未安装，自动回退到 dxcam_cpp（整屏截取）

import threading
import time
import atexit
import ctypes
import ctypes.wintypes
import numpy as np
from log import log

# ============================================================
#  游戏窗口标题（需与 change_window.py 中的 WUKONG_TITLE 一致）
# ============================================================
_GAME_WINDOW_TITLE = "b1  "

# ============================================================
#  全局状态
# ============================================================
_latest_frame = None          # 最新一帧画面 (numpy array, BGRA)
_frame_lock = threading.Lock()
_capture_thread = None
_initialized = False
_use_wgc = False              # 标记当前使用的后端
_stopping = False             # 标记是否正在停止，防止 stop 后帧回调继续写入

# WGC 客户区裁剪参数（WGC 按整个窗口捕获，含标题栏和边框，需裁剪到客户区）
_wgc_crop_top = 0             # 标题栏高度
_wgc_crop_left = 0            # 左边框宽度
_wgc_crop_bottom = 0          # 下边框高度（裁剪后的底边界）
_wgc_crop_right = 0           # 右边框宽度（裁剪后的右边界）
_wgc_need_crop = False        # 是否需要裁剪

# --- 尝试导入 windows-capture ---
try:
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    _use_wgc = True
    log.debug("截屏后端: Windows Graphics Capture (WGC) - 按窗口句柄捕获")
except ImportError:
    _use_wgc = False
    log.debug("截屏后端: dxcam_cpp (降级方案 - 整屏捕获)")

# ============================================================
#  WGC 后端实现
# ============================================================
_capture_control_ref = None  # 保存 capture_control 引用，用于停止


def _wgc_on_frame_arrived(frame: "Frame", capture_control: "InternalCaptureControl"):
    """WGC 帧到达回调：将最新帧缓存到全局变量"""
    global _latest_frame, _capture_control_ref
    if _stopping:
        return
    _capture_control_ref = capture_control
    try:
        # frame.frame_buffer 是 numpy.ndarray (H, W, 4)，默认 BGRA 格式
        frame_np = frame.frame_buffer
        with _frame_lock:
            _latest_frame = frame_np
    except Exception as e:
        if not _stopping:
            log.error(f"WGC 帧处理异常: {e}")


def _wgc_on_closed():
    """WGC 捕获会话关闭回调"""
    log.debug("WGC 捕获会话已关闭")


def _start_wgc_capture():
    """在后台线程中启动 WGC 捕获（capture.start() 是阻塞的）"""
    try:
        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            window_name=_GAME_WINDOW_TITLE,
        )

        @capture.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
            _wgc_on_frame_arrived(frame, capture_control)

        @capture.event
        def on_closed():
            _wgc_on_closed()

        log.debug(f"WGC 开始捕获窗口: '{_GAME_WINDOW_TITLE}'")
        capture.start()  # 阻塞，在线程中运行
    except Exception as e:
        log.error(f"WGC 启动失败: {e}")


# ============================================================
#  WGC 客户区裁剪计算
# ============================================================

def _calc_wgc_crop_params(frame_shape):
    """根据 WGC 首帧尺寸和游戏窗口客户区尺寸，计算裁剪参数
    
    WGC 按整个窗口（含标题栏 + 边框）捕获，帧尺寸通常大于客户区。
    例如客户区 1280×720，WGC 帧可能是 1282×752（多出标题栏 32px、边框各 1px）。
    此函数通过 Win32 API 精确计算偏移量。
    """
    global _wgc_crop_top, _wgc_crop_left, _wgc_crop_bottom, _wgc_crop_right, _wgc_need_crop

    import win32gui
    
    frame_h, frame_w = frame_shape[0], frame_shape[1]

    # 通过窗口标题获取句柄
    hwnd = win32gui.FindWindow(None, _GAME_WINDOW_TITLE)
    if not hwnd:
        log.debug("WGC 裁剪: 未找到游戏窗口，尝试通过类名查找")
        hwnd = win32gui.FindWindow("UnrealWindow", None)

    if hwnd:
        # 获取窗口矩形（含标题栏和边框）和客户区矩形
        window_rect = win32gui.GetWindowRect(hwnd)  # (left, top, right, bottom)
        
        # 获取客户区左上角在屏幕上的位置
        pt = ctypes.wintypes.POINT(0, 0)
        ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
        
        # 客户区矩形
        client_rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(client_rect))
        client_w = client_rect.right
        client_h = client_rect.bottom

        # 计算标题栏高度和边框宽度
        # 客户区左上角在屏幕中的位置 vs 窗口左上角在屏幕中的位置
        border_left = pt.x - window_rect[0]
        border_top = pt.y - window_rect[1]

        _wgc_crop_top = border_top
        _wgc_crop_left = border_left
        _wgc_crop_bottom = border_top + client_h
        _wgc_crop_right = border_left + client_w
        
        # 验证裁剪范围不超出帧边界
        _wgc_crop_bottom = min(_wgc_crop_bottom, frame_h)
        _wgc_crop_right = min(_wgc_crop_right, frame_w)
        
        _wgc_need_crop = (_wgc_crop_top > 0 or _wgc_crop_left > 0 
                          or _wgc_crop_bottom < frame_h or _wgc_crop_right < frame_w)

        actual_w = _wgc_crop_right - _wgc_crop_left
        actual_h = _wgc_crop_bottom - _wgc_crop_top
        log.debug(f"WGC 帧裁剪: 原始={frame_w}x{frame_h}, 客户区={client_w}x{client_h}, "
                  f"裁剪区域=({_wgc_crop_left},{_wgc_crop_top})-({_wgc_crop_right},{_wgc_crop_bottom}), "
                  f"裁剪后={actual_w}x{actual_h}")
        
        if actual_w != client_w or actual_h != client_h:
            log.debug(f"⚠ WGC 裁剪后尺寸({actual_w}x{actual_h})与客户区({client_w}x{client_h})不一致！")
    else:
        # 如果找不到窗口句柄，使用启发式方法（帧尺寸 - 配置的游戏分辨率）
        import yaml
        try:
            with open('./config/game_conf.yaml', 'r', encoding='utf-8') as f:
                gc = yaml.safe_load(f)
            game_w = gc['game_window']['width']
            game_h = gc['game_window']['height']
            
            extra_h = frame_h - game_h
            extra_w = frame_w - game_w
            
            if extra_h > 0 or extra_w > 0:
                _wgc_crop_top = extra_h  # 标题栏在顶部
                _wgc_crop_left = extra_w // 2  # 边框左右均分
                _wgc_crop_bottom = frame_h
                _wgc_crop_right = frame_w - (extra_w - extra_w // 2)
                _wgc_need_crop = True
                log.debug(f"WGC 帧裁剪(启发式): 原始={frame_w}x{frame_h}, 目标={game_w}x{game_h}, "
                          f"裁剪区域=({_wgc_crop_left},{_wgc_crop_top})-({_wgc_crop_right},{_wgc_crop_bottom})")
            else:
                log.debug(f"WGC 帧尺寸({frame_w}x{frame_h})不大于游戏分辨率({game_w}x{game_h})，无需裁剪")
        except Exception as e:
            log.error(f"WGC 裁剪参数计算失败: {e}，将使用原始帧")


# ============================================================
#  dxcam 降级后端
# ============================================================
_dxcam_camera = None


def _init_dxcam(target_fps):
    """降级方案：初始化 dxcam_cpp"""
    global _dxcam_camera
    if _dxcam_camera is None:
        import dxcam_cpp as dxcam
        log.debug("Initializing dxcam_cpp Screen Capture Camera.")
        _dxcam_camera = dxcam.create(output_idx=0, output_color="BGRA")
        _dxcam_camera.start(target_fps=target_fps, video_mode=True)
        atexit.register(_dxcam_camera.stop)
        time.sleep(2)


# ============================================================
#  对外统一接口（与原有接口完全兼容）
# ============================================================

def init_camera(target_fps=30):
    """初始化截屏引擎
    
    - WGC 模式：启动后台线程按窗口句柄捕获（target_fps 参数不适用，WGC 自适应帧率）
    - dxcam 模式：使用 Desktop Duplication API 整屏捕获
    """
    global _initialized, _capture_thread

    if _initialized:
        return

    if _use_wgc:
        _capture_thread = threading.Thread(target=_start_wgc_capture, daemon=True)
        _capture_thread.start()
        # 等待首帧到达
        timeout = 10.0
        start = time.time()
        while time.time() - start < timeout:
            with _frame_lock:
                if _latest_frame is not None:
                    break
            time.sleep(0.1)

        if _latest_frame is None:
            log.error(f"WGC 在 {timeout}s 内未收到任何帧，请确认游戏窗口 '{_GAME_WINDOW_TITLE}' 已打开")
        else:
            log.debug(f"WGC 首帧已就绪，帧大小: {_latest_frame.shape}")
            _calc_wgc_crop_params(_latest_frame.shape)
    else:
        _init_dxcam(target_fps)

    _initialized = True


def grab_screen():
    """获取最新一帧游戏画面
    
    返回: numpy array (H, W, 4)，BGRA 格式；若无帧返回 None
    注意: WGC 模式下会自动裁剪掉标题栏和边框，只返回客户区画面
    """
    if _use_wgc:
        with _frame_lock:
            if _latest_frame is None:
                log.debug("WGC: 暂无可用帧")
                return None
            frame = _latest_frame.copy()
        # 裁剪到客户区（去除标题栏和边框）
        if _wgc_need_crop:
            frame = frame[_wgc_crop_top:_wgc_crop_bottom, _wgc_crop_left:_wgc_crop_right]
        return frame
    else:
        if _dxcam_camera is None:
            log.debug("dxcam: 相机未初始化")
            return None
        frame = _dxcam_camera.get_latest_frame()
        if frame is None:
            log.debug("dxcam: No frame received.")
        return frame


def stop():
    """停止截屏引擎，确保后台线程能被正确终止"""
    global _capture_control_ref, _initialized, _stopping

    if not _initialized:
        return

    _stopping = True

    if _use_wgc:
        # 1. 通过 capture_control 通知 WGC 停止
        if _capture_control_ref is not None:
            try:
                _capture_control_ref.stop()
                log.debug("WGC 捕获已停止")
            except Exception:
                pass
            _capture_control_ref = None

        # 2. 等待后台捕获线程退出（最多2秒）
        if _capture_thread is not None and _capture_thread.is_alive():
            _capture_thread.join(timeout=2)
            if _capture_thread.is_alive():
                log.debug("WGC 后台线程未在超时内退出（daemon线程，进程退出时会自动回收）")
    elif _dxcam_camera is not None:
        try:
            _dxcam_camera.stop()
            log.debug("dxcam 捕获已停止")
        except Exception:
            pass

    _initialized = False


# 注册退出时自动清理
atexit.register(stop)
