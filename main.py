# main.py
from context import Context
import multiprocessing as mp
import time
import sys
import window
import grabscreen
import signal
import cv2


# 标志位，表示是否继续运行
running = True

# 处理 Ctrl+C 的函数
def signal_handler(sig, frame):
    global running
    print("\nGracefully exiting...")
    running = False
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)


def wait_for_game_window():
    while running:
        frame = grabscreen.grab_screen()

        if frame is not None:
            if window.set_windows_offset(frame):
                print("Game window detected and offsets set!")                
                return True

            print("Failed to find the game logo, offsets not set.")

        time.sleep(0.1)


def process(context):
    context.reopen_shared_memory()

    while True:
        try:
            frame, status = context.get_frame_and_status()
            #print(f"Process: Retrieved status at {time.time()}: {status}")
            print('self_blood',status['self_blood'],'boss_blood',status['boss_blood'])

            #e_event = context.get_emergency_event_queue().get()
            #print(e_event)
            #n_event = context.get_normal_event_queue().get()
            #print(n_event)



            cv2.imshow("roi frame", frame)
            cv2.waitKey(30)

        except KeyboardInterrupt:
            print("Process: Exiting...")
            break



if __name__ == '__main__':

    # init camera 
    grabscreen.init_camera(target_fps=30)

    wait_for_game_window()

    # 创建并初始化 Context
    context = Context()

    # 启动子进程
    p_brain = mp.Process(target=process, args=(context,))
    p_brain.start()

    try:
        while True:
            context.update_status()
    except KeyboardInterrupt:
        print("Main process: Terminating child process...")
        p_brain.terminate()  # 终止子进程
        p_brain.join()  # 确保子进程已经终止
        print("Main process: Exiting.")
