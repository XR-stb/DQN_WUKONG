# main.py
from context import Context
import multiprocessing as mp
import time
import sys

def process(context):
    context.reopen_shared_memory()

    while True:
        try:
            frame, status = context.get_frame_and_status()
            print(f"Process: Retrieved status at {time.time()}: {status}")
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("Process: Exiting...")
            break



if __name__ == '__main__':
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
