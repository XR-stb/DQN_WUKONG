from context import Context
import multiprocessing as mp
import time
import signal
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

def signal_handler(sig, frame):
    print("Main process: Exiting...")
    sys.exit(0)

if __name__ == '__main__':
    # 创建并初始化 Context
    context = Context()
    
    # 捕获主进程的 Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # 启动子进程
    p_a = mp.Process(target=process, args=(context,))
    p_a.start()

    try:
        while True:
            context.update_status()
            print(f"Main process: Updated status at {time.time()}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Main process: Terminating child process...")
        p_a.terminate()  # 终止子进程
        p_a.join()  # 确保子进程已经终止
        print("Main process: Exiting.")
