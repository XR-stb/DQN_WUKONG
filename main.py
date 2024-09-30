# main.py

import multiprocessing as mp
import time
import sys
import signal
import cv2

from context import Context
import window
import grabscreen
from log import log
from process_handler import process


# Event to control running state
running_event = mp.Event()


def signal_handler(sig, frame):
    log("Gracefully exiting...")
    running_event.clear()
    sys.exit(0)

def wait_for_game_window(running_event):
    while running_event.is_set():
        frame = grabscreen.grab_screen()
        if frame is not None and window.set_windows_offset(frame):
            log("Game window detected and offsets set!")
            return True
        time.sleep(1)
    return False

def main():
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize camera
    grabscreen.init_camera(target_fps=30)


    running_event.set()

    # Wait for game window
    if not wait_for_game_window(running_event):
        log("Failed to detect game window.")
        return

    # Create and initialize Context
    context = Context()

    # Start child process
    p_brain = mp.Process(target=process, args=(context, running_event))
    p_brain.start()

    try:
        while running_event.is_set():
            context.update_status()
    except KeyboardInterrupt:
        log("Main process: Terminating child process...")
        running_event.clear()
        p_brain.terminate()
        p_brain.join()
        log("Main process: Exiting.")
    except Exception as e:
        log(f"An error occurred in main: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
