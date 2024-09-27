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

def signal_handler(sig, frame):
    log("Gracefully exiting...")
    sys.exit(0)

def wait_for_game_window(running_event):
    while running_event.is_set():
        frame = grabscreen.grab_screen()
        if frame is not None and window.set_windows_offset(frame):
            log("Game window detected and offsets set!")
            return True
        time.sleep(1)
    return False

def process(context, running_event):
    context.reopen_shared_memory()
    emergency_queue = context.get_emergency_event_queue()
    normal_queue = context.get_normal_event_queue()

    while running_event.is_set():
        try:
            frame, status = context.get_frame_and_status()
            # Process emergency events
            while not emergency_queue.empty():
                e_event = emergency_queue.get_nowait()
                log(f"Emergency event: {e_event}")

            # Process normal events
            while not normal_queue.empty():
                n_event = normal_queue.get_nowait()
                if n_event.get('event') == 'skill_2':
                    log(f"Normal event: {n_event}")

            cv2.imshow("ROI Frame", frame)
            cv2.waitKey(30)

        except KeyboardInterrupt:
            log("Process: Exiting...")
            break
        except Exception as e:
            log(f"An error occurred in process: {e}")
            break

def main():
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize camera
    grabscreen.init_camera(target_fps=30)

    # Event to control running state
    running_event = mp.Event()
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
