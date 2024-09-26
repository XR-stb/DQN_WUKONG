import os
import time
from pynput.keyboard import Listener as KeyboardListener, Key, KeyCode
from pynput.mouse import Listener as MouseListener, Button
from actions import ActionExecutor

# Initialize the ActionExecutor
executor = ActionExecutor('./config/actions_config.yaml')

# Variable to keep track of the time of the last event
last_event_time = time.time()

# Dictionary to track the pressed state of keys
pressed_keys = set()

# Variable to track if printing is allowed
allow_printing = True

# Function to clear the console screen
def clear_console():
    # Clear console on Windows
    if os.name == 'nt':
        os.system('cls')
    # Clear console on Unix (Linux/Mac)
    else:
        os.system('clear')

# Function to reload the configuration file
def reload_config():
    global executor, allow_printing
    clear_console()
    print("Reloading configuration file...")
    executor.stop()  # Stop the current action executor
    executor = ActionExecutor('./config/actions_config.yaml')  # Reinitialize with the new configuration
    print("Configuration reloaded.")
    allow_printing = True  # Allow printing again after reloading config

# Function to handle action execution
def execute_action_by_num(num):
    if not allow_printing:
        return
    index = num - 1  # Convert to zero-based index
    print(f"Executing action at index {index}")
    def on_action_finished():
         print(f"动作执行完毕! Index: {index}")
    executor.take_action(index,action_finished_callback=on_action_finished)

# Function to handle keyboard press events
def on_press(key):
    global last_event_time, allow_printing
    current_time = time.time()
    time_diff = current_time - last_event_time
    last_event_time = current_time

    try:
        vk = key.vk
    except AttributeError:
        vk = None

    # Handle numeric keypad decimal point (.) to reload config
    if vk == 110:  # Numeric keypad decimal point
        reload_config()
        return

    # Handle numeric keypad minus (-) to stop printing
    if vk == 109:  # Numeric keypad minus
        allow_printing = False
        return

    # Handle numeric keypad numbers 1-9 to execute actions
    if vk and 97 <= vk <= 105:
        num = vk - 96  # Map vk codes 97-105 to numbers 1-9
        execute_action_by_num(num)
        return

    # Exclude numeric keypad keys from being printed
    if vk and 96 <= vk <= 105:
        return

    # Only print if the key is not already pressed and printing is allowed
    if key not in pressed_keys and allow_printing:
        pressed_keys.add(key)
        event_type = 'press'
        print(f"Event: {event_type}, Key: {key}, Time: {current_time}, Time since last event: {time_diff}")

# Function to handle keyboard release events
def on_release(key):
    global last_event_time
    current_time = time.time()
    time_diff = current_time - last_event_time
    last_event_time = current_time

    try:
        vk = key.vk
    except AttributeError:
        vk = None

    if vk == 110:  # Numeric keypad decimal point
        return

    # Exclude numeric keypad keys from being printed
    if vk and 96 <= vk <= 105:
        return

    # If the key is pressed, print release and remove from pressed_keys
    if key in pressed_keys and allow_printing:
        pressed_keys.remove(key)
        event_type = 'release'
        print(f"Event: {event_type}, Key: {key}, Time: {current_time}, Time since last event: {time_diff}")

# Function to handle mouse click events
def on_click(x, y, button, pressed):
    global last_event_time, allow_printing
    current_time = time.time()
    time_diff = current_time - last_event_time
    last_event_time = current_time

    event_type = 'press_mouse' if pressed else 'release_mouse'
    
    if allow_printing:
        print(f"Event: {event_type}, Button: {button}, Time: {current_time}, Time since last event: {time_diff}")

# Start the keyboard and mouse listeners
keyboard_listener = KeyboardListener(on_press=on_press, on_release=on_release)
mouse_listener = MouseListener(on_click=on_click)

print("开始前有 3 秒倒计时，请切换到目标窗口...")
for i in range(3, 0, -1):
    print(f"{i}秒后开始")
    time.sleep(1)

print("start listener!")

keyboard_listener.start()
mouse_listener.start()

# Keep the main thread running
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    keyboard_listener.stop()
    mouse_listener.stop()
    executor.stop()
