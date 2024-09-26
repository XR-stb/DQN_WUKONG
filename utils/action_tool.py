import os
import time
from pynput.keyboard import Listener as KeyboardListener, Key
from pynput.mouse import Listener as MouseListener, Button
from actions import ActionExecutor

# Initialize the ActionExecutor
executor = ActionExecutor('./config/actions_config.yaml')

# Variable to keep track of the time of the last event
last_event_time = time.time()

# Dictionary to track the pressed state of keys
pressed_keys = set()

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
    global executor
    clear_console()
    print("Reloading configuration file...")
    executor.stop()  # Stop the current action executor
    executor = ActionExecutor('./config/actions_config.yaml')  # Reinitialize with the new configuration
    print("Configuration reloaded.")

# Function to handle action execution
def execute_action_by_num(num):
    index = num - 1  # Convert to zero-based index
    print(f"Executing action at index {index}")
    executor.take_action(index)

# Function to handle keyboard press events
def on_press(key):
    global last_event_time
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

    # Handle numeric keypad numbers 1-9 to execute actions
    if vk and 97 <= vk <= 105:
        num = vk - 96  # Map vk codes 97-105 to numbers 1-9
        execute_action_by_num(num)
        return

    # Exclude numeric keypad keys from being printed
    if vk and 96 <= vk <= 105:
        return

    # Only print if the key is not already pressed
    if key not in pressed_keys:
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
    if key in pressed_keys:
        pressed_keys.remove(key)
        event_type = 'release'
        print(f"Event: {event_type}, Key: {key}, Time: {current_time}, Time since last event: {time_diff}")

# Function to handle mouse click events
def on_click(x, y, button, pressed):
    global last_event_time
    current_time = time.time()
    time_diff = current_time - last_event_time
    last_event_time = current_time

    event_type = 'press_mouse' if pressed else 'release_mouse'
    print(f"Event: {event_type}, Button: {button}, Time: {current_time}, Time since last event: {time_diff}")

# Start the keyboard and mouse listeners
keyboard_listener = KeyboardListener(on_press=on_press, on_release=on_release)
mouse_listener = MouseListener(on_click=on_click)

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
