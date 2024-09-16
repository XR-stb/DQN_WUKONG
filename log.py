import os
import inspect
import datetime


def log(message, *args):
    """
    Logs a formatted message to both the console and a file with timestamp and code line information.

    Args:
        message (str): The message format string.
        *args: Values to format into the message string.
    """
    # Get current time and code line information
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
        :-3
    ]  # Get time with milliseconds
    frame_info = inspect.stack()[1]  # Get the caller's frame information
    code_line = frame_info.lineno
    file_name = frame_info.filename.split("/")[-1]

    # Determine the format type and format the message
    if "%" in message:
        try:
            formatted_message = message % args
        except TypeError:
            formatted_message = message  # Use the raw message if formatting fails
    else:
        try:
            formatted_message = message.format(*args)
        except IndexError:
            formatted_message = message  # Use the raw message if formatting fails

    # Formatted log message for file
    log_message = (
        f"[{current_time}] {file_name} (Line {code_line}): {formatted_message}"
    )

    # Print the formatted message to the console
    print(formatted_message)

    # Ensure the 'logs' directory exists, if not create it
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Append the formatted message to a log file
    log_file_path = os.path.join(log_directory, "running.log")
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(log_message + "\n")
