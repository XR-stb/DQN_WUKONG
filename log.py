import logging
import logging.handlers
import os

# Ensure the 'logs' directory exists
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, "running.log")

# Configure logging
logger = logging.getLogger("optimized_logger")
logger.setLevel(logging.DEBUG)  # 设置为 DEBUG 以捕获所有级别的日志

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.handlers.RotatingFileHandler(
    log_file_path,
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",  # Specify UTF-8 encoding here
)

# Create formatters and add them to the handlers
formatter_console = logging.Formatter("%(message)s")
formatter_file = logging.Formatter(
    "[%(asctime)s.%(msecs)03d]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler.setFormatter(formatter_console)
file_handler.setFormatter(formatter_file)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ANSI color codes
class LogColors:
    RESET = "\033[0m"
    GREEN = "\033[92m"  # Green
    RED = "\033[91m"  # Red


class CustomLogger:
    def __init__(self, logger):
        self.logger = logger

    def debug(self, message, *args):
        formatted_message = self._format_message(message, *args)
        self.logger.debug(formatted_message)

    def info(self, message, *args):
        formatted_message = self._format_message(message, *args)
        console_message = f"{LogColors.GREEN}{formatted_message}{LogColors.RESET}"
        self.logger.info(console_message)

    def error(self, message, *args):
        formatted_message = self._format_message(message, *args)
        console_message = f"{LogColors.RED}{formatted_message}{LogColors.RESET}"
        self.logger.error(console_message)

    def _format_message(self, message, *args):
        if args:
            try:
                return message % args if "%" in message else message.format(*args)
            except Exception:
                return message
        return message


# Instantiate the custom logger
log = CustomLogger(logger)
