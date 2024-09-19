import logging
import logging.handlers
import os

# Ensure the 'logs' directory exists
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, "running.log")

# Configure logging
logger = logging.getLogger("optimized_logger")
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.handlers.RotatingFileHandler(
    log_file_path,
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding='utf-8'  # Specify UTF-8 encoding here
)


# Create formatters and add them to the handlers
formatter_console = logging.Formatter(
    '%(message)s'  # 只显示消息，不包括时间
)
formatter_file = logging.Formatter(
    '[%(asctime)s.%(msecs)03d]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

console_handler.setFormatter(formatter_console)
file_handler.setFormatter(formatter_file)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def log(message, *args):
    if args:
        try:
            formatted_message = message % args if '%' in message else message.format(*args)
        except Exception:
            formatted_message = message
    else:
        formatted_message = message

    logger.info(formatted_message)
