import logging
import os
import datetime
import time
from functools import wraps
import logging.config
import sys

# Create logs folder if it doesn't exist
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)  # Better way to create folder

# Get the base module name of the running script (e.g., main.py → "main")
if hasattr(sys.modules["__main__"], "__file__"):
    module_name = os.path.splitext(os.path.basename(sys.modules["__main__"].__file__))[
        0
    ]
else:
    module_name = "interactive"  # fallback for environments like notebooks or REPL

# Generate a new log file path with module name + timestamp
log_file_path = os.path.join(
    log_folder,
    f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{module_name}.log",
)

# Configure logging with dictionary config
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            # 'format': '%(name)s:%(module)s:%(funcName)s:%(levelname)s:%(asctime)s:%(lineno)d:%(message)s'
            "format": "%(module)s:%(funcName)s:%(levelname)s:%(lineno)d:%(message)s"
        },
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": log_file_path,
            "formatter": "standard",
            "encoding": "utf-8",
            "mode": "a+",
        },
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console"],  # Only console by default
            "level": "WARNING",
        },
        "app_logger": {  # Your main logger that you'll use in other files
            "level": "DEBUG",
            "handlers": ["file", "console"],
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

# Create a logger that will be imported by other modules
logger = logging.getLogger("app_logger")
logger.info(f"Logging initialized. Log file: {log_file_path}")


def log_time_taken_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_taken = end_time - start_time

        # Log the time taken using the logger
        logger.info(
            f"Function '{func.__name__}' took {time_taken:.4f} seconds to complete"
        )
        return result

    return wrapper


def log_time_taken_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        time_taken = end_time - start_time

        # Log the time taken using the logger
        logger.info(
            f"Function '{func.__name__}' took {time_taken:.4f} seconds to complete"
        )
        return result

    return wrapper


# Original decorator function to log time taken by functions
def log_time_taken(func):
    @wraps(func)  # Add wraps to preserve function metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_taken = end_time - start_time

        # Log the time taken using the logger
        logger.info(
            f"Function '{func.__name__}' took {time_taken:.4f} seconds to complete"
        )
        return result

    return wrapper


# Explicitly silence the noisy AWS SDK loggers
for logger_name in ["botocore", "boto3", "urllib3", "hooks", "regions", "endpoint"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Log the application start
logger.info("Application has been started")