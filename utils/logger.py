import logging
import sys
import os
from pathlib import Path

LOG_FILENAME = "training.log"
LOG_PATH = Path(os.getcwd()) / LOG_FILENAME 

# Ensure log file exists in the execution directory, if not create one
if not LOG_PATH.exists():
    LOG_PATH.touch()

with LOG_PATH.open("a", encoding="utf-8") as f:
    f.write("\n")

# Formatting
with_time = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
no_time = logging.Formatter('[%(levelname)s] %(message)s')


######################
# Log categories:
######################

# Console only logs
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(with_time)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(no_time)

# Logger for console only
console_logger = logging.getLogger("console_only_logger")
console_logger.setLevel(logging.INFO)
console_logger.addHandler(stderr_handler)
console_logger.propagate = False

# File + console logs 
file_handler = logging.FileHandler(LOG_PATH, encoding='utf-8')
file_handler.setFormatter(with_time)

# Logger for file + console
file_logger = logging.getLogger("file_logger")
file_logger.setLevel(logging.INFO)
file_logger.addHandler(stdout_handler)
file_logger.addHandler(file_handler)
file_logger.propagate = False

# Public definitions
def log_info(msg, *args):
    """Info with timestamp, saved to file."""
    file_logger.info(msg, *args)

def log_info_console(msg, *args):
    """Info without timestamp, not saved to file."""
    console_logger.info(msg, *args)

def log_warning(msg, *args):
    """Warning with timestamp, saved to file."""
    file_logger.warning(msg, *args)

def log_error(msg, *args):
    """Error with timestamp, saved to file."""
    file_logger.error(msg, *args)

def log_training(msg, *args):
    """Alias for info-level training logs."""
    log_info(msg, *args)

