# logger_setup.py
import logging

def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_file = 'punching_detection.log'
    logging.root.handlers = []  # Clear existing handlers (which may write to stderr)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
    print("Logging is set up.")
    logging.info("Logging is set up.")