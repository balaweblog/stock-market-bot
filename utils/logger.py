import logging
import sys

def setup_logger():
    """
    Set up a centralized logger.
    """
    logger = logging.getLogger("stock_analyzer")
    logger.setLevel(logging.INFO)

    # Prevent adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler("analysis.log", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

log = setup_logger()
