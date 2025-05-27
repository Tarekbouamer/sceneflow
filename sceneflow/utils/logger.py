import logging
import os
import sys

from loguru import logger
from mmengine.logging import MMLogger

# Configure Loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{function}</cyan> - <level>{message}</level>",
)

# File logging
log_file = os.environ.get("LOG_FILE", "logs/infer.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logger.add(log_file, rotation="500 KB", retention="10 days", level="INFO")

# Suppress mmocr and mmengine logs
MMLogger.get_instance("mmocr").setLevel(logging.ERROR)
MMLogger.get_instance("mmengine").setLevel(logging.ERROR)

__all__ = ["logger"]
