import os
import sys

from loguru import logger

logger.remove()

logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)

log_file = os.environ.get("LOG_FILE", "logs/infer.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logger.add(log_file, rotation="500 KB", retention="10 days", level="INFO")

__all__ = ["logger"]
