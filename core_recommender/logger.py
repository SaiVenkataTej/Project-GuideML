"""
Centralized Logging Configuration for GuideML

Rationale:
----------
- **Structured Logging**: Consistent format across all modules (timestamp, level, module, message).
- **Multiple Handlers**: Console (for development) + File (for production debugging).
- **Level Control**: Easy switch between DEBUG (verbose) and INFO (production) modes.
- **Performance**: Avoids print() overhead and provides filtering capabilities.

Usage:
------
from core_recommender.logger import get_logger

logger = get_logger(__name__)
logger.info("Model training started")
logger.warning("High correlation detected")
logger.error("Model failed to converge", exc_info=True)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# =========================================================================
# Configuration
# =========================================================================

LOG_DIR = Path(__file__).parent.parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

DEFAULT_LOG_FILE = LOG_DIR / 'guideml.log'
ERROR_LOG_FILE = LOG_DIR / 'error.log'

# Log format with timestamp, level, module, and message
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Default level
DEFAULT_LEVEL = logging.INFO

# =========================================================================
# Logger Factory
# =========================================================================

def get_logger(name: str, level: Optional[int] = None, log_to_file: bool = True) -> logging.Logger:
    """
    Creates or retrieves a logger with standardized configuration.
    
    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to INFO.
        log_to_file: Whether to write logs to file in addition to console.
                     Defaults to True.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Set level
    logger.setLevel(level or DEFAULT_LEVEL)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # Console Handler (stdout for INFO+, stderr for WARNING+)
    # FIX: Force utf-8 encoding to prevent Windows crash on emojis
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for older python or non-standard environments
        pass
        
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handlers (if enabled)
    if log_to_file:
        # Main log file (all levels)
        file_handler = logging.FileHandler(DEFAULT_LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Error-only log file (WARNING+)
        error_handler = logging.FileHandler(ERROR_LOG_FILE, mode='a', encoding='utf-8')
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    
    # Prevent propagation to root logger (avoid duplicate logs)
    logger.propagate = False
    
    return logger


def set_log_level(level: int):
    """
    Updates the logging level globally for all GuideML loggers.
    
    Args:
        level: New logging level (logging.DEBUG, logging.INFO, etc.)
    """
    root_logger = logging.getLogger('core_recommender')
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


# =========================================================================
# Progress Logger (Special Handler for UI Updates)
# =========================================================================

class ProgressLogHandler(logging.Handler):
    """
    Custom log handler that forwards progress updates to a callback.
    
    Usage for Flask/UI integration:
    
    from core_recommender.logger import ProgressLogHandler
    
    progress_handler = ProgressLogHandler(callback=my_progress_function)
    logger.addHandler(progress_handler)
    """
    
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.current_percent = 0
    
    def emit(self, record):
        """
        Called when a log record is emitted.
        Checks for special progress markers and calls callback.
        """
        if self.callback and hasattr(record, 'progress_percent'):
            self.callback(record.progress_percent, record.getMessage())


def log_progress(logger: logging.Logger, percent: int, message: str):
    """
    Helper function to log progress with percentage.
    
    Args:
        logger: Logger instance.
        percent: Progress percentage (0-100).
        message: Progress message.
    """
    # Create log record with custom attribute
    record = logger.makeRecord(
        logger.name, logging.INFO, '', 0, message, (), None
    )
    record.progress_percent = percent
    logger.handle(record)
    
    # Also log normally for file/console
    logger.info(f"[{percent}%] {message}")
