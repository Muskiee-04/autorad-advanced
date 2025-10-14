"""
Logging configuration for AutoRad
"""
import logging
import sys

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger