"""
Logger Utility Module
Setup and configure logging
"""

import logging
import sys
from pathlib import Path


def setup_logger(debug: bool = False) -> logging.Logger:
    """Setup and configure logging"""
    log_level: int = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Configure logging format
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup handlers
    handlers: list = [
        logging.FileHandler("logs/zenith.log"),
        logging.StreamHandler(sys.stdout)
    ]
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)
