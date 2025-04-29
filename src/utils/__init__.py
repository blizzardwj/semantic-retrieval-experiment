import logging

__all__ = [
    "build_logger"
]


def build_logger(name: str = "module"):
    """
    Build a logger with a specific name.
    """
    logger = logging.getLogger(name if name else __name__)

    # add handler if not exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s = %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    
    return logger
    