from pathlib import Path
from src.utils import build_logger

logger = build_logger(__name__)

def load_config(path: Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        path: Path to the YAML configuration file.
    
    Returns:
        dict: Configuration dictionary
    """
    import yaml

    config_data = {}
    if not path.is_file():
        logger.debug(f"Config file not found at {path}, using default values")
        return config_data

    try:
        with open(path, 'r', encoding='utf-8') as f:
            loaded_yaml = yaml.safe_load(f)
            if isinstance(loaded_yaml, dict):
                config_data = loaded_yaml
            else:
                logger.warning(f"Config file {path} does not contain a dictionary root.")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file {path}: {e}")
    except Exception as e:
        # Catch other potential file reading errors
        logger.error(f"Error loading config file {path}: {e}")

    return config_data