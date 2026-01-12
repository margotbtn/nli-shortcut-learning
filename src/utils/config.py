from __future__ import annotations

import yaml


def load_yaml_config(config_path: str = 'src/configs/base.yaml') -> dict:
    """Loads a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        A dictionary containing the configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
