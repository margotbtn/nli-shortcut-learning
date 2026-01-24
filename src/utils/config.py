from __future__ import annotations

from pathlib import Path
import yaml


def load_yaml_config(
        config_path: Path | str,
        overrides: None | dict[tuple[str] | str, object],
        ) -> dict:
    """Loads a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        overrides: Variables to override in the configuration.
                   The dict key can be directly the key in the config
                   or the tuple of keys leading to the variable.
    
    Returns:
        A dictionary containing the configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Override variables if provided
    if overrides:
        for key_path, value in overrides.items():
            if isinstance(key_path, tuple):
                cursor = config
                for key in key_path[:-1]:
                    cursor = cursor[key]
                cursor[key_path[-1]] = value
            else:
                config[key_path] = value
    
    return config
