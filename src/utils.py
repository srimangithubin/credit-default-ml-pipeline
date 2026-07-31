"""
Utility functions for the credit default ML pipeline.
"""
from pathlib import Path
import yaml

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load the YAML configuration file.
    parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    returns
    -------
    dict
        The loaded configuration as a dictionary.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_directories(config: dict) -> None:
    """
    Create directories specified in the configuration.
    parameters
    ----------
    config : dict
        The configuration dictionary containing directory paths.
    """
    Path(config['outputs']['model_dir']).mkdiir(parents=True, exist_ok=True)
    Path(config['outputs']['report_dir']).mkdir(parents=True, exist_ok=True)
    Path("reports/images").mkdir(parents=True, exist_ok=True)