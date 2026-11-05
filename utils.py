"""
Utility functions for loading model configurations.
"""

from pathlib import Path
from typing import Dict, Any
import yaml


def load_model_config(model_name: str, config_dir: Path = None) -> Dict[str, Any]:
    """
    Load model configuration from YAML file.

    Args:
        model_name: Name of the model (e.g., 'sd3', 'wan')
        config_dir: Directory containing config files (default: ./configs)

    Returns:
        Dictionary containing model configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if config_dir is None:
        config_dir = Path(__file__).parent / "configs"

    config_path = config_dir / f"{model_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Available models: {', '.join([p.stem for p in config_dir.glob('*.yaml')])}"
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Empty or invalid config file: {config_path}")

    return config
