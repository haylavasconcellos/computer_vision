import yaml
from pathlib import Path
from typing import Dict, Any


def load_model_configs() -> Dict[str, Any]:
    config_path = Path(__file__).parent / "model_configs.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model_config(model_name: str) -> Dict[str, Any]:
    configs = load_model_configs()
    if model_name not in configs:
        raise ValueError(f"Configuration for model '{model_name}' not found")
    return configs[model_name]
