from yaml import safe_load
from typing import Any
import os


# Read the configs in configs.yaml
config_path = os.path.join(os.path.dirname(__file__), 'configs.yaml')
with open(config_path, 'r') as f:
    CONFIGS: dict[str, Any] = safe_load(f)