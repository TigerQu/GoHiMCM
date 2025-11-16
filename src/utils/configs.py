from yaml import safe_load
from typing import Any


# Read the configs in configs.yaml
with open('utils/configs.yaml', 'r') as f:
    CONFIGS: dict[str, Any] = safe_load(f)