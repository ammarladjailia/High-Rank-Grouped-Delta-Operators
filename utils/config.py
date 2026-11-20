from copy import deepcopy
from pathlib import Path

import yaml


def deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def load_config(path: str | Path) -> dict:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    base_name = config.pop("base_config", None)

    if base_name is None:
        return config

    base_config = load_config(path.parent / base_name)
    return deep_update(base_config, config)
