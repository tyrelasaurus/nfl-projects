from typing import Dict, Any, List


def find_config_file(possible_paths: List[str]) -> str:
    import os
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No configuration file found. Searched: {possible_paths}")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for key, value in (override or {}).items():
        if key == 'environments':
            continue
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def validate_logging_level(level: str) -> bool:
    valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    return str(level).upper() in valid_levels

