"""Configuration loader — YAML files, env var overrides, Python API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from kvfleet.config.schema import FleetConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(data: dict[str, Any], prefix: str = "KVFLEET") -> dict[str, Any]:
    """Apply environment variable overrides.

    Convention: KVFLEET__SECTION__KEY=value
    Double underscores separate nesting levels.
    Example: KVFLEET__STRATEGY=cost_first
    """
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(f"{prefix}__"):
            continue
        parts = env_key[len(prefix) + 2 :].lower().split("__")
        target = data
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        # Try to parse as number or bool
        final_key = parts[-1]
        target[final_key] = _coerce_value(env_value)
    return data


def _coerce_value(value: str) -> Any:
    """Attempt to coerce string env value to appropriate type."""
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> FleetConfig:
    """Load fleet configuration from YAML file with env var overrides.

    Args:
        path: Path to YAML config file. If None, uses KVFLEET_CONFIG env var.
        overrides: Optional dict of overrides to merge on top.

    Returns:
        Validated FleetConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    data: dict[str, Any] = {}

    # Resolve path
    if path is None:
        env_path = os.environ.get("KVFLEET_CONFIG")
        if env_path:
            path = Path(env_path)

    # Load YAML
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    # Apply env overrides
    data = _apply_env_overrides(data)

    # Apply programmatic overrides
    if overrides:
        data = _deep_merge(data, overrides)

    return FleetConfig(**data)


def save_config(config: FleetConfig, path: str | Path) -> None:
    """Save fleet configuration to a YAML file.

    Args:
        config: FleetConfig to save.
        path: Output file path.
    """
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json", exclude_defaults=True)
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
