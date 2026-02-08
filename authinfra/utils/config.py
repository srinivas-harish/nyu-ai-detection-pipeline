"""
Minimal configuration handling: environment variables + optional config file.

No secrets management; assume env for sensitive values. Config file is for
non-sensitive defaults (paths, feature flags, log level).
"""

import os
from pathlib import Path
from typing import Any


def _load_config_file(path: str | Path | None) -> dict[str, Any]:
    """Load a single config file. Supports JSON only for simplicity."""
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    suffix = p.suffix.lower()
    if suffix == ".json":
        try:
            import json
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_config(
    config_path: str | Path | None = None,
    env_prefix: str = "AUTHINFRA_",
) -> dict[str, Any]:
    """
    Build config from (1) config file, (2) environment variables.

    - config_path: optional path to JSON config file.
    - env_prefix: only env vars starting with this prefix are included;
      prefix is stripped and key lowercased (e.g. AUTHINFRA_LOG_LEVEL -> log_level).

    File values are overridden by env. Returned keys are lowercased.
    """
    out: dict[str, Any] = {}
    if config_path is None:
        config_path = os.environ.get("AUTHINFRA_CONFIG")
    if config_path:
        file_cfg = _load_config_file(config_path)
        for k, v in file_cfg.items():
            if isinstance(k, str):
                out[k.lower()] = v

    for key, value in os.environ.items():
        if not key.startswith(env_prefix) or not isinstance(value, str):
            continue
        sub = key[len(env_prefix):].lower()
        if sub:
            out[sub] = value

    return out
