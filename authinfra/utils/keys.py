"""
Load API keys from a local file. Keys are never logged, printed, or transmitted.
Use only for provider API calls (e.g. Gemini). Path defaults to repo_root/api_keys.md.
"""

import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_key_file(content: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            if key and val:
                out[key] = val
    return out


def load_api_keys(path: str | Path | None = None) -> dict[str, str]:
    """
    Load KEY=value pairs from a file. Default path: repo_root/api_keys.md.
    Returns empty dict if file missing or unreadable. Never log or print keys.
    """
    if path is None:
        path = os.environ.get("AUTHINFRA_API_KEYS_PATH", str(_REPO_ROOT / "api_keys.md"))
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
        return _parse_key_file(raw)
    except Exception:
        return {}
