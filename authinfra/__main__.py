"""
CLI entrypoint: python -m authinfra [command] ...

Lists available commands if no command given. No training, no external APIs.
"""

import sys

from authinfra.utils.config import load_config
from authinfra.utils.logging import configure_logging, get_logger


COMMANDS = {
    "config": "Print effective config (env + optional config file).",
    "version": "Print package version.",
}


def _cmd_config() -> int:
    cfg = load_config()
    # Simple print; could be JSON for machine use
    for k in sorted(cfg):
        print(f"{k}={cfg[k]!r}")
    return 0


def _cmd_version() -> int:
    from authinfra import __version__
    print(__version__)
    return 0


def main() -> int:
    cfg = load_config()
    log_level = cfg.get("log_level", "INFO").upper()
    json_logs = cfg.get("json_logs", "true").lower() in ("1", "true", "yes")
    configure_logging(level=log_level, json_logs=json_logs)

    argv = sys.argv[1:]
    if not argv:
        print("authinfra — available commands:")
        for name, desc in sorted(COMMANDS.items()):
            print(f"  {name}\t{desc}")
        return 0

    cmd = argv[0].lower()
    if cmd not in COMMANDS:
        logger = get_logger("cli")
        logger.warning("unknown command", extra={"command": cmd, "available": list(COMMANDS)})
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print("Available:", ", ".join(sorted(COMMANDS)), file=sys.stderr)
        return 1

    if cmd == "config":
        return _cmd_config()
    if cmd == "version":
        return _cmd_version()
    return 0


if __name__ == "__main__":
    sys.exit(main())
