"""
CLI entrypoint: python -m authinfra [command] ...

Lists available commands if no command given. No training, no external APIs.
"""

import json
import sys
from pathlib import Path

from authinfra.utils.config import load_config
from authinfra.utils.logging import configure_logging, get_logger


COMMANDS = {
    "config": "Print effective config (env + optional config file).",
    "version": "Print package version.",
    "detector-download": "Download the baseline HF detector model (Hello-SimpleAI/chatgpt-detector-roberta).",
    "detector-infer": "Run baseline detector on a local text file; print JSON result to stdout.",
}


def _cmd_config() -> int:
    cfg = load_config()
    for k in sorted(cfg):
        print(f"{k}={cfg[k]!r}")
    return 0


def _cmd_version() -> int:
    from authinfra import __version__
    print(__version__)
    return 0


def _cmd_detector_download(argv: list[str]) -> int:
    """Download baseline model. Optional: --cache-dir DIR."""
    cache_dir = None
    args = argv[1:]
    for i, a in enumerate(args):
        if a == "--cache-dir" and i + 1 < len(args):
            cache_dir = args[i + 1]
            break
    from authinfra.detectors.baseline import ensure_downloaded, get_model_id
    ok = ensure_downloaded(cache_dir=cache_dir)
    out = {"model": get_model_id(), "downloaded": ok, "error": None if ok else "download or load failed"}
    print(json.dumps(out))
    return 0 if ok else 1


def _cmd_detector_infer(argv: list[str]) -> int:
    """Run inference on text file. --input PATH [--cache-dir DIR]."""
    args = argv[1:]
    input_path = None
    cache_dir = None
    i = 0
    while i < len(args):
        if args[i] == "--input" and i + 1 < len(args):
            input_path = args[i + 1]
            i += 2
            continue
        if args[i] == "--cache-dir" and i + 1 < len(args):
            cache_dir = args[i + 1]
            i += 2
            continue
        i += 1
    if not input_path:
        print(json.dumps({"error": "missing --input PATH", "model": "Hello-SimpleAI/chatgpt-detector-roberta"}))
        return 1
    p = Path(input_path)
    if not p.is_file():
        print(json.dumps({"error": f"file not found: {input_path}", "model": "Hello-SimpleAI/chatgpt-detector-roberta"}))
        return 1
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(json.dumps({"error": str(e), "model": "Hello-SimpleAI/chatgpt-detector-roberta"}))
        return 1
    from authinfra.detectors.baseline import run_inference
    result = run_inference(text, cache_dir=cache_dir)
    print(json.dumps(result))
    return 0 if result.get("error") is None else 1


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
    if cmd == "detector-download":
        return _cmd_detector_download(argv)
    if cmd == "detector-infer":
        return _cmd_detector_infer(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
