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
    "generate": "Run generation job: chunk text, apply prompt + model, write JSONL. Dry-run by default.",
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


def _parse_argv(args: list[str]) -> dict:
    """Parse key=value style and --key value. Boolean flags: --no-dry-run sets no_dry_run=true."""
    out = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and "=" in a:
            k, v = a.split("=", 1)
            out[k[2:].replace("-", "_")] = v
            i += 1
        elif a == "--no-dry-run":
            out["no_dry_run"] = "true"
            i += 1
        elif a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[a[2:].replace("-", "_")] = args[i + 1]
            i += 2
        else:
            i += 1
    return out


def _cmd_generate(argv: list[str]) -> int:
    """Run generation job. --input PATH --output PATH [--prompt-id ID] [--model NAME] [--no-dry-run] [--min-tokens N] [--max-tokens N] [--overlap N]."""
    args = argv[1:]
    parsed = _parse_argv(args)
    input_path = parsed.get("input")
    output_path = parsed.get("output")
    prompt_id = parsed.get("prompt_id", "1")
    model = parsed.get("model", "dry-run")
    dry_run = parsed.get("no_dry_run", "").lower() not in ("1", "true", "yes")
    min_tokens = int(parsed.get("min_tokens", "300"))
    max_tokens = int(parsed.get("max_tokens", "1000"))
    overlap = int(parsed.get("overlap", "32"))

    if not input_path or not output_path:
        print(json.dumps({"error": "missing --input or --output", "usage": "generate --input PATH --output PATH [--prompt-id ID] [--model NAME] [--no-dry-run]"}))
        return 1
    p = Path(input_path)
    if not p.is_file():
        print(json.dumps({"error": f"file not found: {input_path}"}))
        return 1
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 1

    from authinfra.generation.runner import run_generation
    written, err_count = run_generation(
        text,
        prompt_id=prompt_id,
        model_name=model,
        output_path=output_path,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap,
        dry_run=dry_run,
    )
    summary = {"output": output_path, "lines_written": written, "error_count": err_count, "dry_run": dry_run}
    print(json.dumps(summary))
    return 0 if err_count == 0 else 1


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
    if cmd == "generate":
        return _cmd_generate(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
