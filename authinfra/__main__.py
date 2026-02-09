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
    "dataset-compile": "Compile dataset from generation JSONL(s); write manifest, train.jsonl, valid.jsonl.",
    "dataset-summary": "Print dataset summary counts (train_count, valid_count, total) from a compiled folder.",
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
        elif a == "--inputs" and i + 1 < len(args):
            out["inputs"] = args[i + 1]
            i += 2
        elif a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[a[2:].replace("-", "_")] = args[i + 1]
            i += 2
        else:
            i += 1
    return out


def _collect_text_files(dir_path: Path) -> list[Path]:
    """Recursively list .txt and .md files under dir_path."""
    out: list[Path] = []
    for ext in ("*.txt", "*.md"):
        for p in dir_path.rglob(ext):
            if p.is_file():
                out.append(p)
    return sorted(out)


def _cmd_generate(argv: list[str]) -> int:
    """Run generation. --input_file PATH or --input_dir DIR (mutually exclusive), --output_path PATH. Optional: --concurrency N."""
    args = argv[1:]
    parsed = _parse_argv(args)
    input_file = parsed.get("input_file") or parsed.get("input")
    input_dir = parsed.get("input_dir")
    output_path = parsed.get("output_path") or parsed.get("output")
    prompt_id = parsed.get("prompt_id", "1")
    model = parsed.get("model", "dry-run")
    dry_run = parsed.get("no_dry_run", "").lower() not in ("1", "true", "yes")
    min_tokens = int(parsed.get("min_tokens", "300"))
    max_tokens = int(parsed.get("max_tokens", "1000"))
    overlap = int(parsed.get("overlap", "32"))
    concurrency = None
    if parsed.get("concurrency"):
        try:
            concurrency = max(1, min(32, int(parsed["concurrency"])))
        except ValueError:
            pass

    if not output_path:
        print(json.dumps({"error": "missing --output_path or --output"}))
        return 1
    if input_file and input_dir:
        print(json.dumps({"error": "use either --input_file or --input_dir, not both"}))
        return 1
    if not input_file and not input_dir:
        print(json.dumps({"error": "missing --input_file or --input_dir (or --input for single file)"}))
        return 1

    from authinfra.generation.runner import run_generation

    out_path = Path(output_path)
    if input_dir:
        in_dir = Path(input_dir)
        if not in_dir.is_dir():
            print(json.dumps({"error": f"input_dir not a directory: {input_dir}"}))
            return 1
        out_path.mkdir(parents=True, exist_ok=True)
        files = _collect_text_files(in_dir)
        if not files:
            print(json.dumps({"error": "no .txt or .md files under input_dir", "input_dir": str(in_dir)}))
            return 1
        total_written = 0
        total_errors = 0
        aggregated_path = out_path / "aggregated.jsonl"
        with open(aggregated_path, "w", encoding="utf-8") as agg_f:
            for fp in files:
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    print(json.dumps({"file": str(fp), "error": str(e)}), file=sys.stderr)
                    total_errors += 1
                    continue
                rel = fp.relative_to(in_dir)
                out_name = rel.with_suffix(".jsonl").as_posix().replace("/", "_")
                per_file_path = out_path / out_name
                written, err_count = run_generation(
                    text,
                    prompt_id=prompt_id,
                    model_name=model,
                    output_path=per_file_path,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap,
                    dry_run=dry_run,
                    concurrency=concurrency,
                )
                total_written += written
                total_errors += err_count
                with open(per_file_path, encoding="utf-8") as pf:
                    for line in pf:
                        if line.strip():
                            agg_f.write(line)
        summary = {
            "mode": "mass",
            "input_dir": str(in_dir),
            "output_path": str(out_path),
            "files_processed": len(files),
            "lines_written": total_written,
            "error_count": total_errors,
            "aggregated": str(aggregated_path),
            "dry_run": dry_run,
        }
        print(json.dumps(summary))
        return 0 if total_errors == 0 else 1

    p = Path(input_file)
    if not p.is_file():
        print(json.dumps({"error": f"file not found: {input_file}"}))
        return 1
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 1

    if out_path.suffix.lower() == ".jsonl" or not out_path.exists() and out_path.suffix:
        out_file = out_path
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"{p.stem}.jsonl"

    written, err_count = run_generation(
        text,
        prompt_id=prompt_id,
        model_name=model,
        output_path=out_file,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap,
        dry_run=dry_run,
        concurrency=concurrency,
    )
    summary = {"output": str(out_file), "lines_written": written, "error_count": err_count, "dry_run": dry_run}
    print(json.dumps(summary))
    return 0 if err_count == 0 else 1


def _cmd_dataset_compile(argv: list[str]) -> int:
    """Compile dataset. --name NAME --output-dir DIR --inputs PATH [PATH ...] [--models id1,id2] [--prompts id1,id2] [--split-ratio 0.9] [--split-seed 0]."""
    args = argv[1:]
    parsed = _parse_argv(args)
    name = parsed.get("name", "dataset")
    output_dir = parsed.get("output_dir")
    inputs_raw = parsed.get("inputs", "")
    model_ids = parsed.get("models", "").strip().split(",") if parsed.get("models") else None
    prompt_ids = [p.strip() for p in parsed.get("prompts", "").split(",") if p.strip()] if parsed.get("prompts") else None
    split_ratio = float(parsed.get("split_ratio", "0.9"))
    split_seed = int(parsed.get("split_seed", "0"))

    if not output_dir:
        print(json.dumps({"error": "missing --output-dir"}))
        return 1
    input_paths = [p.strip() for p in inputs_raw.split() if p.strip()]
    if not input_paths and parsed.get("input"):
        input_paths = [parsed["input"]]
    if not input_paths:
        print(json.dumps({"error": "missing --inputs (space-separated paths or --inputs path1 path2)"}))
        return 1

    from authinfra.datasets.compiler import compile_dataset
    manifest = compile_dataset(
        input_paths,
        output_dir,
        dataset_name=name,
        model_ids=model_ids,
        prompt_ids=prompt_ids or None,
        split_ratio=split_ratio,
        split_seed=split_seed,
    )
    print(json.dumps({"manifest": str(Path(output_dir) / "manifest.json"), "train_count": manifest["train_count"], "valid_count": manifest["valid_count"], "filter_log": manifest["filter_log"]}))
    return 0


def _cmd_dataset_summary(argv: list[str]) -> int:
    """Print dataset summary (counts only). --dataset-dir PATH."""
    args = argv[1:]
    parsed = _parse_argv(args)
    dataset_dir = parsed.get("dataset_dir")

    if not dataset_dir:
        print(json.dumps({"error": "missing --dataset-dir"}))
        return 1
    from authinfra.datasets.compiler import dataset_summary_counts
    counts = dataset_summary_counts(dataset_dir)
    if counts is None:
        print(json.dumps({"error": "no manifest.json in dataset dir"}))
        return 1
    print(json.dumps(counts))
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
    if cmd == "detector-download":
        return _cmd_detector_download(argv)
    if cmd == "detector-infer":
        return _cmd_detector_infer(argv)
    if cmd == "generate":
        return _cmd_generate(argv)
    if cmd == "dataset-compile":
        return _cmd_dataset_compile(argv)
    if cmd == "dataset-summary":
        return _cmd_dataset_summary(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
