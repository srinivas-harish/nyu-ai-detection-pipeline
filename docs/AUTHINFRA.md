# AuthInfra scaffold

Minimal scaffold for AI text detection infrastructure. Empirical and reversible.

## What exists

- **Package `authinfra/`** with subpackages:
  - `datasets` — placeholder
  - `generation` — placeholder
  - `detectors` — placeholder
  - `inference` — placeholder
  - `training` — stub only (no training logic)
  - `utils` — structured JSON logging and config loader (env + optional JSON config file)
- **CLI**: `python -m authinfra` prints available commands; `config` and `version` are implemented.
- **Config**: `load_config()` from env (`AUTHINFRA_*`) and optional file (`AUTHINFRA_CONFIG` or path argument).
- **Logging**: `configure_logging()`, `get_logger()`; JSON logs by default.
- **Layout**: `services/api` and `services/worker` placeholders; `docs/` and `artifacts/` for future use.

## What does NOT exist yet

- No model code.
- No training loops or training logic.
- No external API integration.
- No detector implementation.
- No dataset loading implementation.
- No inference pipeline implementation.

## Running the CLI

From repo root:

```bash
python -m authinfra
python -m authinfra version
python -m authinfra config
```

Disable JSON logs: `AUTHINFRA_JSON_LOGS=false python -m authinfra version`
