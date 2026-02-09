# API keys (paste your own in `api_keys.md` — that file is gitignored)

Copy this file to `api_keys.md` and replace the placeholders. Do not commit `api_keys.md`.

```bash
cp api_keys.example.md api_keys.md
```

## Format

Paste one key per line. Scripts that need keys (e.g. some data_helpers) may read from `data_helpers/api_keys.txt` in `KEY=value` form. Use this file as a personal, gitignored stash and copy into that file or env as needed.

Example entries:

```
GEMINI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
CLAUDE_API_KEY=your_key_here
GROK_API_KEY=your_key_here
```
