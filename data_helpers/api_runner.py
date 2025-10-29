"""Tiny CLI for Gemini / DeepSeek / ChatGPT / Claude / Grok. Importable too.

You must create an `api_keys.txt` under data_helpers/ with the following: 
GEMINI_API_KEY=<YOUR_KEY>
DEEPSEEK_API_KEY=<YOUR_KEY>
OPENAI_API_KEY=<YOUR_KEY>
CLAUDE_API_KEY=<YOUR_KEY>
GROK_API_KEY=<YOUR_KEY>


"""

import argparse
import json
import time
import os

import requests
 
GEMINI_SUPPORTED = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-image",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

DEFAULT_MODELS = {
    "chatgpt": "gpt-4o-mini",
    "claude": "claude-sonnet-4-5",
    "grok": "grok-2-latest",
    "deepseek": "deepseek-chat",
}

KEY_FILE_DEFAULT = os.path.join(os.path.dirname(__file__), "api_keys.txt")


def _load_keys(path: str | None) -> dict:
    """Load KEY=VALUE lines from a text file."""
    p = path or KEY_FILE_DEFAULT
    if not os.path.isfile(p):
        raise FileNotFoundError(f"api key file not found: {p}")
    out = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _extract_gemini_text(payload: dict) -> str:
    """Get text from Gemini JSON."""
    try:
        candidates = payload.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            return ""
        text = parts[0].get("text")
        return text if isinstance(text, str) else ""
    except Exception:
        return ""


def _extract_deepseek_text(payload: dict) -> str:
    """Get text from DeepSeek JSON."""
    try:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) else ""
    except Exception:
        return ""


def _call_gemini(prompt: str, max_output: int, api_key: str, model_name: str, timeout: int = 60) -> str:
    """Call Gemini."""
    model_name = model_name or "gemini-1.5-flash"
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": str(prompt)}]}],
        "generationConfig": {"maxOutputTokens": 4096 ,"temperature": 0.3,
            "responseMimeType": "text/plain" },
    }
    headers = {"Content-Type": "application/json; charset=utf-8"}
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        print(f"resp is {resp.json()}")
        resp.raise_for_status()
        data = resp.json()
        return _extract_gemini_text(data)
    except requests.exceptions.RequestException as e:
        return f"[network-error] {e}"
    except ValueError:
        return "[parse-error] Non-JSON response from Gemini"


def _call_deepseek(prompt: str, max_output: int, api_key: str, model_name: str = "deepseek-chat", timeout: int = 60) -> str:
    """Call DeepSeek."""
    url = "https://api.deepseek.com/chat/completions"
    payload = {
        "model": model_name or "deepseek-chat",
        "messages": [{"role": "user", "content": str(prompt)}],
        "max_tokens": int(max_output),
    }
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return _extract_deepseek_text(data)
    except requests.exceptions.RequestException as e:
        return f"[network-error] {e}"
    except ValueError:
        return "[parse-error] Non-JSON response from DeepSeek"


def run_api(model: str, prompt: str, max_output: int = 1024, api_key: str = "") -> str:
    model = (model or "").strip().lower()
    keys = _load_keys(keys_path)
    # pick provider
    provider = (
        "gemini" if model.startswith("gemini") else
        "deepseek" if model.startswith("deepseek") else
        "chatgpt" if model.startswith("chatgpt") or model.startswith("openai") else
        "claude" if model.startswith("claude") else
        "grok" if model.startswith("grok") or model.startswith("xai") else
        ""
    )
    if not provider:
        raise ValueError("--model must start with gemini, deepseek, chatgpt, claude, or grok")

    # get api key
    key_map = {
        "gemini": ["GEMINI_API_KEY"],
        "deepseek": ["DEEPSEEK_API_KEY"],
        "chatgpt": ["OPENAI_API_KEY", "CHATGPT_API_KEY"],
        "claude": ["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"],
        "grok": ["GROK_API_KEY", "XAI_API_KEY"],
    }
    api_key = None
    for k in key_map[provider]:
        if k in keys and keys[k]:
            api_key = keys[k]
            break
    if not api_key:
        raise ValueError(f"missing api key for provider '{provider}' in {keys_path or KEY_FILE_DEFAULT}")

    start = time.time()
    if model.startswith("gemini"):
        if model not in GEMINI_SUPPORTED:
            raise ValueError(
                "unsupported gemini model. choose one of: " + ", ".join(GEMINI_SUPPORTED)
            )
        out = _call_gemini(prompt, max_output, api_key, model)
        print(f"out is {out}")
        elapsed = time.time() - start
        if not quiet:
            print(f"API: gemini | Elapsed: {elapsed:.2f}s")
            print("Output:\n" + (out or ""))
        return out
    elif model.startswith("deepseek"):
        out = _call_deepseek(prompt, max_output, api_key, model)
        elapsed = time.time() - start
        if not quiet:
            print(f"API: deepseek | Elapsed: {elapsed:.2f}s")
            print("Output:\n" + (out or ""))
        return out
    elif model.startswith("chatgpt") or model.startswith("openai"):
        # allow "chatgpt" to map to default model
        model_name = model if model != "chatgpt" else DEFAULT_MODELS["chatgpt"]
        out = _call_openai(prompt, max_output, api_key, model_name)
        elapsed = time.time() - start
        if not quiet:
            print(f"API: chatgpt | Elapsed: {elapsed:.2f}s")
            print("Output:\n" + (out or ""))
        return out
    elif model.startswith("claude"):
        out = _call_claude(prompt, max_output, api_key, model)
        elapsed = time.time() - start
        if not quiet:
            print(f"API: claude | Elapsed: {elapsed:.2f}s")
            print("Output:\n" + (out or ""))
        return out
    elif model.startswith("grok") or model.startswith("xai"):
        model_name = model if model != "grok" else DEFAULT_MODELS["grok"]
        out = _call_grok(prompt, max_output, api_key, model_name)
        elapsed = time.time() - start
        if not quiet:
            print(f"API: grok | Elapsed: {elapsed:.2f}s")
            print("Output:\n" + (out or ""))
        return out
    else:
        raise ValueError("unsupported --model")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(
        "Unified API interface. Supported Gemini: " + ", ".join(GEMINI_SUPPORTED)
    ))
    p.add_argument(
        "--model",
        required=True,
        help="Model id (e.g., gemini-2.5-pro, deepseek-chat, chatgpt, claude-sonnet-4-5, grok-2-latest)",
    )
    p.add_argument("--prompt", required=True, help="Prompt text to send to the model")
    p.add_argument("--max_output", type=int, default=1024, help="Maximum output tokens/length")
    p.add_argument("--api", required=True, help="API key string for the chosen provider")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_api(args.model, args.prompt, args.max_output, args.keys, quiet=False)
