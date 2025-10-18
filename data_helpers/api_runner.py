"""Tiny CLI for Gemini/DeepSeek. Also importable."""

import argparse
import json
import time

import requests


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


def _call_gemini(prompt: str, max_output: int, api_key: str, timeout: int = 60) -> str:
    """Call Gemini."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-pro:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": str(prompt)}]}],
        "generationConfig": {"maxOutputTokens": int(max_output)},
    }
    headers = {"Content-Type": "application/json; charset=utf-8"}
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return _extract_gemini_text(data)
    except requests.exceptions.RequestException as e:
        return f"[network-error] {e}"
    except ValueError:
        return "[parse-error] Non-JSON response from Gemini"


def _call_deepseek(prompt: str, max_output: int, api_key: str, timeout: int = 60) -> str:
    """Call DeepSeek."""
    url = "https://api.deepseek.com/chat/completions"
    payload = {
        "model": "deepseek-chat",
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


def run_api(model: str, prompt: str, max_output: int = 512, api_key: str = "") -> str:
    """One switch for both APIs."""
    model = (model or "").strip().lower()
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError("--api key is required")
    start = time.time()
    if model == "gemini":
        out = _call_gemini(prompt, max_output, api_key)
        elapsed = time.time() - start
        print(f"API: gemini | Elapsed: {elapsed:.2f}s")
        print("Output:\n" + (out or ""))
        return out
    elif model == "deepseek":
        out = _call_deepseek(prompt, max_output, api_key)
        elapsed = time.time() - start
        print(f"API: deepseek | Elapsed: {elapsed:.2f}s")
        print("Output:\n" + (out or ""))
        return out
    else:
        raise ValueError("--model must be either 'gemini' or 'deepseek'")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified API interface for Gemini and DeepSeek.")
    p.add_argument("--model", required=True, choices=["gemini", "deepseek"], help="Which API to use")
    p.add_argument("--prompt", required=True, help="Prompt text to send to the model")
    p.add_argument("--max_output", type=int, default=512, help="Maximum output tokens/length")
    p.add_argument("--api", required=True, help="API key string for the chosen provider")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_api(args.model, args.prompt, args.max_output, args.api)
