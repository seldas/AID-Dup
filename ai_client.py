"""
LLM client for the standalone Dedup Study app.

Four FIXED model options (the study's model-ablation axis), each bound to a
provider and configured entirely through the .env file next to server.py
(see .env.example):

- llama-3.1  -> Ollama   (OpenAI-compatible /v1 endpoint, local)
- llama-4    -> vLLM     (OpenAI-compatible /v1 endpoint)
- sonnet-4.6 -> Elsa     (FDA internal runPixel proxy)
- haiku-4.5  -> Elsa

The UI only picks WHICH option runs (stored in the settings table);
endpoints, credentials, and model/engine ids live in .env so they never
enter the database or the browser.

Every call returns (text, usage) where usage carries token counts and an
estimated cost, so runs record LLM spend without a usage-log table.
"""

import json
import logging
import os
import re
from urllib.parse import quote_plus

import requests

try:  # Elsa uses verify=False (internal proxy) -- keep the log clean
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

logger = logging.getLogger(__name__)

# (connect, read) seconds: fail FAST when the endpoint host is unreachable
# (e.g. an internal vLLM host from outside the network) instead of hanging a
# synchronous request for minutes, but still allow slow models to generate.
REQUEST_TIMEOUT = (10, 300)

DEFAULT_MODEL_OPTION = "llama-3.1"

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_env_loaded = False


def _load_env():
    """Tiny .env loader (KEY=VALUE lines, # comments) -- values already set
    in the real environment win, so deployments can override the file."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    if not os.path.exists(_ENV_PATH):
        return
    with open(_ENV_PATH, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _env(key: str, default: str = "") -> str:
    _load_env()
    return os.environ.get(key, default)


def model_options() -> dict:
    """The four fixed options, with connection details resolved from .env at
    call time. `configured` flags whether the required .env keys are set.
    Also merges custom model profiles stored in database settings."""
    defaults = {
        "llama-3.1": {
            "label": "Llama-3.1 (Ollama)",
            "provider": "openai",
            "base_url": _env("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "api_key": _env("OLLAMA_API_KEY", "ollama"),
            "model": _env("OLLAMA_MODEL", "llama3.1"),
            "configured": bool(_env("OLLAMA_MODEL", "llama3.1")),
        },
        "llama-4": {
            "label": "Llama-4 (vLLM)",
            "provider": "openai",
            "base_url": _env("VLLM_BASE_URL"),
            "api_key": _env("VLLM_API_KEY", "EMPTY"),
            "model": _env("VLLM_MODEL"),
            "configured": bool(_env("VLLM_BASE_URL") and _env("VLLM_MODEL")),
        },
        "sonnet-4.6": {
            "label": "Claude Sonnet 4.6 (Elsa)",
            "provider": "elsa",
            "base_url": _env("ELSA_BASE_URL"),
            "api_key": "",
            "model": _env("ELSA_SONNET_MODEL_ID"),
            "configured": bool(_env("ELSA_BASE_URL") and _env("ELSA_API_NAME")
                               and _env("ELSA_API_KEY") and _env("ELSA_SONNET_MODEL_ID")),
        },
        "haiku-4.5": {
            "label": "Claude Haiku 4.5 (Elsa)",
            "provider": "elsa",
            "base_url": _env("ELSA_BASE_URL"),
            "api_key": "",
            "model": _env("ELSA_HAIKU_MODEL_ID"),
            "configured": bool(_env("ELSA_BASE_URL") and _env("ELSA_API_NAME")
                               and _env("ELSA_API_KEY") and _env("ELSA_HAIKU_MODEL_ID")),
        },
    }
    try:
        import db
        custom = db.get_setting("ai_models", {})
        if custom:
            defaults.update(custom)
    except Exception:
        pass
    return defaults


def resolve_option(option_key: str, bypass_configured: bool = False) -> dict:
    """Full call settings for one model option. Raises AIError when the
    option is unknown or its .env keys are missing."""
    options = model_options()
    option = options.get(option_key)
    if option is None:
        raise AIError(f"Unknown model option {option_key!r}. Choose one of: {', '.join(options)}.")
    if not bypass_configured and not option["configured"]:
        raise AIError(
            f"Model option '{option['label']}' is not configured. "
            "Fill in its keys in Dedup_Study/.env (see .env.example) and restart the server."
        )
    return {
        **option,
        "option_key": option_key,
        "temperature": float(_env("DEDUP_AI_TEMPERATURE", "0.1")),
        "max_tokens": int(_env("DEDUP_AI_MAX_TOKENS", "8000")),
    }


class AIError(Exception):
    pass


def estimate_cost(model_label: str, input_tokens: int, output_tokens: int) -> float:
    label = (model_label or "").lower()
    if "sonnet" in label:
        return input_tokens * 3.00 / 1e6 + output_tokens * 15.00 / 1e6
    if "haiku" in label:
        return input_tokens * 1.00 / 1e6 + output_tokens * 5.00 / 1e6
    if "gemini" in label:
        return input_tokens * 0.075 / 1e6 + output_tokens * 0.30 / 1e6
    return 0.0  # local llama models are free


def generate(prompt: str, settings: dict) -> tuple:
    """One LLM call. Returns (text, {"input_tokens", "output_tokens", "cost"}).
    Raises AIError on any failure."""
    provider = (settings.get("provider") or "openai").lower()
    if provider == "elsa":
        text, input_tokens, output_tokens = _call_elsa(prompt, settings)
    else:
        text, input_tokens, output_tokens = _call_openai_compatible(prompt, settings)
    return text or "", {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": estimate_cost(settings.get("label") or settings.get("model"), input_tokens, output_tokens),
    }


def _call_openai_compatible(prompt: str, settings: dict) -> tuple:
    """Ollama / vLLM / any OpenAI-compatible chat-completions endpoint."""
    base = (settings.get("base_url") or "").rstrip("/")
    if not base:
        raise AIError("No base_url configured for this model option (check .env).")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.get('api_key') or 'EMPTY'}"}
    payload = {
        "model": settings.get("model"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(settings.get("temperature") or 0.1),
        "max_tokens": int(settings.get("max_tokens") or 8000),
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        raise AIError(f"Could not reach the AI endpoint at {base}: {e}")
    if resp.status_code != 200:
        raise AIError(f"AI endpoint error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise AIError(f"Unexpected response shape: {json.dumps(data)[:300]}")
    usage = data.get("usage") or {}
    return text, usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0


def _call_elsa(prompt: str, settings: dict) -> tuple:
    """FDA Elsa runPixel proxy (same call shape as the main app): a
    form-encoded LLM(...) pixel with basic auth. Token counts are estimated
    (the proxy does not report usage)."""
    base_url = settings.get("base_url") or _env("ELSA_BASE_URL")
    username = _env("ELSA_API_NAME")
    password = _env("ELSA_API_KEY")
    model = settings.get("model")
    if not (base_url and username and password and model):
        raise AIError("Elsa is not configured (ELSA_BASE_URL / ELSA_API_NAME / ELSA_API_KEY / model id in .env).")

    command = (
        f'LLM(engine = "{model}", command = "<encode>{prompt}</encode>", '
        f'paramValues = [{{"max_completion_tokens": {int(settings.get("max_tokens") or 8000)}, '
        f'"temperature": {float(settings.get("temperature") or 0.1)}}}])'
    )
    try:
        resp = requests.post(
            base_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=f"expression={quote_plus(command)}",
            auth=(username, password),
            verify=False,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise AIError(f"Could not connect to Elsa: {e}")
    if resp.status_code != 200:
        raise AIError(f"Elsa error {resp.status_code}: {resp.text[:300]}")
    try:
        text = json.loads(resp.text)["pixelReturn"][0]["output"]["response"]
    except Exception:
        raise AIError(f"Elsa returned an unexpected payload: {resp.text[:300]}")
    # Elsa doesn't report token usage; approximate at 4 chars/token.
    return text, len(prompt) // 4, len(text) // 4


def parse_json_response(text: str):
    """Robust JSON extraction from a model response: strips code fences, then
    falls back to the first balanced {...} object. Raises ValueError."""
    if not text or not str(text).strip():
        raise ValueError("Empty response text")
    text = str(text).strip()
    # curly quotes / BOM / zero-width cleanup
    text = (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'")
                .replace("﻿", "").replace("​", ""))

    fence = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", text, re.DOTALL)
    candidates = [fence.group(1).strip()] if fence else [text]
    candidates += re.findall(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, re.DOTALL)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    obj = _extract_first_json_object(text)
    if obj is not None:
        return json.loads(obj)
    raise ValueError(f"No JSON object found in response: {text[:200]}")


def _extract_first_json_object(s: str):
    """First balanced {...} object, ignoring braces inside strings."""
    start = s.find("{")
    if start == -1:
        return None
    in_string = False
    escape = False
    quote_char = None
    depth = 0
    obj_start = None
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
                quote_char = None
            continue
        if ch in ('"', "'"):
            in_string = True
            quote_char = ch
            continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                return s[obj_start:i + 1]
    return None
