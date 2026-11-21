"""Shared Gemini access: clients, retry/backoff, 503 fallback, 429 stop, and call logging.

Every genai call in the project goes through `call()` so that:
  * 429 -> SpendCapHit is raised immediately (project spend cap; a human must look).
  * 503 / 500 / connection errors -> exponential backoff, then fall back (Hydrogen <-> Neon).
  * the request (model, config, prompt; inline media stripped) and output are logged to logs/.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from . import config

load_dotenv(config.ROOT / ".env")

_client = None
_beta_client = None


class SpendCapHit(RuntimeError):
    """HTTP 429: project spend cap (aistudio.google.com/spend). Stop and tell the user."""


def client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client()
    return _client


def beta_client():
    """v1beta client for Decisions + Managed Agents."""
    global _beta_client
    if _beta_client is None:
        from google import genai

        _beta_client = genai.Client(http_options={"api_version": "v1beta"})
    return _beta_client


def client_for(model: str):
    """v1beta client for preview models (config.V1BETA_MODELS), default client otherwise."""
    return beta_client() if model in config.V1BETA_MODELS else client()


def _status(e: Exception) -> int | None:
    for attr in ("status_code", "code"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    return None


_B64_RUN = re.compile(r"(?:[A-Za-z0-9+/=_-]|\\n|\n){2000,}")  # base64, including runs broken by (escaped) newlines


def _strip_media(obj: Any) -> Any:
    """Drop inline bytes/base64 from requests and responses before logging."""
    if isinstance(obj, str):
        return _B64_RUN.sub(lambda m: f"<stripped base64 {len(m.group(0))} chars>", obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "data" and isinstance(v, (str, bytes)) and len(v) > 256:
                out[k] = f"<stripped {len(v)} chars>"
            else:
                out[k] = _strip_media(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_strip_media(v) for v in obj]
    if isinstance(obj, bytes):
        return f"<bytes {len(obj)}>"
    return obj


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json", exclude_none=True)
        except Exception:
            pass
    if isinstance(obj, (dict, list, tuple, str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _log(record: dict) -> Path:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = config.LOG_DIR / f"genai_calls_{day}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(_strip_media(_to_jsonable(record)), default=repr) + "\n")
    return path


def call(
    fn: Callable[..., Any],
    *,
    kind: str,
    model: str,
    tag: str = "",
    max_retries: int = 4,
    fallback: bool = False,
    **kwargs: Any,
) -> Any:
    """Invoke `fn(model=model, **kwargs)` with retry/backoff + logging.

    kind: short label ("interactions.create", "decisions.create", ...).
    fallback: on repeated 503, retry once with config.FALLBACK[model].
    """
    call_id = uuid.uuid4().hex[:12]
    attempt, cur_model, delay = 0, model, 2.0
    while True:
        t0 = time.time()
        try:
            resp = fn(model=cur_model, **kwargs)
            dt = time.time() - t0
            _log({
                "id": call_id, "ts": datetime.now(timezone.utc).isoformat(), "kind": kind, "tag": tag,
                "model": cur_model, "requested_model": model, "attempt": attempt, "latency_s": round(dt, 3),
                "request": kwargs, "response": resp,
            })
            return resp
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            code = _status(e)
            _log({
                "id": call_id, "ts": datetime.now(timezone.utc).isoformat(), "kind": kind, "tag": tag,
                "model": cur_model, "attempt": attempt, "latency_s": round(dt, 3), "request": kwargs,
                "error": {"status": code, "type": type(e).__name__, "message": str(e)[:2000]},
            })
            if code == 429:
                raise SpendCapHit(
                    f"429 from {cur_model}: project spend cap hit (aistudio.google.com/spend). Stopping."
                ) from e
            retryable = code in (500, 502, 503, 504) or code is None and "connect" in type(e).__name__.lower()
            if not retryable:
                raise
            attempt += 1
            if attempt > max_retries:
                alt = config.FALLBACK.get(cur_model) if fallback else None
                if alt and cur_model == model:
                    cur_model, attempt, delay = alt, 0, 2.0
                    continue
                raise
            time.sleep(delay + random.uniform(0, 1))
            delay = min(delay * 2, 30)


def upload_video(path: str | Path, wait_s: float = 300) -> Any:
    """Upload a video with the Files API and wait until it is ACTIVE."""
    c = client()
    f = call(lambda model, **kw: c.files.upload(**kw), kind="files.upload", model="files", file=str(path))
    t0 = time.time()
    while str(getattr(f, "state", "")).upper().endswith("PROCESSING"):
        if time.time() - t0 > wait_s:
            raise TimeoutError(f"file {f.name} still processing after {wait_s}s")
        time.sleep(2)
        f = c.files.get(name=f.name)
    return f


def have_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
