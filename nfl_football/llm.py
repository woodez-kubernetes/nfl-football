"""Ollama client.

Two things make a 7B model usable here:

* **Schema-constrained generation.** Ollama accepts a JSON schema as ``format``,
  not just ``"json"``. Generation is constrained to match, so an ``enum`` field
  cannot come back with an out-of-vocabulary value. Guardrail #2 from plan.md §2
  is enforced by the decoder rather than by post-hoc validation.
* **An explicit ``num_ctx``.** Ollama defaults to 4096, which silently truncates
  the Stage-1 briefs. Every call sets it.

Retries re-send the conversation with a corrective message appended, so the
model sees what it got wrong.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Callable

import requests

from nfl_football import config


class LLMError(RuntimeError):
    """Raised when the model cannot produce a valid response within the retry budget."""


@dataclass
class LLMResult:
    data: dict
    prompt_tokens: int
    eval_tokens: int
    duration: float
    attempts: int


@dataclass
class Stats:
    calls: int = 0
    cached: int = 0
    retries: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    eval_tokens: int = 0
    seconds: float = 0.0

    def record(self, result: LLMResult) -> None:
        self.calls += 1
        self.retries += result.attempts - 1
        self.prompt_tokens += result.prompt_tokens
        self.eval_tokens += result.eval_tokens
        self.seconds += result.duration

    def summary(self) -> str:
        return (
            f"{self.calls} LLM calls ({self.cached} cached, {self.retries} retries, "
            f"{self.failures} failed) | {self.prompt_tokens:,} prompt + "
            f"{self.eval_tokens:,} generated tokens | {self.seconds:.1f}s"
        )


STATS = Stats()


def chat_json(
    prompt: str,
    schema: dict,
    *,
    system: str | None = None,
    validator: Callable[[dict], None] | None = None,
    model: str = config.OLLAMA_MODEL,
    num_ctx: int = config.OLLAMA_NUM_CTX,
    num_predict: int = config.OLLAMA_NUM_PREDICT,
    temperature: float = config.OLLAMA_TEMPERATURE,
    retries: int = config.OLLAMA_MAX_RETRIES,
    timeout: int = config.OLLAMA_TIMEOUT,
) -> LLMResult:
    """Send one prompt, get back a dict matching ``schema``.

    ``validator`` may raise ``ValueError`` to reject a structurally valid but
    semantically wrong response; its message is fed back to the model on retry.

    ``num_predict`` caps generated tokens. Wall time on this host is almost
    entirely generation (~10 tok/s), so that cap is the main runtime control.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    started = time.time()
    last_error = ""

    for attempt in range(1, retries + 2):
        payload = {
            "model": model,
            "stream": False,
            "format": schema,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "temperature": temperature,
            },
            "messages": messages,
        }
        try:
            resp = requests.post(
                f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=timeout
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as exc:
            STATS.failures += 1
            raise LLMError(f"Ollama request failed: {exc}") from exc

        content = (body.get("message") or {}).get("content", "")
        try:
            data = _extract_json(content)
            _require_keys(data, schema)
            if validator is not None:
                validator(data)
        except ValueError as exc:
            last_error = str(exc)
            if attempt > retries:
                break
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That response was rejected: {last_error}. "
                        "Reply again with only the corrected JSON object."
                    ),
                }
            )
            continue

        result = LLMResult(
            data=data,
            prompt_tokens=int(body.get("prompt_eval_count") or 0),
            eval_tokens=int(body.get("eval_count") or 0),
            duration=time.time() - started,
            attempts=attempt,
        )
        STATS.record(result)
        return result

    STATS.failures += 1
    raise LLMError(
        f"no valid response after {retries + 1} attempts; last error: {last_error}"
    )


def chat_json_cached(
    prompt: str,
    schema: dict,
    *,
    system: str | None = None,
    ttl: float = config.TTL_LLM,
    refresh: bool = False,
    **kwargs,
) -> LLMResult:
    """:func:`chat_json` with the response cached on disk.

    The key hashes model, schema, system and prompt, so any change to the input
    — a new headline, a tweaked template — misses the cache naturally. At ~10
    tok/s a full week is minutes of generation; this makes reruns free.
    """
    from nfl_football.sources.cache import default_cache

    model = kwargs.get("model", config.OLLAMA_MODEL)
    digest = hashlib.sha256(
        json.dumps(
            [model, schema, system, prompt, kwargs.get("temperature", config.OLLAMA_TEMPERATURE)],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    key = f"llm:{digest}"
    cache = default_cache()

    if not refresh:
        hit = cache.get(key, ttl)
        if hit is not None:
            STATS.cached += 1
            return LLMResult(
                data=json.loads(hit), prompt_tokens=0, eval_tokens=0,
                duration=0.0, attempts=0,
            )

    result = chat_json(prompt, schema, system=system, **kwargs)
    cache.put(key, json.dumps(result.data))
    return result


def _extract_json(content: str) -> dict:
    """Parse the model's reply. Schema mode returns clean JSON, but the decoder
    sometimes appends trailing whitespace, so fall back to a brace scan."""
    text = (content or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("response was not JSON")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("response was not a JSON object")
    return data


def _require_keys(data: dict, schema: dict) -> None:
    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        raise ValueError(f"missing required keys: {', '.join(missing)}")


def health_check(model: str = config.OLLAMA_MODEL) -> tuple[bool, str]:
    """Verify the host is reachable and the model is present."""
    try:
        resp = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
    except requests.RequestException as exc:
        return False, f"cannot reach Ollama at {config.OLLAMA_URL}: {exc}"
    if model not in names:
        return False, f"model {model!r} not on host; available: {', '.join(names) or 'none'}"
    return True, f"{config.OLLAMA_URL} ready with {model}"
