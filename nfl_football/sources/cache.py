"""SQLite-backed HTTP cache.

Every outbound request in this project goes through :func:`cached_get`, so the
User-Agent, timeout and TTL policy live in exactly one place. Re-running a week
inside the TTL makes zero network calls, which keeps development off ESPN's back.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Iterable

import requests

from nfl_football import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    key        TEXT PRIMARY KEY,
    fetched_at REAL NOT NULL,
    payload    TEXT NOT NULL
)
"""


class Cache:
    def __init__(self, path=None):
        self.path = path or config.CACHE_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, key: str, ttl: float) -> str | None:
        row = self._conn.execute(
            "SELECT fetched_at, payload FROM http_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        fetched_at, payload = row
        if time.time() - fetched_at > ttl:
            return None
        return payload

    def put(self, key: str, payload: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO http_cache (key, fetched_at, payload) VALUES (?, ?, ?)",
            (key, time.time(), payload),
        )
        self._conn.commit()

    def age(self, key: str) -> float | None:
        """Seconds since this key was cached, or None if absent."""
        row = self._conn.execute(
            "SELECT fetched_at FROM http_cache WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else time.time() - row[0]

    def close(self) -> None:
        self._conn.close()


_default_cache: Cache | None = None


def default_cache() -> Cache:
    global _default_cache
    if _default_cache is None:
        _default_cache = Cache()
    return _default_cache


# Counters so the CLI can report how much of a run was served from cache.
STATS = {"hits": 0, "misses": 0}


def cached_get(
    url: str,
    *,
    ttl: float,
    params: dict | None = None,
    refresh: bool = False,
    cache: Cache | None = None,
) -> str:
    """GET ``url``, returning response text, transparently cached for ``ttl``.

    Raises ``requests.HTTPError`` on a non-2xx response only when there is no
    usable cached copy to fall back on.
    """
    cache = cache or default_cache()
    key = _cache_key(url, params)

    if not refresh:
        hit = cache.get(key, ttl)
        if hit is not None:
            STATS["hits"] += 1
            return hit

    STATS["misses"] += 1
    try:
        # No custom User-Agent by design — see the note in config.py. ESPN
        # 403s on browser-like and unrecognised UAs but accepts the default
        # "python-requests/x.y".
        resp = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        # A stale copy beats no data — a dead source shouldn't kill the run.
        stale = cache.get(key, ttl=float("inf"))
        if stale is not None:
            return stale
        raise

    cache.put(key, resp.text)
    return resp.text


def _cache_key(url: str, params: dict | None) -> str:
    if not params:
        return url
    parts: Iterable[str] = (f"{k}={params[k]}" for k in sorted(params))
    return f"{url}?{'&'.join(parts)}"
