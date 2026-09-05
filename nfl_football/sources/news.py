"""Team news via Google News RSS.

Headlines only — title, publisher and date. No article bodies are fetched: RSS
gives us everything we use, and scraping publisher pages would mean paywalls,
rate limits and murkier terms of use for no real gain in signal.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import feedparser

from nfl_football import config
from nfl_football.models import NewsItem, Team
from nfl_football.sources.cache import cached_get

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def load_team_news(
    team: Team | str,
    *,
    window: str = config.NEWS_WINDOW,
    limit: int = config.MAX_NEWS_PER_TEAM,
    refresh: bool = False,
) -> list[NewsItem]:
    name = team.display_name if isinstance(team, Team) else str(team)
    params = {
        "q": f'"{name}" when:{window}',
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    raw = cached_get(
        config.GNEWS_RSS, params=params, ttl=config.TTL_NEWS, refresh=refresh
    )
    return parse_feed(raw, limit=limit)


def parse_feed(raw: str, *, limit: int = config.MAX_NEWS_PER_TEAM) -> list[NewsItem]:
    parsed = feedparser.parse(raw)
    items: list[NewsItem] = []
    seen: set[str] = set()

    for entry in parsed.entries:
        source = (getattr(entry, "source", None) or {}).get("title") or "unknown"
        title = _strip_source_suffix(entry.get("title") or "", source)
        if not title:
            continue

        key = _normalise(title)
        if key in seen:
            continue
        seen.add(key)

        items.append(
            NewsItem(
                title=title,
                source=source,
                link=entry.get("link") or "",
                published=_parse_date(entry.get("published_parsed")),
            )
        )

    # Newest first; undated entries sort last.
    items.sort(key=lambda i: i.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return items[:limit]


def _strip_source_suffix(title: str, source: str) -> str:
    """Google News appends " - Publisher" to every headline; we already carry
    the publisher separately."""
    suffix = f" - {source}"
    if source and title.endswith(suffix):
        title = title[: -len(suffix)]
    return title.strip()


def _normalise(title: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", title.lower())).strip()


def _parse_date(parsed_time) -> datetime | None:
    if not parsed_time:
        return None
    return datetime(*parsed_time[:6], tzinfo=timezone.utc)
