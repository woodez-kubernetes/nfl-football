"""nflverse games.csv — odds fallback when ESPN has no line posted.

**Sign convention:** nflverse ``spread_line`` is positive when the *home* team
is favoured, the opposite of ESPN. It is negated on ingest so that everything
downstream sees the single convention documented on ``models.Odds``.

Verified 2026-09-02: nflverse and ESPN agree exactly on 2026 Wk1 NE@SEA
(-175 / +145, total 44.5).
"""

from __future__ import annotations

import csv
import io
from dataclasses import replace

from nfl_football import config
from nfl_football.models import Odds
from nfl_football.sources.cache import cached_get
from nfl_football.teams import to_espn


def load_odds(
    season: int, week: int, *, refresh: bool = False
) -> dict[tuple[str, str], Odds]:
    """Returns ``{(away_espn_abbr, home_espn_abbr): Odds}`` for one week."""
    raw = cached_get(
        config.NFLVERSE_GAMES, ttl=config.TTL_SCHEDULE, refresh=refresh
    )
    return parse_games(raw, season, week)


def parse_games(raw: str, season: int, week: int) -> dict[tuple[str, str], Odds]:
    out: dict[tuple[str, str], Odds] = {}

    for row in csv.DictReader(io.StringIO(raw)):
        if row.get("season") != str(season) or row.get("week") != str(week):
            continue

        away = to_espn(row.get("away_team", ""))
        home = to_espn(row.get("home_team", ""))
        if not away or not home:
            continue

        spread = _as_float(row.get("spread_line"))
        out[(away, home)] = Odds(
            provider="nflverse",
            source="nflverse",
            # Negate: nflverse positive = home favoured, ESPN negative = home favoured.
            home_spread=None if spread is None else -spread,
            over_under=_as_float(row.get("total_line")),
            home_moneyline=_as_int(row.get("home_moneyline")),
            away_moneyline=_as_int(row.get("away_moneyline")),
            details=None,
        )

    return out


def fill_missing(games: list, season: int, week: int, *, refresh: bool = False) -> int:
    """Attach nflverse odds to any Game lacking a usable line.

    Replaces entries in ``games`` in place (Game itself is frozen) and returns
    how many were filled. Skips the CSV download entirely when nothing needs it.
    """
    if not _needs_odds(games):
        return 0
    return _fill_from_table(games, load_odds(season, week, refresh=refresh))


def _needs_odds(games: list) -> bool:
    return any(g.odds is None or not g.odds.has_moneyline for g in games)


def _fill_from_table(games: list, table: dict[tuple[str, str], Odds]) -> int:
    """Pure half of :func:`fill_missing` — no network, so it is directly testable."""
    filled = 0
    for i, game in enumerate(games):
        if game.odds is not None and game.odds.has_moneyline:
            continue
        replacement = table.get((game.away.abbreviation, game.home.abbreviation))
        if replacement is None:
            continue
        games[i] = replace(game, odds=replacement)
        filled += 1
    return filled


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
