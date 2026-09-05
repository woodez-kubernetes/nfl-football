"""ESPN scoreboard -> Game objects, with odds attached.

Field paths verified against a live payload on 2026-09-02; see plan.md §3 for
the full table. Two things that bite:

* ``homeTeamOdds.moneyLine`` is always ``null``. Real moneylines live under
  ``odds[0].moneyline.<side>.close.odds`` as strings.
* ``odds[0].spread`` is home-relative and **negative means home favoured**.
"""

from __future__ import annotations

import json
from datetime import datetime

from nfl_football import config
from nfl_football.models import Game, Odds, Team
from nfl_football.sources.cache import cached_get


def load_games(
    season: int,
    week: int,
    season_type: int = config.SEASON_TYPE_REGULAR,
    *,
    refresh: bool = False,
) -> list[Game]:
    params = {"seasontype": season_type, "week": week, "dates": season}
    raw = cached_get(
        config.ESPN_SCOREBOARD, params=params, ttl=config.TTL_ODDS, refresh=refresh
    )
    return parse_scoreboard(json.loads(raw))


def resolve_current_week(*, refresh: bool = False) -> tuple[int, int, int]:
    """Ask ESPN which week it currently is: (season, week, season_type).

    Preferred over the calendar arithmetic in ``config.current_season_week()``.
    """
    raw = cached_get(config.ESPN_SCOREBOARD, ttl=config.TTL_ODDS, refresh=refresh)
    payload = json.loads(raw)
    season = payload.get("season", {}) or {}
    week = payload.get("week", {}) or {}
    fallback_season, fallback_week = config.current_season_week()
    return (
        int(season.get("year") or fallback_season),
        int(week.get("number") or fallback_week),
        int(season.get("type") or config.SEASON_TYPE_REGULAR),
    )


def parse_scoreboard(payload: dict) -> list[Game]:
    season_block = payload.get("season", {}) or {}
    games: list[Game] = []

    for event in payload.get("events", []) or []:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]

        sides = {c.get("homeAway"): c for c in comp.get("competitors", []) or []}
        if "home" not in sides or "away" not in sides:
            continue

        venue = comp.get("venue") or {}
        week_block = event.get("week") or {}

        games.append(
            Game(
                espn_id=str(event.get("id")),
                season=int(season_block.get("year") or 0),
                week=int(week_block.get("number") or 0),
                season_type=int(season_block.get("type") or config.SEASON_TYPE_REGULAR),
                kickoff=_parse_kickoff(event.get("date")),
                home=_parse_team(sides["home"]),
                away=_parse_team(sides["away"]),
                venue=venue.get("fullName"),
                indoor=bool(venue.get("indoor", False)),
                odds=parse_odds(comp),
            )
        )

    games.sort(key=lambda g: g.kickoff)
    return games


def _parse_team(competitor: dict) -> Team:
    t = competitor.get("team", {}) or {}
    return Team(
        espn_id=str(t.get("id", "")),
        abbreviation=(t.get("abbreviation") or "").upper(),
        display_name=t.get("displayName") or "",
        location=t.get("location") or "",
        nickname=t.get("name") or "",
    )


def _parse_kickoff(value: str | None) -> datetime:
    # ESPN emits "2026-09-10T00:20Z"; fromisoformat needs +00:00.
    if not value:
        raise ValueError("event is missing a kickoff date")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_odds(comp: dict) -> Odds | None:
    """Extract the first book's line. Returns None when nothing is posted yet,
    which is normal early in the week and must not be treated as an error."""
    odds_list = comp.get("odds") or []
    if not odds_list:
        return None
    o = odds_list[0]

    home_spread = _as_float(o.get("spread"))
    if home_spread is None:
        home_spread = _as_float(_leg(o, "pointSpread", "home").get("line"))

    over_under = _as_float(o.get("overUnder"))
    if over_under is None:
        over_under = _as_float(_leg(o, "total", "over").get("line"))

    return Odds(
        provider=(o.get("provider") or {}).get("name") or "unknown",
        source="espn",
        home_spread=home_spread,
        over_under=over_under,
        home_moneyline=_moneyline(o, "home"),
        away_moneyline=_moneyline(o, "away"),
        details=o.get("details"),
    )


def _leg(odds: dict, market: str, side: str) -> dict:
    """``odds[market][side].close``, falling back to ``.open``."""
    block = ((odds.get(market) or {}).get(side) or {})
    return block.get("close") or block.get("open") or {}


def _moneyline(odds: dict, side: str) -> int | None:
    return _as_american(_leg(odds, "moneyline", side).get("odds"))


def _as_american(value) -> int | None:
    """Parse ESPN's string odds. Handles "+145", "-175", "EVEN", "OFF"."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().upper()
    if text in {"EVEN", "EV", "PK"}:
        return 100
    if text in {"", "OFF", "-", "N/A"}:
        return None
    try:
        return int(float(text.replace("+", "")))
    except ValueError:
        return None


def _as_float(value) -> float | None:
    """Parse a line value. Handles floats and strings like "o44.5"/"-3.5"."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lstrip("ou").replace("+", "")
    try:
        return float(text)
    except ValueError:
        return None
