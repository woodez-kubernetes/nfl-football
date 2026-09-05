"""ESPN injury report, keyed by ESPN team id.

The endpoint returns all 32 teams in one ~9 MB payload, so it is fetched once
per run and indexed, never per-team.
"""

from __future__ import annotations

import json

from nfl_football import config
from nfl_football.models import Injury
from nfl_football.sources.cache import cached_get


def load_injuries(*, refresh: bool = False) -> dict[str, list[Injury]]:
    """Returns ``{espn_team_id: [Injury, ...]}``."""
    raw = cached_get(config.ESPN_INJURIES, ttl=config.TTL_INJURIES, refresh=refresh)
    return parse_injuries(json.loads(raw))


def parse_injuries(payload: dict) -> dict[str, list[Injury]]:
    by_team: dict[str, list[Injury]] = {}

    for team_block in payload.get("injuries", []) or []:
        team_id = str(team_block.get("id", ""))
        entries: list[Injury] = []

        for item in team_block.get("injuries", []) or []:
            athlete = item.get("athlete") or {}
            position = (athlete.get("position") or {}).get("abbreviation")
            comment = item.get("longComment") or item.get("shortComment")
            entries.append(
                Injury(
                    player=athlete.get("displayName") or "unknown",
                    position=position,
                    status=item.get("status"),
                    comment=comment,
                )
            )

        by_team[team_id] = entries

    return by_team


def summarise(entries: list[Injury], limit: int = 12) -> list[str]:
    """Compact one-line-per-player strings for the LLM prompt.

    Ordered by severity so a truncated list keeps the players who matter.
    """
    rank = {"out": 0, "injured reserve": 0, "doubtful": 1, "questionable": 2}
    ordered = sorted(entries, key=lambda i: rank.get((i.status or "").lower(), 3))
    return [
        f"{i.player} ({i.position or '?'}) — {i.status or 'unknown'}"
        for i in ordered[:limit]
    ]
