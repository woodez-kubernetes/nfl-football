"""Team abbreviation normalisation.

ESPN and nflverse disagree on two current teams. Everything downstream keys on
the ESPN abbreviation; :func:`to_espn` converts inbound nflverse codes.
"""

from __future__ import annotations

# nflverse code -> ESPN code
_NFLVERSE_TO_ESPN = {
    "LA": "LAR",  # Rams
    "WAS": "WSH",  # Commanders
    # Relocated franchises, present in nflverse history only.
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
}

_ESPN_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS"}


def to_espn(abbr: str) -> str:
    abbr = (abbr or "").upper().strip()
    return _NFLVERSE_TO_ESPN.get(abbr, abbr)


def to_nflverse(abbr: str) -> str:
    abbr = (abbr or "").upper().strip()
    return _ESPN_TO_NFLVERSE.get(abbr, abbr)
