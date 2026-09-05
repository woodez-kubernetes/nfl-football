"""NFL weekly matchup thesis generator."""

__version__ = "0.1.0"

from nfl_football.models import (
    GameReport,
    Game,
    Injury,
    NewsItem,
    Odds,
    Team,
    TeamBrief,
    Thesis,
    WinProbability,
)

__all__ = [
    "Game",
    "GameReport",
    "Injury",
    "NewsItem",
    "Odds",
    "Team",
    "TeamBrief",
    "Thesis",
    "WinProbability",
]
