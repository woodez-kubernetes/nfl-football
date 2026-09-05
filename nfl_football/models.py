"""Core data contracts.

Every later stage codes against these types. Two conventions are fixed here and
must be honoured by all source parsers:

* ``Odds.home_spread`` follows the ESPN sign convention — **negative means the
  home team is favoured** (``-3.5`` = home favoured by 3.5). nflverse uses the
  opposite sign, so ``sources/nflverse.py`` flips it on the way in.
* ``Odds`` moneylines are American odds as ints (``-175``, ``+145``). ESPN
  returns them as strings; parsers normalise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Team:
    espn_id: str
    abbreviation: str
    display_name: str
    location: str
    nickname: str

    def __str__(self) -> str:
        return self.display_name


@dataclass(frozen=True)
class Odds:
    """A single book's line for one game. Any field may be None before the
    book posts it, which is normal early in the week."""

    provider: str
    source: str  # "espn" | "nflverse"
    home_spread: float | None = None  # negative = home favoured
    over_under: float | None = None
    home_moneyline: int | None = None
    away_moneyline: int | None = None
    details: str | None = None  # e.g. "SEA -3.5"

    @property
    def has_moneyline(self) -> bool:
        return self.home_moneyline is not None and self.away_moneyline is not None


@dataclass(frozen=True)
class Game:
    espn_id: str
    season: int
    week: int
    season_type: int  # 1 = pre, 2 = regular, 3 = post
    kickoff: datetime  # tz-aware, UTC
    home: Team
    away: Team
    venue: str | None = None
    indoor: bool = False
    odds: Odds | None = None

    @property
    def matchup(self) -> str:
        return f"{self.away.abbreviation}@{self.home.abbreviation}"

    @property
    def title(self) -> str:
        return f"{self.away.display_name} at {self.home.display_name}"


@dataclass(frozen=True)
class WinProbability:
    """De-vigged win probabilities. ``home`` and ``away`` always sum to 1.0."""

    home: float
    away: float
    method: str  # "moneyline" | "spread"
    overround: float | None = None  # book's vig, only set for the moneyline path

    @property
    def favorite(self) -> str | None:
        """"home", "away", or None for a true pick'em."""
        if abs(self.home - self.away) < 1e-9:
            return None
        return "home" if self.home > self.away else "away"


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str
    link: str
    published: datetime | None = None


@dataclass(frozen=True)
class Injury:
    player: str
    position: str | None
    status: str | None
    comment: str | None = None


@dataclass
class TeamBrief:
    """Stage-1 output: ~40 headlines compressed into structured signal.

    ``storylines`` and ``momentum`` are written by the model. ``injuries`` is
    passed through from ESPN's structured report and never touches the LLM.
    """

    team_abbr: str
    injuries: list[str] = field(default_factory=list)
    storylines: list[str] = field(default_factory=list)
    momentum: str = ""


@dataclass
class Thesis:
    """Stage-2 LLM output. ``pick`` is the report's verdict.

    ``is_upset_call`` is computed in Python by comparing ``pick`` against the
    de-vigged favourite — never self-reported by the model.
    """

    game_id: str
    pick: str | None  # exact Team.display_name, or None if validation failed
    confidence: str  # derived from the market margin, not model-reported
    thesis: str
    key_factors: list[str] = field(default_factory=list)
    news_vs_market: str = ""
    is_upset_call: bool = False
    # From the blind first pass: what the news alone favoured, before the model
    # saw any betting line. Lets the report show where news and market diverge.
    news_lean: str | None = None
    lean_reason: str = ""

    @property
    def lean_differs_from_pick(self) -> bool:
        return (
            self.news_lean is not None
            and self.pick is not None
            and self.news_lean != self.pick
        )


@dataclass
class GameReport:
    """Everything the renderers need for one game."""

    game: Game
    probability: WinProbability | None
    thesis: Thesis | None
    home_brief: TeamBrief | None = None
    away_brief: TeamBrief | None = None
    home_news: list[NewsItem] = field(default_factory=list)
    away_news: list[NewsItem] = field(default_factory=list)
    home_injuries: list[Injury] = field(default_factory=list)
    away_injuries: list[Injury] = field(default_factory=list)

    @property
    def market_favorite(self) -> Team | None:
        if self.probability is None or self.probability.favorite is None:
            return None
        return self.game.home if self.probability.favorite == "home" else self.game.away
