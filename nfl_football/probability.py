"""Win probability from betting markets.

This module is the reason the LLM never does arithmetic. Everything here is
deterministic and tested; the model receives the output as a finished fact.

Two paths:

* **Moneyline (preferred).** American odds -> implied probability -> remove the
  bookmaker's vig by normalising the pair to sum to 1.0.
* **Spread (fallback).** A logistic curve on the point spread, used when a book
  has posted a spread but no moneyline.

Sign convention: ``Odds.home_spread`` is negative when the home team is
favoured (ESPN convention). See ``models.Odds``.
"""

from __future__ import annotations

import math

from nfl_football.models import Game, Odds, WinProbability

# Logit per point of spread, fitted on nflverse 1999-2026.
#
# Fitted by least squares through the origin on logit(de-vigged market
# probability) against spread_line, over the 5,407 games carrying both a spread
# and both moneylines. Fitting through the origin is justified empirically: the
# free-intercept fit gives intercept = 0.0063, i.e. a pick'em is a coin flip.
#
#   fit vs market probability : mean abs err 0.0147, median 0.0128, p95 0.0348
#
# Validated against actual results on 6,952 regular-season games (ties dropped):
#
#   n-weighted mean abs calibration error : 0.0243
#   Brier score  0.2118  (coin flip: 0.2500)
#
# Re-derive with tests/calibrate_spread.py if the historical data is refreshed.
SPREAD_LOGIT_PER_POINT = 0.13932


def american_to_implied(moneyline: int) -> float:
    """American odds -> implied probability, vig included.

    ``-175`` -> 0.6364, ``+145`` -> 0.4082. These sum to more than 1.0; that
    excess is the book's margin, removed by :func:`devig`.
    """
    if moneyline < 0:
        return (-moneyline) / ((-moneyline) + 100.0)
    if moneyline > 0:
        return 100.0 / (moneyline + 100.0)
    raise ValueError("moneyline of 0 is not a valid American price")


def devig(p_home: float, p_away: float) -> tuple[float, float, float]:
    """Normalise a raw pair to sum to 1.0.

    Returns ``(home, away, overround)`` where overround is the book's margin
    (typically ~0.025 in this dataset).
    """
    total = p_home + p_away
    if total <= 0:
        raise ValueError("implied probabilities must be positive")
    return p_home / total, p_away / total, total - 1.0


def from_moneyline(home_moneyline: int, away_moneyline: int) -> WinProbability:
    home, away, overround = devig(
        american_to_implied(home_moneyline), american_to_implied(away_moneyline)
    )
    return WinProbability(
        home=home, away=away, method="moneyline", overround=overround
    )


def from_spread(home_spread: float) -> WinProbability:
    """Logistic fallback. ``home_spread`` is negative when home is favoured."""
    # The curve was fitted against nflverse's sign (positive = home favoured),
    # so flip into that frame.
    home_margin = -home_spread
    home = 1.0 / (1.0 + math.exp(-SPREAD_LOGIT_PER_POINT * home_margin))
    return WinProbability(home=home, away=1.0 - home, method="spread", overround=None)


def win_probability(odds: Odds | None) -> WinProbability | None:
    """Best available probability for a line, or None if nothing is posted.

    Prefers the moneyline; falls back to the spread; returns None when neither
    is available, which is normal before books post a game.
    """
    if odds is None:
        return None
    if odds.has_moneyline:
        try:
            return from_moneyline(odds.home_moneyline, odds.away_moneyline)
        except ValueError:
            pass  # malformed price: fall through to the spread
    if odds.home_spread is not None:
        return from_spread(odds.home_spread)
    return None


def confidence_label(probability: WinProbability | None) -> str:
    """Confidence derived from the market margin, not asked of the model.

    The model returned "high" for all 16 games in testing — the same degenerate
    behaviour that got the Stage-4 ``sentiment`` field dropped. Deriving it keeps
    the value meaningful and consistent with "numbers in Python".
    """
    if probability is None:
        return "unknown"
    edge = max(probability.home, probability.away)
    if edge >= 0.70:
        return "high"
    if edge >= 0.55:
        return "medium"
    return "low"


def plain_language(probability: WinProbability, game: Game) -> str:
    """Render probabilities the way the LLM must see them.

    The model is never shown American odds — ``-175`` is exactly the notation it
    misread in testing. It gets percentages by team name instead. See plan.md §2.
    """
    return (
        f"{game.home.display_name} {probability.home * 100:.0f}%, "
        f"{game.away.display_name} {probability.away * 100:.0f}%"
    )
