"""Orchestration: fetch -> brief -> thesis -> assembled report.

Ordering matters for cost. Briefs are built once per *team* (32) rather than
once per appearance (would be 32 anyway, but the dict guards future formats),
and the whole week is assembled before any rendering happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from nfl_football import briefs, config, probability, theses
from nfl_football.models import Game, GameReport, Team, TeamBrief
from nfl_football.sources import injuries as injuries_src
from nfl_football.sources import news as news_src
from nfl_football.sources import nflverse, schedule


@dataclass
class WeekReport:
    season: int
    week: int
    generated_at: datetime
    games: list[GameReport] = field(default_factory=list)
    odds_filled_from_nflverse: int = 0

    @property
    def upset_calls(self) -> list[GameReport]:
        return [g for g in self.games if g.thesis and g.thesis.is_upset_call]

    @property
    def games_without_a_pick(self) -> list[GameReport]:
        return [g for g in self.games if not (g.thesis and g.thesis.pick)]


def build_week(
    season: int | None = None,
    week: int | None = None,
    *,
    refresh: bool = False,
    use_llm: bool = True,
    only: str | None = None,
    progress=None,
) -> WeekReport:
    """Assemble a full week. ``only`` filters to one matchup, e.g. "DEN@KC"."""
    if season is None or week is None:
        resolved_season, resolved_week, season_type = schedule.resolve_current_week(
            refresh=refresh
        )
        season = season if season is not None else resolved_season
        week = week if week is not None else resolved_week
    else:
        season_type = config.SEASON_TYPE_REGULAR

    games = schedule.load_games(season, week, season_type, refresh=refresh)
    filled = nflverse.fill_missing(games, season, week, refresh=refresh)

    if only:
        wanted = only.upper()
        games = [g for g in games if g.matchup.upper() == wanted]

    all_injuries = injuries_src.load_injuries(refresh=refresh)
    team_briefs: dict[str, TeamBrief] = {}
    reports: list[GameReport] = []

    for index, game in enumerate(games, start=1):
        if progress:
            progress(index, len(games), game)

        prob = probability.win_probability(game.odds)
        home_news = news_src.load_team_news(game.home, refresh=refresh)
        away_news = news_src.load_team_news(game.away, refresh=refresh)
        home_inj = all_injuries.get(game.home.espn_id, [])
        away_inj = all_injuries.get(game.away.espn_id, [])

        if use_llm:
            home_brief = _brief(team_briefs, game.home, home_news, home_inj, refresh)
            away_brief = _brief(team_briefs, game.away, away_news, away_inj, refresh)
            thesis = theses.build_thesis(
                game, prob, home_brief, away_brief, refresh=refresh
            )
        else:
            home_brief = away_brief = None
            thesis = None

        reports.append(
            GameReport(
                game=game,
                probability=prob,
                thesis=thesis,
                home_brief=home_brief,
                away_brief=away_brief,
                home_news=home_news,
                away_news=away_news,
                home_injuries=home_inj,
                away_injuries=away_inj,
            )
        )

    return WeekReport(
        season=season,
        week=week,
        generated_at=datetime.now(timezone.utc),
        games=reports,
        odds_filled_from_nflverse=filled,
    )


def _brief(cache, team: Team, news, injuries, refresh: bool) -> TeamBrief:
    if team.abbreviation not in cache:
        cache[team.abbreviation] = briefs.build_team_brief(
            team, news, injuries, refresh=refresh
        )
    return cache[team.abbreviation]
