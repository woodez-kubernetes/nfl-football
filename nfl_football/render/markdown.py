"""Markdown report for the terminal.

Shows three things side by side for every game — what the news alone favoured,
what the market favours, and the model's final pick — so a disagreement is
visible rather than buried in prose.
"""

from __future__ import annotations

import shutil

from nfl_football import config
from nfl_football.models import GameReport
from nfl_football.pipeline import WeekReport

BAR_WIDTH = 24


def render(report: WeekReport, *, width: int | None = None, sources: bool = True) -> str:
    width = width or min(shutil.get_terminal_size((100, 24)).columns, 100)
    out: list[str] = []

    out.append(f"# NFL {report.season} — Week {report.week}")
    out.append("")
    generated = config.to_local(report.generated_at)
    out.append(f"_Generated {generated:%Y-%m-%d %H:%M %Z}_")
    out.append("")
    out.extend(_overview(report, width))

    for game_report in report.games:
        out.append("")
        out.extend(_game(game_report, width, sources))

    return "\n".join(out) + "\n"


def _overview(report: WeekReport, width: int) -> list[str]:
    rows = []
    for gr in report.games:
        thesis = gr.thesis
        pick = _abbr(gr, thesis.pick) if thesis and thesis.pick else "—"
        lean = _abbr(gr, thesis.news_lean) if thesis and thesis.news_lean else "—"
        fav = gr.market_favorite.abbreviation if gr.market_favorite else "—"
        flag = "UPSET" if thesis and thesis.is_upset_call else ""
        conf = thesis.confidence if thesis else ""
        rows.append(
            f"| {gr.game.matchup:<8} | {config.to_local(gr.game.kickoff):%a %H:%M}  | {fav:^6} "
            f"| {lean:^6} | {pick:^6} | {conf:<7} | {flag:<5} |"
        )

    header = [
        f"## Week at a glance — times in {config.REPORT_TZ_NAME.split('/')[-1].replace('_', ' ')}",
        "",
        "| Game     | Kickoff    | Market | News   | PICK   | Conf.   | Flag  |",
        "|----------|------------|--------|--------|--------|---------|-------|",
    ]
    footer = [""]
    upsets = len(report.upset_calls)
    missing = len(report.games_without_a_pick)
    summary = f"{upsets} of {len(report.games)} picks go against the market."
    if missing:
        summary += f" {missing} game(s) have no pick — the model failed."
    footer.append(summary)
    return header + rows + footer


def _game(gr: GameReport, width: int, sources: bool) -> list[str]:
    game, thesis, prob = gr.game, gr.thesis, gr.probability
    out = ["---", "", f"## {game.title}", ""]

    venue = f"{game.venue}{' (indoor)' if game.indoor else ''}" if game.venue else "venue TBD"
    kickoff = config.to_local(game.kickoff)
    out.append(f"**{kickoff:%A %d %B, %H:%M %Z}** · {venue}")
    out.append("")

    if prob is None:
        out.append("_No betting line posted._")
    else:
        out.append(_bar(game.away.abbreviation, prob.away))
        out.append(_bar(game.home.abbreviation, prob.home))
        out.append("")
        out.append(_line_summary(gr))
    out.append("")

    if thesis is None:
        out.append("_No thesis generated (--no-llm)._")
        return out

    if thesis.pick is None:
        out.append("**No pick** — the model did not return a valid response.")
        return out

    verdict = f"### Pick: {thesis.pick}"
    if thesis.is_upset_call:
        verdict += "  ⚠️ **UPSET CALL**"
    out.append(verdict)
    out.append("")

    fav = gr.market_favorite.display_name if gr.market_favorite else "no favourite"
    out.append(f"- Market favours: **{fav}** · confidence *{thesis.confidence}*")
    if thesis.news_lean:
        marker = " (differs from final pick)" if thesis.lean_differs_from_pick else ""
        out.append(f"- News alone favoured: **{thesis.news_lean}**{marker}")
        if thesis.lean_reason:
            out.append(f"  - {thesis.lean_reason}")
    out.append("")

    out.append(thesis.thesis)
    out.append("")

    if thesis.key_factors:
        out.append("**Key factors**")
        out.extend(f"- {factor}" for factor in thesis.key_factors)
        out.append("")

    if thesis.news_vs_market:
        out.append(f"**News vs market:** {thesis.news_vs_market}")
        out.append("")

    out.extend(_injuries(gr))
    if sources:
        out.extend(_sources(gr))
    return out


def _injuries(gr: GameReport) -> list[str]:
    out: list[str] = []
    for brief, team in ((gr.away_brief, gr.game.away), (gr.home_brief, gr.game.home)):
        if brief and brief.injuries:
            out.append(f"**{team.abbreviation} injuries**")
            out.extend(f"- {entry}" for entry in brief.injuries[:6])
            out.append("")
    return out


def _sources(gr: GameReport) -> list[str]:
    """Every headline the briefs drew on, so a claim can be traced back."""
    out: list[str] = []
    for items, team in ((gr.away_news, gr.game.away), (gr.home_news, gr.game.home)):
        if not items:
            continue
        out.append(f"<details><summary>{team.abbreviation} sources ({len(items)})</summary>")
        out.append("")
        for item in items[:10]:
            date = f"{item.published:%b %d}" if item.published else "—"
            out.append(f"- [{item.title}]({item.link}) — {item.source}, {date}")
        out.append("")
        out.append("</details>")
        out.append("")
    return out


def _line_summary(gr: GameReport) -> str:
    odds = gr.game.odds
    if odds is None:
        return ""
    parts = []
    if odds.details:
        parts.append(f"Line: {odds.details}")
    elif odds.home_spread is not None:
        parts.append(f"Spread: {odds.home_spread:+g} (home)")
    if odds.over_under is not None:
        parts.append(f"O/U {odds.over_under}")
    if gr.probability and gr.probability.method == "spread":
        parts.append("probability from spread")
    parts.append(f"via {odds.provider}")
    return " · ".join(parts)


def _bar(label: str, value: float) -> str:
    filled = round(value * BAR_WIDTH)
    return f"`{label:<4}` `{'█' * filled}{'░' * (BAR_WIDTH - filled)}` **{value * 100:.0f}%**"


def _abbr(gr: GameReport, display_name: str) -> str:
    if display_name == gr.game.home.display_name:
        return gr.game.home.abbreviation
    if display_name == gr.game.away.display_name:
        return gr.game.away.abbreviation
    return "?"
