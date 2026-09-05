"""Stage 1 of the map-reduce: compress a team's week of news into a brief.

One LLM call per team, 32 per week. This step exists because 32 teams x 40
headlines does not fit usefully in a 7B model's context in a single call.

Schema bounds are load-bearing, not decoration. The host generates at ~10 tok/s,
so output length *is* runtime: unbounded arrays measured 20.2s per brief against
~10s with ``minItems: 3, maxItems: 4``, at the same useful content.
"""

from __future__ import annotations

from pathlib import Path

from nfl_football import config, llm
from nfl_football.models import Injury, NewsItem, TeamBrief, Team
from nfl_football.sources.injuries import summarise as summarise_injuries

PROMPT = (Path(__file__).parent / "prompts" / "team_brief.txt").read_text()

SYSTEM = (
    "You are an NFL beat reporter writing a factual situation report. "
    "You summarise only what your sources say. You never speculate about "
    "who will win a game."
)

# The model produces prose only. Injuries are passed through from ESPN's
# structured report untouched — asking the model to restate them wasted
# generation, invited fabrication, and (measured on the first 32-team run)
# crowded out the actual news: every storyline came back as a paraphrased
# injury line. A `sentiment` field was also dropped after returning "mixed"
# for 28 of 32 teams, i.e. carrying no signal for its token cost.
SCHEMA = {
    "type": "object",
    "properties": {
        "storylines": {
            "type": "array",
            "minItems": 3,
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 110},
        },
        "momentum": {"type": "string", "maxLength": 160},
    },
    "required": ["storylines", "momentum"],
}


def build_prompt(
    team: Team, news: list[NewsItem], injuries: list[Injury]
) -> str:
    injury_lines = summarise_injuries(injuries) or ["No injuries reported."]
    headlines = [
        f"- [{item.published:%b %d}] {item.title} ({item.source})"
        if item.published
        else f"- {item.title} ({item.source})"
        for item in news
    ] or ["- No headlines found."]
    return PROMPT.format(
        team=team.display_name,
        injuries="\n".join(f"- {line}" for line in injury_lines),
        headlines="\n".join(headlines),
    )


def build_team_brief(
    team: Team,
    news: list[NewsItem],
    injuries: list[Injury],
    *,
    refresh: bool = False,
) -> TeamBrief:
    """One brief for one team. Falls back to a source-only brief if the model
    fails, so a single bad call cannot sink the week."""
    prompt = build_prompt(team, news, injuries)
    try:
        result = llm.chat_json_cached(
            prompt, SCHEMA, system=SYSTEM, refresh=refresh,
            num_predict=config.NUM_PREDICT_BRIEF,
        )
    except llm.LLMError:
        return _fallback_brief(team, injuries)

    return TeamBrief(
        team_abbr=team.abbreviation,
        # Straight from ESPN, never through the model.
        injuries=summarise_injuries(injuries),
        storylines=[str(x) for x in result.data.get("storylines", [])],
        momentum=str(result.data.get("momentum", "")),
    )


def _fallback_brief(team: Team, injuries: list[Injury]) -> TeamBrief:
    """No narrative, but the injury facts survive — they came from the source,
    not the model."""
    return TeamBrief(
        team_abbr=team.abbreviation,
        injuries=summarise_injuries(injuries),
        storylines=[],
        momentum="",
    )
