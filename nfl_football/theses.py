"""Stage 2 of the map-reduce: two team briefs + the market -> a verdict.

The user's decision (plan.md §1) is that the **LLM makes the final pick**, not
the odds. The three guardrails from §2 are all enforced here:

1. The model sees probabilities as plain-language percentages, never American
   odds — that notation is what it misread in the original smoke test.
2. ``pick`` is a schema ``enum`` of exactly the two team names, so Ollama's
   constrained decoding cannot emit anything else.
3. ``is_upset_call`` is computed **in Python** by comparing the pick against the
   de-vigged favourite. The model is never asked to self-report disagreement.
"""

from __future__ import annotations

from pathlib import Path

from nfl_football import config, llm
from nfl_football.models import Game, TeamBrief, Thesis, WinProbability
from nfl_football.probability import confidence_label, plain_language

PROMPT = (Path(__file__).parent / "prompts" / "game_thesis.txt").read_text()
LEAN_PROMPT = (Path(__file__).parent / "prompts" / "news_lean.txt").read_text()

SYSTEM = (
    "You are an NFL analyst making a straight call on who wins. You argue from "
    "the evidence in front of you and you never invent facts."
)

LEAN_SYSTEM = (
    "You are an NFL analyst assessing which team is in better shape right now, "
    "using only the reporting in front of you."
)


def build_schema(game: Game) -> dict:
    """Per-game schema: the pick enum is this game's two teams and nothing else.

    ``confidence`` is deliberately absent — the model returned "high" for all 16
    games, so it is derived from the market margin in Python instead.
    """
    return {
        "type": "object",
        "properties": {
            "pick": {
                "type": "string",
                "enum": [game.home.display_name, game.away.display_name],
            },
            "thesis": {"type": "string", "maxLength": 520},
            # maxLength is a hard grammar constraint: the decoder closes the
            # string at the limit, mid-word if necessary. Headroom above the
            # prompt's 15-word guidance keeps factors from being cut off.
            "key_factors": {
                "type": "array",
                "minItems": 3,
                "maxItems": 4,
                "items": {"type": "string", "maxLength": 170},
            },
            "news_vs_market": {"type": "string", "maxLength": 220},
        },
        "required": ["pick", "thesis", "key_factors", "news_vs_market"],
    }


def build_lean_schema(game: Game) -> dict:
    return {
        "type": "object",
        "properties": {
            "news_lean": {
                "type": "string",
                "enum": [game.home.display_name, game.away.display_name],
            },
            "lean_reason": {"type": "string", "maxLength": 240},
        },
        "required": ["news_lean", "lean_reason"],
    }


def describe_market(game: Game, probability: WinProbability | None) -> str:
    """Plain language only. Guardrail #1 — no American odds reach the model."""
    if probability is None:
        return "No betting line has been posted for this game yet."

    lines = [f"Implied win probability: {plain_language(probability, game)}"]

    odds = game.odds
    if odds and odds.home_spread is not None:
        if odds.home_spread < 0:
            lines.append(
                f"Spread: {game.home.display_name} favoured by {abs(odds.home_spread)}"
            )
        elif odds.home_spread > 0:
            lines.append(
                f"Spread: {game.away.display_name} favoured by {odds.home_spread}"
            )
        else:
            lines.append("Spread: pick'em")
    if odds and odds.over_under is not None:
        lines.append(f"Expected combined points: {odds.over_under}")
    if probability.method == "spread":
        lines.append("(Probability estimated from the spread; no moneyline posted.)")
    return "\n".join(lines)


def _common_fields(game: Game, home_brief: TeamBrief, away_brief: TeamBrief) -> dict:
    return {
        "away_team": game.away.display_name,
        "home_team": game.home.display_name,
        # Local time, not UTC: a 00:20 UTC kickoff is the previous evening in
        # Toronto, so UTC told the model the wrong day of week — which matters
        # for rest days and short weeks.
        "kickoff": f"{config.to_local(game.kickoff):%A %d %B %Y, %H:%M %Z}",
        "venue": f"{game.venue or 'unknown'}{' (indoor)' if game.indoor else ''}",
        "away_momentum": away_brief.momentum or "unknown",
        "home_momentum": home_brief.momentum or "unknown",
        "away_storylines": _bullets(away_brief.storylines),
        "home_storylines": _bullets(home_brief.storylines),
        "away_injuries": _bullets(away_brief.injuries, empty="- None reported."),
        "home_injuries": _bullets(home_brief.injuries, empty="- None reported."),
    }


def build_lean_prompt(
    game: Game, home_brief: TeamBrief, away_brief: TeamBrief
) -> str:
    """Pass 1. Contains no betting information of any kind — that isolation is
    the whole point: with the market visible the model agreed with it 16/16."""
    return LEAN_PROMPT.format(**_common_fields(game, home_brief, away_brief))


def build_prompt(
    game: Game,
    probability: WinProbability | None,
    home_brief: TeamBrief,
    away_brief: TeamBrief,
    news_lean: str | None = None,
    lean_reason: str = "",
) -> str:
    if news_lean:
        lean_text = f"{news_lean}. {lean_reason}".strip()
    else:
        lean_text = "No independent read was available."
    return PROMPT.format(
        market=describe_market(game, probability),
        news_lean=lean_text,
        **_common_fields(game, home_brief, away_brief),
    )


def _bullets(items: list[str], empty: str = "- No notable developments.") -> str:
    return "\n".join(f"- {i}" for i in items) if items else empty


def build_thesis(
    game: Game,
    probability: WinProbability | None,
    home_brief: TeamBrief,
    away_brief: TeamBrief,
    *,
    refresh: bool = False,
) -> Thesis:
    valid = {game.home.display_name, game.away.display_name}

    def validator(data: dict) -> None:
        # Redundant given the enum, but a cheap backstop if the schema is ever
        # loosened or the server stops constraining.
        if data.get("pick") not in valid:
            raise ValueError(
                f"pick must be exactly one of: {' | '.join(sorted(valid))}"
            )
        if not str(data.get("thesis", "")).strip():
            raise ValueError("thesis must not be empty")

    # Pass 1: news only, no betting line in context at all.
    news_lean, lean_reason = _news_lean(
        game, home_brief, away_brief, refresh=refresh
    )

    # Pass 2: the market, plus the model's own blind read fed back to it.
    prompt = build_prompt(
        game, probability, home_brief, away_brief, news_lean, lean_reason
    )
    try:
        result = llm.chat_json_cached(
            prompt,
            build_schema(game),
            system=SYSTEM,
            validator=validator,
            refresh=refresh,
            num_predict=config.NUM_PREDICT_THESIS,
        )
    except llm.LLMError:
        return _fallback_thesis(game, probability, news_lean, lean_reason)

    data = result.data
    pick = str(data.get("pick", ""))
    return Thesis(
        game_id=game.espn_id,
        pick=pick if pick in valid else None,
        confidence=confidence_label(probability),  # derived, not model-reported
        thesis=str(data.get("thesis", "")),
        key_factors=[str(x) for x in data.get("key_factors", [])],
        news_vs_market=str(data.get("news_vs_market", "")),
        is_upset_call=is_upset_call(game, probability, pick),
        news_lean=news_lean,
        lean_reason=lean_reason,
    )


def _news_lean(
    game: Game, home_brief: TeamBrief, away_brief: TeamBrief, *, refresh: bool = False
) -> tuple[str | None, str]:
    """Pass 1: which team the news favours, decided with no market in context."""
    valid = {game.home.display_name, game.away.display_name}
    try:
        result = llm.chat_json_cached(
            build_lean_prompt(game, home_brief, away_brief),
            build_lean_schema(game),
            system=LEAN_SYSTEM,
            refresh=refresh,
            num_predict=config.NUM_PREDICT_LEAN,
        )
    except llm.LLMError:
        return None, ""
    lean = str(result.data.get("news_lean", ""))
    return (lean if lean in valid else None), str(result.data.get("lean_reason", ""))


def is_upset_call(
    game: Game, probability: WinProbability | None, pick: str | None
) -> bool:
    """True when the model picked against the de-vigged market favourite.

    Computed here, never asked of the model. A pick'em has no favourite, so it
    can never be an upset.
    """
    if probability is None or pick is None or probability.favorite is None:
        return False
    favourite = (
        game.home.display_name
        if probability.favorite == "home"
        else game.away.display_name
    )
    return pick != favourite


def _fallback_thesis(
    game: Game,
    probability: WinProbability | None,
    news_lean: str | None = None,
    lean_reason: str = "",
) -> Thesis:
    """The model failed. Record that honestly rather than inventing a verdict."""
    return Thesis(
        game_id=game.espn_id,
        pick=None,
        confidence=confidence_label(probability),
        thesis="No thesis available — the model did not return a valid response.",
        key_factors=[],
        news_vs_market="",
        is_upset_call=False,
        news_lean=news_lean,
        lean_reason=lean_reason,
    )
