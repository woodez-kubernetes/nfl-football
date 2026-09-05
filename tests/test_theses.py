"""Stage 5: game theses and the upset-call logic. Mocked LLM."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from nfl_football import llm, theses
from nfl_football.models import Game, Odds, Team, TeamBrief, WinProbability

HOME = Team("26", "SEA", "Seattle Seahawks", "Seattle", "Seahawks")
AWAY = Team("17", "NE", "New England Patriots", "New England", "Patriots")

ODDS = Odds(provider="DraftKings", source="espn", home_spread=-3.5,
            over_under=44.5, home_moneyline=-175, away_moneyline=145,
            details="SEA -3.5")

GAME = Game(espn_id="401", season=2026, week=1, season_type=2,
            kickoff=datetime(2026, 9, 10, 0, 20, tzinfo=timezone.utc),
            home=HOME, away=AWAY, venue="Lumen Field", indoor=False, odds=ODDS)

# Seattle favoured.
PROB = WinProbability(home=0.609, away=0.391, method="moneyline", overround=0.045)

HOME_BRIEF = TeamBrief("SEA", injuries=["Tory Horton (WR) — Questionable"],
                       storylines=["Roster reshuffled"], momentum="Steady")
AWAY_BRIEF = TeamBrief("NE", injuries=["Milton Williams (DT) — Out"],
                       storylines=["New scheme installed"], momentum="Rebuilding")

GOOD = {
    "pick": "Seattle Seahawks",
    "thesis": "Seattle's home form carries it.",
    "key_factors": ["Home field", "Depth at receiver", "New England's new scheme"],
    "news_vs_market": "News supports the market.",
}

LEAN = {"news_lean": "Seattle Seahawks", "lean_reason": "Healthier roster."}


def fake_result(data):
    return llm.LLMResult(data=data, prompt_tokens=200, eval_tokens=100,
                         duration=1.0, attempts=1)


def two_pass(lean=LEAN, final=GOOD):
    """side_effect for the two chat_json_cached calls: lean, then thesis."""
    return [fake_result(lean), fake_result(final)]


class TestSchema(unittest.TestCase):
    def test_pick_enum_is_exactly_the_two_teams(self):
        """Guardrail #2: constrained decoding makes a bad pick impossible."""
        schema = theses.build_schema(GAME)
        self.assertEqual(sorted(schema["properties"]["pick"]["enum"]),
                         sorted(["Seattle Seahawks", "New England Patriots"]))

    def test_confidence_is_not_asked_of_the_model(self):
        """It came back "high" on all 16 games; it is derived in Python now."""
        self.assertNotIn("confidence", theses.build_schema(GAME)["properties"])

    def test_lean_schema_enum_is_the_two_teams(self):
        schema = theses.build_lean_schema(GAME)
        self.assertEqual(sorted(schema["properties"]["news_lean"]["enum"]),
                         sorted(["Seattle Seahawks", "New England Patriots"]))

    def test_key_factors_bounded(self):
        spec = theses.build_schema(GAME)["properties"]["key_factors"]
        self.assertEqual((spec["minItems"], spec["maxItems"]), (3, 4))


class TestDescribeMarket(unittest.TestCase):
    def test_uses_percentages_not_american_odds(self):
        """Guardrail #1: the notation the model misread must never appear."""
        text = theses.describe_market(GAME, PROB)
        self.assertIn("Seattle Seahawks 61%", text)
        self.assertNotIn("-175", text)
        self.assertNotIn("+145", text)
        self.assertNotIn("145", text)

    def test_home_favourite_reads_naturally(self):
        self.assertIn("Seattle Seahawks favoured by 3.5",
                      theses.describe_market(GAME, PROB))

    def test_away_favourite_reads_naturally(self):
        from dataclasses import replace
        game = replace(GAME, odds=replace(ODDS, home_spread=3.5))
        self.assertIn("New England Patriots favoured by 3.5",
                      theses.describe_market(game, PROB))

    def test_pickem(self):
        from dataclasses import replace
        game = replace(GAME, odds=replace(ODDS, home_spread=0.0))
        self.assertIn("pick'em", theses.describe_market(game, PROB))

    def test_no_line_posted(self):
        text = theses.describe_market(GAME, None)
        self.assertIn("No betting line", text)

    def test_flags_spread_derived_probability(self):
        est = WinProbability(home=0.62, away=0.38, method="spread", overround=None)
        self.assertIn("no moneyline posted", theses.describe_market(GAME, est))


class TestUpsetCall(unittest.TestCase):
    def test_agreeing_with_market_is_not_an_upset(self):
        self.assertFalse(theses.is_upset_call(GAME, PROB, "Seattle Seahawks"))

    def test_picking_the_underdog_is_an_upset(self):
        self.assertTrue(theses.is_upset_call(GAME, PROB, "New England Patriots"))

    def test_away_favourite_case(self):
        prob = WinProbability(home=0.39, away=0.61, method="moneyline", overround=0.04)
        self.assertFalse(theses.is_upset_call(GAME, prob, "New England Patriots"))
        self.assertTrue(theses.is_upset_call(GAME, prob, "Seattle Seahawks"))

    def test_pickem_can_never_be_an_upset(self):
        even = WinProbability(home=0.5, away=0.5, method="moneyline", overround=0.04)
        self.assertFalse(theses.is_upset_call(GAME, even, "Seattle Seahawks"))
        self.assertFalse(theses.is_upset_call(GAME, even, "New England Patriots"))

    def test_no_probability_means_no_upset(self):
        self.assertFalse(theses.is_upset_call(GAME, None, "Seattle Seahawks"))

    def test_failed_pick_is_not_an_upset(self):
        self.assertFalse(theses.is_upset_call(GAME, PROB, None))


class TestConfidence(unittest.TestCase):
    def test_derived_from_market_margin(self):
        from nfl_football.probability import confidence_label
        self.assertEqual(confidence_label(WinProbability(0.81, 0.19, "moneyline")), "high")
        self.assertEqual(confidence_label(WinProbability(0.61, 0.39, "moneyline")), "medium")
        self.assertEqual(confidence_label(WinProbability(0.52, 0.48, "moneyline")), "low")
        self.assertEqual(confidence_label(None), "unknown")

    def test_symmetric_for_away_favourite(self):
        from nfl_football.probability import confidence_label
        self.assertEqual(confidence_label(WinProbability(0.19, 0.81, "moneyline")), "high")


class TestNewsLeanPass(unittest.TestCase):
    def test_lean_prompt_contains_no_betting_information(self):
        """The isolation is the point: with the market visible the model agreed
        with it in 16/16 games."""
        prompt = theses.build_lean_prompt(GAME, HOME_BRIEF, AWAY_BRIEF)
        for token in ("spread", "Spread", "probability", "favoured", "market",
                      "44.5", "3.5", "%"):
            self.assertNotIn(token, prompt, f"leaked {token!r} into the blind pass")

    def test_lean_prompt_still_has_the_news(self):
        prompt = theses.build_lean_prompt(GAME, HOME_BRIEF, AWAY_BRIEF)
        self.assertIn("Roster reshuffled", prompt)
        self.assertIn("New scheme installed", prompt)

    def test_lean_is_fed_into_the_second_pass(self):
        prompts = []

        def capture(prompt, schema, **kwargs):
            prompts.append(prompt)
            return fake_result(LEAN if len(prompts) == 1 else GOOD)

        with patch.object(llm, "chat_json_cached", side_effect=capture):
            theses.build_thesis(GAME, PROB, HOME_BRIEF, AWAY_BRIEF)

        self.assertEqual(len(prompts), 2)
        self.assertIn("Healthier roster.", prompts[1])
        self.assertIn("before you saw any betting line", prompts[1])

    def test_lean_failure_does_not_block_the_thesis(self):
        with patch.object(llm, "chat_json_cached",
                          side_effect=[llm.LLMError("down"), fake_result(GOOD)]):
            t = theses.build_thesis(GAME, PROB, HOME_BRIEF, AWAY_BRIEF)
        self.assertIsNone(t.news_lean)
        self.assertEqual(t.pick, "Seattle Seahawks")


class TestBuildThesis(unittest.TestCase):
    def test_happy_path(self):
        with patch.object(llm, "chat_json_cached", side_effect=two_pass()):
            t = theses.build_thesis(GAME, PROB, HOME_BRIEF, AWAY_BRIEF)
        self.assertEqual(t.pick, "Seattle Seahawks")
        self.assertEqual(t.game_id, "401")
        self.assertEqual(len(t.key_factors), 3)
        self.assertFalse(t.is_upset_call)
        self.assertEqual(t.news_lean, "Seattle Seahawks")
        self.assertEqual(t.confidence, "medium")  # derived from 0.609

    def test_upset_call_computed_not_reported(self):
        """Even if the model claims agreement, Python decides."""
        underdog = dict(GOOD, pick="New England Patriots",
                        news_vs_market="News agrees with the market.")
        with patch.object(llm, "chat_json_cached", side_effect=two_pass(final=underdog)):
            t = theses.build_thesis(GAME, PROB, HOME_BRIEF, AWAY_BRIEF)
        self.assertTrue(t.is_upset_call)

    def test_lean_differs_from_pick_is_surfaced(self):
        lean = {"news_lean": "New England Patriots", "lean_reason": "Fewer injuries."}
        with patch.object(llm, "chat_json_cached", side_effect=two_pass(lean=lean)):
            t = theses.build_thesis(GAME, PROB, HOME_BRIEF, AWAY_BRIEF)
        self.assertTrue(t.lean_differs_from_pick)

    def test_llm_failure_records_no_pick(self):
        with patch.object(llm, "chat_json_cached", side_effect=llm.LLMError("down")):
            t = theses.build_thesis(GAME, PROB, HOME_BRIEF, AWAY_BRIEF)
        self.assertIsNone(t.pick)
        self.assertIn("No thesis available", t.thesis)
        self.assertFalse(t.is_upset_call)

    def test_out_of_vocabulary_pick_is_rejected(self):
        """Backstop if constrained decoding is ever bypassed."""
        with patch.object(llm, "chat_json_cached",
                          side_effect=two_pass(final=dict(GOOD, pick="Denver Broncos"))):
            t = theses.build_thesis(GAME, PROB, HOME_BRIEF, AWAY_BRIEF)
        self.assertIsNone(t.pick)

    def test_out_of_vocabulary_lean_is_rejected(self):
        bad = {"news_lean": "Denver Broncos", "lean_reason": "x"}
        with patch.object(llm, "chat_json_cached", side_effect=two_pass(lean=bad)):
            t = theses.build_thesis(GAME, PROB, HOME_BRIEF, AWAY_BRIEF)
        self.assertIsNone(t.news_lean)

    def test_validator_rejects_bad_pick_and_empty_thesis(self):
        captured = {}

        def capture(prompt, schema, **kwargs):
            if "validator" in kwargs:
                captured["validator"] = kwargs["validator"]
                return fake_result(GOOD)
            return fake_result(LEAN)

        with patch.object(llm, "chat_json_cached", side_effect=capture):
            theses.build_thesis(GAME, PROB, HOME_BRIEF, AWAY_BRIEF)

        validator = captured["validator"]
        validator(GOOD)  # valid, no raise
        with self.assertRaises(ValueError):
            validator({"pick": "Denver Broncos", "thesis": "x"})
        with self.assertRaises(ValueError):
            validator({"pick": "Seattle Seahawks", "thesis": "   "})

    def test_prompt_includes_both_briefs(self):
        prompt = theses.build_prompt(GAME, PROB, HOME_BRIEF, AWAY_BRIEF)
        self.assertIn("Roster reshuffled", prompt)
        self.assertIn("New scheme installed", prompt)
        self.assertIn("Milton Williams (DT) — Out", prompt)
        self.assertIn("Lumen Field", prompt)

    def test_prompt_permits_disagreement(self):
        prompt = theses.build_prompt(GAME, PROB, HOME_BRIEF, AWAY_BRIEF)
        self.assertIn("not required to agree with the market", prompt)
        self.assertIn("do not default to the market", prompt)

    def test_empty_briefs_do_not_crash(self):
        empty = TeamBrief("SEA")
        prompt = theses.build_prompt(GAME, PROB, empty, empty)
        self.assertIn("No notable developments", prompt)
        self.assertIn("None reported", prompt)


if __name__ == "__main__":
    unittest.main()
