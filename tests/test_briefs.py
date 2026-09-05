"""Stage 4: team briefs. Mocked LLM — no host required."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from nfl_football import briefs, llm
from nfl_football.models import Injury, NewsItem, Team

FIXTURES = Path(__file__).parent / "fixtures"

TEAM = Team(
    espn_id="12", abbreviation="KC", display_name="Kansas City Chiefs",
    location="Kansas City", nickname="Chiefs",
)

NEWS = [
    NewsItem("Chiefs sign veteran guard", "ESPN", "http://x/1",
             datetime(2026, 9, 1, tzinfo=timezone.utc)),
    NewsItem("Sneed embraces nickel role", "kansascity.com", "http://x/2",
             datetime(2026, 8, 31, tzinfo=timezone.utc)),
]

INJURIES = [
    Injury("Patrick Mahomes", "QB", "Questionable", "limited in practice"),
    Injury("Chris Jones", "DT", "Out", "ankle"),
]


def fake_result(data: dict) -> llm.LLMResult:
    return llm.LLMResult(data=data, prompt_tokens=100, eval_tokens=50,
                         duration=1.0, attempts=1)


GOOD = {"storylines": ["Signed a veteran guard", "Sneed moves to nickel",
                       "Practice squad reshuffled"],
        "momentum": "Roster churn ahead of the opener."}


class TestBuildPrompt(unittest.TestCase):
    def test_contains_headlines_and_team(self):
        prompt = briefs.build_prompt(TEAM, NEWS, INJURIES)
        self.assertIn("Kansas City Chiefs", prompt)
        self.assertIn("Chiefs sign veteran guard", prompt)
        self.assertIn("ESPN", prompt)

    def test_headlines_precede_injuries(self):
        """Ordering is deliberate: injuries first caused the model to write
        injury restatements instead of news storylines."""
        prompt = briefs.build_prompt(TEAM, NEWS, INJURIES)
        self.assertLess(prompt.index("Chiefs sign veteran guard"),
                        prompt.index("Patrick Mahomes"))

    def test_instructs_against_injury_storylines(self):
        prompt = briefs.build_prompt(TEAM, NEWS, INJURIES)
        self.assertIn("NOT be summarised back", prompt)

    def test_handles_no_news(self):
        prompt = briefs.build_prompt(TEAM, [], INJURIES)
        self.assertIn("No headlines found", prompt)

    def test_handles_no_injuries(self):
        prompt = briefs.build_prompt(TEAM, NEWS, [])
        self.assertIn("No injuries reported", prompt)

    def test_undated_headline_does_not_crash(self):
        undated = [NewsItem("Trade rumours swirl", "AP", "http://x/3", None)]
        self.assertIn("Trade rumours swirl", briefs.build_prompt(TEAM, undated, []))


class TestSchema(unittest.TestCase):
    def test_model_is_not_asked_for_injuries(self):
        """Injuries come from ESPN; routing them through the model wasted
        generation and crowded out the news."""
        self.assertNotIn("injuries", briefs.SCHEMA["properties"])
        self.assertNotIn("sentiment", briefs.SCHEMA["properties"])

    def test_storylines_bounded(self):
        spec = briefs.SCHEMA["properties"]["storylines"]
        self.assertEqual(spec["minItems"], 3)
        self.assertEqual(spec["maxItems"], 4)
        self.assertIn("maxLength", spec["items"])


class TestBuildTeamBrief(unittest.TestCase):
    def test_happy_path(self):
        with patch.object(llm, "chat_json_cached", return_value=fake_result(GOOD)):
            brief = briefs.build_team_brief(TEAM, NEWS, INJURIES)
        self.assertEqual(brief.team_abbr, "KC")
        self.assertEqual(len(brief.storylines), 3)
        self.assertEqual(brief.momentum, "Roster churn ahead of the opener.")

    def test_injuries_bypass_the_model(self):
        """Even if the model returns injuries, ESPN's list is what survives."""
        polluted = dict(GOOD, injuries=["Made up player — Out"])
        with patch.object(llm, "chat_json_cached", return_value=fake_result(polluted)):
            brief = briefs.build_team_brief(TEAM, NEWS, INJURIES)
        self.assertNotIn("Made up player — Out", brief.injuries)
        self.assertTrue(any("Chris Jones" in i for i in brief.injuries))

    def test_injuries_ordered_most_serious_first(self):
        with patch.object(llm, "chat_json_cached", return_value=fake_result(GOOD)):
            brief = briefs.build_team_brief(TEAM, NEWS, INJURIES)
        self.assertIn("Chris Jones", brief.injuries[0])  # Out before Questionable

    def test_llm_failure_falls_back_without_losing_facts(self):
        with patch.object(llm, "chat_json_cached", side_effect=llm.LLMError("down")):
            brief = briefs.build_team_brief(TEAM, NEWS, INJURIES)
        self.assertEqual(brief.storylines, [])
        self.assertEqual(brief.momentum, "")
        self.assertTrue(brief.injuries)  # source facts survive

    def test_non_string_storylines_coerced(self):
        with patch.object(llm, "chat_json_cached",
                          return_value=fake_result({"storylines": [1, 2, 3], "momentum": 7})):
            brief = briefs.build_team_brief(TEAM, NEWS, INJURIES)
        self.assertTrue(all(isinstance(s, str) for s in brief.storylines))
        self.assertIsInstance(brief.momentum, str)


class TestResponseCaching(unittest.TestCase):
    """chat_json_cached is content-addressed: same prompt hits, changed prompt misses."""

    def setUp(self):
        import tempfile
        from nfl_football.sources import cache as cache_mod
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = cache_mod.Cache(Path(self.tmp.name) / "c.db")
        self._patch = patch.object(cache_mod, "default_cache", return_value=self.cache)
        self._patch.start()
        llm.STATS = llm.Stats()

    def tearDown(self):
        self._patch.stop()
        self.cache.close()
        self.tmp.cleanup()

    def test_second_identical_call_is_served_from_cache(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}},
                  "required": ["a"]}
        with patch.object(llm, "chat_json", return_value=fake_result({"a": "x"})) as m:
            llm.chat_json_cached("same prompt", schema)
            llm.chat_json_cached("same prompt", schema)
        self.assertEqual(m.call_count, 1)
        self.assertEqual(llm.STATS.cached, 1)

    def test_changed_prompt_misses(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}},
                  "required": ["a"]}
        with patch.object(llm, "chat_json", return_value=fake_result({"a": "x"})) as m:
            llm.chat_json_cached("prompt one", schema)
            llm.chat_json_cached("prompt two", schema)
        self.assertEqual(m.call_count, 2)

    def test_refresh_bypasses_cache(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}},
                  "required": ["a"]}
        with patch.object(llm, "chat_json", return_value=fake_result({"a": "x"})) as m:
            llm.chat_json_cached("same", schema)
            llm.chat_json_cached("same", schema, refresh=True)
        self.assertEqual(m.call_count, 2)


if __name__ == "__main__":
    unittest.main()
