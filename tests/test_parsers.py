"""Stage 1: source parser tests. All offline, driven by tests/fixtures/."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from nfl_football import config
from nfl_football.sources import injuries, news, nflverse, schedule
from nfl_football.sources.cache import Cache
from nfl_football.teams import to_espn, to_nflverse

FIXTURES = Path(__file__).parent / "fixtures"


def load_games():
    with open(FIXTURES / "espn_scoreboard_2026_wk1.json") as fh:
        return schedule.parse_scoreboard(json.load(fh))


class TestSchedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = load_games()

    def test_parses_full_slate(self):
        self.assertEqual(len(self.games), 16)
        self.assertTrue(all(g.season == 2026 and g.week == 1 for g in self.games))

    def test_every_game_has_odds_with_moneyline(self):
        for g in self.games:
            self.assertIsNotNone(g.odds, g.matchup)
            self.assertTrue(g.odds.has_moneyline, g.matchup)

    def test_kickoffs_tz_aware_and_sorted(self):
        self.assertTrue(all(g.kickoff.tzinfo is not None for g in self.games))
        self.assertEqual(self.games, sorted(self.games, key=lambda g: g.kickoff))

    def test_home_favourite_spread_is_negative(self):
        """NE@SEA: details "SEA -3.5", SEA is home -> home_spread negative."""
        g = next(g for g in self.games if g.matchup == "NE@SEA")
        self.assertEqual(g.odds.home_spread, -3.5)
        self.assertEqual(g.odds.home_moneyline, -175)
        self.assertEqual(g.odds.away_moneyline, 145)

    def test_away_favourite_spread_is_positive(self):
        """BAL@IND: details "BAL -3.5", BAL is away -> home_spread positive.

        This is the case that distinguishes the ESPN and nflverse conventions.
        """
        g = next(g for g in self.games if g.matchup == "BAL@IND")
        self.assertEqual(g.odds.home_spread, 3.5)
        self.assertEqual(g.odds.away_moneyline, -175)
        self.assertTrue(g.indoor)

    def test_missing_odds_is_not_an_error(self):
        self.assertIsNone(schedule.parse_odds({}))
        self.assertIsNone(schedule.parse_odds({"odds": []}))

    def test_american_odds_parsing(self):
        cases = {
            "-175": -175, "+145": 145, "145": 145, 145: 145,
            "EVEN": 100, "PK": 100, "OFF": None, "": None, None: None, "junk": None,
        }
        for raw, expected in cases.items():
            self.assertEqual(schedule._as_american(raw), expected, repr(raw))

    def test_line_value_parsing(self):
        self.assertEqual(schedule._as_float("o44.5"), 44.5)
        self.assertEqual(schedule._as_float("u44.5"), 44.5)
        self.assertEqual(schedule._as_float("-3.5"), -3.5)
        self.assertEqual(schedule._as_float(-3.5), -3.5)
        self.assertIsNone(schedule._as_float("junk"))
        self.assertIsNone(schedule._as_float(None))


class TestNflverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (FIXTURES / "nflverse_2026_wk1.csv").read_text()
        cls.table = nflverse.parse_games(cls.raw, 2026, 1)
        cls.games = load_games()

    def test_abbreviations_align_with_espn(self):
        espn_keys = {(g.away.abbreviation, g.home.abbreviation) for g in self.games}
        self.assertEqual(espn_keys - set(self.table), set())

    def test_spread_sign_is_flipped_to_espn_convention(self):
        """nflverse stores NE@SEA as +3.5 (home favoured); we store -3.5."""
        self.assertIn("3.5", self.raw)  # stored positive upstream
        odds = self.table[("NE", "SEA")]
        self.assertEqual(odds.home_spread, -3.5)

    def test_away_favourite_sign_flip(self):
        """nflverse CHI@CAR is -2.5 (away favoured); we store +2.5."""
        odds = self.table[("CHI", "CAR")]
        self.assertEqual(odds.home_spread, 2.5)
        self.assertEqual(odds.away_moneyline, -162)

    def test_agrees_with_espn_on_spreads_and_totals(self):
        for g in self.games:
            n = self.table[(g.away.abbreviation, g.home.abbreviation)]
            self.assertEqual(g.odds.home_spread, n.home_spread, g.matchup)
            self.assertEqual(g.odds.over_under, n.over_under, g.matchup)

    def test_fill_missing_fills_all_when_odds_absent(self):
        stripped = [replace(g, odds=None) for g in self.games]
        filled = nflverse._fill_from_table(stripped, self.table)
        self.assertEqual(filled, 16)
        self.assertTrue(all(g.odds and g.odds.has_moneyline for g in stripped))
        self.assertEqual(stripped[0].odds.source, "nflverse")

    def test_fill_missing_leaves_existing_odds_alone(self):
        half = [replace(g, odds=None) if i % 2 == 0 else g
                for i, g in enumerate(self.games)]
        filled = nflverse._fill_from_table(half, self.table)
        self.assertEqual(filled, 8)
        self.assertEqual(half[1].odds.source, "espn")


class TestTeams(unittest.TestCase):
    def test_known_aliases(self):
        self.assertEqual(to_espn("LA"), "LAR")
        self.assertEqual(to_espn("WAS"), "WSH")
        self.assertEqual(to_nflverse("LAR"), "LA")
        self.assertEqual(to_nflverse("WSH"), "WAS")

    def test_passthrough_and_case(self):
        self.assertEqual(to_espn("kc"), "KC")
        self.assertEqual(to_espn("SEA"), "SEA")
        self.assertEqual(to_espn(""), "")


class TestNews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = news.parse_feed((FIXTURES / "gnews_chiefs.xml").read_text())

    def test_caps_at_limit(self):
        self.assertEqual(len(self.items), config.MAX_NEWS_PER_TEAM)

    def test_publisher_suffix_stripped(self):
        for i in self.items:
            self.assertFalse(i.title.endswith(f" - {i.source}"), i.title)

    def test_sorted_newest_first(self):
        dates = [i.published for i in self.items]
        self.assertEqual(dates, sorted(dates, reverse=True))

    def test_deduplicates_by_normalised_title(self):
        keys = [news._normalise(i.title) for i in self.items]
        self.assertEqual(len(keys), len(set(keys)))

    def test_normalise_ignores_punctuation_and_case(self):
        self.assertEqual(news._normalise("Chiefs' Win!"), news._normalise("chiefs win"))


class TestInjuries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURES / "espn_injuries.json") as fh:
            cls.by_team = injuries.parse_injuries(json.load(fh))

    def test_indexed_by_espn_team_id(self):
        self.assertIn("12", self.by_team)  # Kansas City
        self.assertTrue(self.by_team["12"])

    def test_fields_populated(self):
        entry = self.by_team["12"][0]
        self.assertTrue(entry.player)
        self.assertTrue(entry.status)

    def test_summarise_orders_by_severity(self):
        entries = self.by_team["12"]
        lines = injuries.summarise(entries, limit=100)
        severities = [ln.rsplit("— ", 1)[1].lower() for ln in lines]
        rank = {"out": 0, "injured reserve": 0, "doubtful": 1, "questionable": 2}
        scores = [rank.get(s, 3) for s in severities]
        self.assertEqual(scores, sorted(scores))

    def test_summarise_respects_limit(self):
        self.assertLessEqual(len(injuries.summarise(self.by_team["12"], limit=5)), 5)


class TestCache(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Cache(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.cache.close()
        self.tmp.cleanup()

    def test_roundtrip(self):
        self.cache.put("k", "payload")
        self.assertEqual(self.cache.get("k", ttl=60), "payload")

    def test_miss_on_absent_key(self):
        self.assertIsNone(self.cache.get("nope", ttl=60))

    def test_expired_entry_is_a_miss(self):
        self.cache.put("k", "payload")
        self.assertIsNone(self.cache.get("k", ttl=0))

    def test_stale_still_readable_with_infinite_ttl(self):
        """The fallback path used when a source is down."""
        self.cache.put("k", "payload")
        self.assertEqual(self.cache.get("k", ttl=float("inf")), "payload")


if __name__ == "__main__":
    unittest.main()
