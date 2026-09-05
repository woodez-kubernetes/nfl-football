"""Stage 2: probability engine. Hand-checked arithmetic, no network."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from nfl_football import probability as prob
from nfl_football.models import Odds
from nfl_football.sources import schedule

FIXTURES = Path(__file__).parent / "fixtures"


class TestImpliedProbability(unittest.TestCase):
    def test_negative_moneyline(self):
        # -175 risks 175 to win 100 -> 175/275
        self.assertAlmostEqual(prob.american_to_implied(-175), 175 / 275, places=10)
        self.assertAlmostEqual(prob.american_to_implied(-175), 0.6363636364, places=9)

    def test_positive_moneyline(self):
        # +145 risks 100 to win 145 -> 100/245
        self.assertAlmostEqual(prob.american_to_implied(145), 100 / 245, places=10)
        self.assertAlmostEqual(prob.american_to_implied(145), 0.4081632653, places=9)

    def test_standard_juice(self):
        self.assertAlmostEqual(prob.american_to_implied(-110), 110 / 210, places=10)

    def test_even_money(self):
        self.assertAlmostEqual(prob.american_to_implied(100), 0.5, places=10)

    def test_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            prob.american_to_implied(0)

    def test_longshot_and_lock_stay_in_range(self):
        for ml in (-10000, -550, -101, 101, 410, 10000):
            p = prob.american_to_implied(ml)
            self.assertGreater(p, 0.0)
            self.assertLess(p, 1.0)


class TestDevig(unittest.TestCase):
    def test_pickem_is_exactly_even(self):
        home, away, overround = prob.devig(
            prob.american_to_implied(-110), prob.american_to_implied(-110)
        )
        self.assertAlmostEqual(home, 0.5, places=12)
        self.assertAlmostEqual(away, 0.5, places=12)
        # Two -110 sides imply 104.76% -> 4.76% overround.
        self.assertAlmostEqual(overround, 2 * (110 / 210) - 1, places=12)

    def test_always_sums_to_one(self):
        for pair in [(-175, 145), (-550, 410), (-102, -118), (240, -298)]:
            home, away, _ = prob.devig(
                prob.american_to_implied(pair[0]), prob.american_to_implied(pair[1])
            )
            self.assertAlmostEqual(home + away, 1.0, places=12, msg=str(pair))

    def test_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            prob.devig(0.0, 0.0)


class TestFromMoneyline(unittest.TestCase):
    def test_known_game(self):
        """2026 Wk1 NE@SEA: SEA -175, NE +145."""
        p = prob.from_moneyline(-175, 145)
        expected = (175 / 275) / ((175 / 275) + (100 / 245))
        self.assertAlmostEqual(p.home, expected, places=12)
        self.assertAlmostEqual(p.home, 0.6092, places=4)
        self.assertAlmostEqual(p.home + p.away, 1.0, places=12)
        self.assertEqual(p.method, "moneyline")
        self.assertEqual(p.favorite, "home")

    def test_heavy_favourite(self):
        """ARI@LAC: LAC -550, ARI +410."""
        p = prob.from_moneyline(-550, 410)
        self.assertAlmostEqual(p.home, 0.8119, places=3)
        self.assertEqual(p.favorite, "home")

    def test_away_favourite(self):
        """BAL@IND: home IND +145, away BAL -175."""
        p = prob.from_moneyline(145, -175)
        self.assertAlmostEqual(p.away, 0.6092, places=4)
        self.assertEqual(p.favorite, "away")

    def test_overround_reported(self):
        p = prob.from_moneyline(-175, 145)
        self.assertGreater(p.overround, 0.0)
        self.assertLess(p.overround, 0.2)

    def test_near_coinflip_has_no_runaway_favourite(self):
        p = prob.from_moneyline(-102, -118)  # GB@MIN
        self.assertLess(abs(p.home - p.away), 0.06)


class TestFromSpread(unittest.TestCase):
    def test_pickem_is_even(self):
        p = prob.from_spread(0.0)
        self.assertAlmostEqual(p.home, 0.5, places=12)
        self.assertIsNone(p.favorite)
        self.assertIsNone(p.overround)
        self.assertEqual(p.method, "spread")

    def test_home_favourite_is_negative_spread(self):
        """ESPN convention: -3.5 means the home team is favoured."""
        p = prob.from_spread(-3.5)
        self.assertGreater(p.home, 0.5)
        self.assertEqual(p.favorite, "home")
        expected = 1 / (1 + math.exp(-prob.SPREAD_LOGIT_PER_POINT * 3.5))
        self.assertAlmostEqual(p.home, expected, places=12)
        self.assertAlmostEqual(p.home, 0.6195, places=3)

    def test_away_favourite_is_positive_spread(self):
        p = prob.from_spread(3.5)
        self.assertEqual(p.favorite, "away")
        self.assertAlmostEqual(p.away, 0.6195, places=3)

    def test_symmetry(self):
        for s in (1.0, 3.5, 7.0, 10.5, 14.0):
            self.assertAlmostEqual(
                prob.from_spread(-s).home, prob.from_spread(s).away, places=12
            )

    def test_monotonic_in_spread(self):
        probs = [prob.from_spread(-s).home for s in range(0, 21)]
        self.assertEqual(probs, sorted(probs))

    def test_always_sums_to_one(self):
        for s in (-14.0, -3.5, 0.0, 2.5, 17.0):
            p = prob.from_spread(s)
            self.assertAlmostEqual(p.home + p.away, 1.0, places=12)

    def test_tracks_the_moneyline_path(self):
        """The fallback should land near the market it stands in for.

        NE@SEA: spread -3.5 vs moneyline -175/+145.
        """
        by_spread = prob.from_spread(-3.5).home
        by_moneyline = prob.from_moneyline(-175, 145).home
        self.assertLess(abs(by_spread - by_moneyline), 0.02)


class TestDispatch(unittest.TestCase):
    def test_prefers_moneyline(self):
        odds = Odds(provider="p", source="espn", home_spread=-3.5,
                    home_moneyline=-175, away_moneyline=145)
        self.assertEqual(prob.win_probability(odds).method, "moneyline")

    def test_falls_back_to_spread(self):
        odds = Odds(provider="p", source="espn", home_spread=-3.5)
        self.assertEqual(prob.win_probability(odds).method, "spread")

    def test_none_when_nothing_posted(self):
        self.assertIsNone(prob.win_probability(None))
        self.assertIsNone(prob.win_probability(Odds(provider="p", source="espn")))

    def test_partial_moneyline_falls_back(self):
        odds = Odds(provider="p", source="espn", home_spread=-3.5, home_moneyline=-175)
        self.assertEqual(prob.win_probability(odds).method, "spread")

    def test_malformed_price_falls_back_to_spread(self):
        odds = Odds(provider="p", source="espn", home_spread=-7.0,
                    home_moneyline=0, away_moneyline=0)
        self.assertEqual(prob.win_probability(odds).method, "spread")


class TestAgainstRealSlate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURES / "espn_scoreboard_2026_wk1.json") as fh:
            cls.games = schedule.parse_scoreboard(json.load(fh))

    def test_every_game_gets_a_probability(self):
        for g in self.games:
            p = prob.win_probability(g.odds)
            self.assertIsNotNone(p, g.matchup)
            self.assertAlmostEqual(p.home + p.away, 1.0, places=12, msg=g.matchup)

    def test_favourite_agrees_with_spread_sign(self):
        """The de-vigged moneyline favourite must match the side the spread favours."""
        for g in self.games:
            p = prob.win_probability(g.odds)
            if g.odds.home_spread == 0:
                continue
            expected = "home" if g.odds.home_spread < 0 else "away"
            self.assertEqual(p.favorite, expected, g.matchup)

    def test_plain_language_has_no_american_odds(self):
        """The LLM must never see "-175" — that is the notation it misreads."""
        g = next(g for g in self.games if g.matchup == "NE@SEA")
        text = prob.plain_language(prob.win_probability(g.odds), g)
        self.assertIn("Seattle Seahawks 61%", text)
        self.assertIn("New England Patriots 39%", text)
        self.assertNotIn("-175", text)
        self.assertNotIn("+145", text)


if __name__ == "__main__":
    unittest.main()
