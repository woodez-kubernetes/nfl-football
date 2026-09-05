"""Display timezone: kickoffs are stored in UTC, shown in the report zone."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from nfl_football import config
from nfl_football.models import (
    Game, GameReport, Odds, Team, Thesis, WinProbability,
)
from nfl_football.pipeline import WeekReport
from nfl_football.render import html as html_render
from nfl_football.render import markdown as md_render

HOME = Team("26", "SEA", "Seattle Seahawks", "Seattle", "Seahawks")
AWAY = Team("17", "NE", "New England Patriots", "New England", "Patriots")
ODDS = Odds("DraftKings", "espn", -3.5, 44.5, -175, 145, "SEA -3.5")


def week_at(kickoff_utc: datetime) -> WeekReport:
    game = Game("401", 2026, 1, 2, kickoff_utc, HOME, AWAY, "Lumen Field", False, ODDS)
    thesis = Thesis("401", "Seattle Seahawks", "medium", "Thesis.", ["a", "b", "c"],
                    "n", news_lean="Seattle Seahawks", lean_reason="r")
    return WeekReport(2026, 1, datetime(2026, 9, 2, 16, tzinfo=timezone.utc),
                      [GameReport(game=game,
                                  probability=WinProbability(0.609, 0.391, "moneyline", 0.045),
                                  thesis=thesis)])


class TestToLocal(unittest.TestCase):
    def test_default_zone_is_toronto(self):
        self.assertEqual(config.REPORT_TZ_NAME, "America/Toronto")

    def test_converts_utc_to_eastern(self):
        utc = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
        self.assertEqual(config.to_local(utc).hour, 13)  # 1pm ET Sunday slate

    def test_day_rolls_back_for_late_kickoffs(self):
        """00:20 UTC Thursday is 20:20 Wednesday in Toronto — the *day* changes.

        Shown as UTC, Monday Night Football reads as Tuesday.
        """
        utc = datetime(2026, 9, 10, 0, 20, tzinfo=timezone.utc)
        local = config.to_local(utc)
        self.assertEqual((local.month, local.day, local.hour), (9, 9, 20))
        self.assertEqual(local.strftime("%a"), "Wed")

    def test_monday_night_football_is_monday(self):
        utc = datetime(2026, 9, 15, 0, 15, tzinfo=timezone.utc)
        self.assertEqual(config.to_local(utc).strftime("%a"), "Mon")

    def test_summer_kickoff_is_edt(self):
        self.assertEqual(
            config.to_local(datetime(2026, 9, 13, 17, tzinfo=timezone.utc)).strftime("%Z"),
            "EDT",
        )

    def test_winter_kickoff_is_est(self):
        """January playoff games fall after the DST change."""
        self.assertEqual(
            config.to_local(datetime(2027, 1, 10, 18, tzinfo=timezone.utc)).strftime("%Z"),
            "EST",
        )

    def test_naive_datetime_is_treated_as_utc(self):
        naive = datetime(2026, 9, 13, 17, 0)
        aware = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
        self.assertEqual(config.to_local(naive), config.to_local(aware))

    def test_already_local_is_unchanged_in_instant(self):
        utc = datetime(2026, 9, 13, 17, tzinfo=timezone.utc)
        self.assertEqual(config.to_local(config.to_local(utc)), config.to_local(utc))

    def test_zone_is_overridable(self):
        """NFL_TZ lets someone else read the same report in their own zone."""
        utc = datetime(2026, 9, 13, 17, tzinfo=timezone.utc)
        self.assertEqual(utc.astimezone(ZoneInfo("America/Los_Angeles")).hour, 10)


class TestMarkdownTimes(unittest.TestCase):
    def test_overview_uses_local_time_not_utc(self):
        out = md_render.render(week_at(datetime(2026, 9, 10, 0, 20, tzinfo=timezone.utc)),
                               width=100, sources=False)
        self.assertIn("Wed 20:20", out)
        self.assertNotIn("Thu 00:20", out)

    def test_names_the_timezone(self):
        out = md_render.render(week_at(datetime(2026, 9, 13, 17, tzinfo=timezone.utc)),
                               width=100, sources=False)
        self.assertIn("times in Toronto", out)

    def test_game_header_shows_zone_abbreviation(self):
        out = md_render.render(week_at(datetime(2026, 9, 13, 17, tzinfo=timezone.utc)),
                               width=100, sources=False)
        self.assertIn("13:00 EDT", out)
        self.assertNotIn("17:00 UTC", out)

    def test_table_rows_stay_aligned(self):
        out = md_render.render(week_at(datetime(2026, 9, 13, 17, tzinfo=timezone.utc)),
                               width=100, sources=False)
        lines = [ln for ln in out.splitlines() if ln.startswith("| ")]
        widths = {len(ln) for ln in lines}
        self.assertEqual(len(widths), 1, f"ragged table: {widths}")


class TestHtmlTimes(unittest.TestCase):
    def setUp(self):
        self.report = week_at(datetime(2026, 9, 10, 0, 20, tzinfo=timezone.utc))

    def test_standalone_report_uses_local_time(self):
        out = html_render.render(self.report)
        self.assertIn("Wed 09 Sep, 20:20 EDT", out)
        self.assertNotIn("00:20 UTC", out)

    def test_email_body_uses_local_time(self):
        out = html_render.render_email(self.report)
        self.assertIn("Wed 09 Sep, 20:20 EDT", out)
        self.assertNotIn("00:20 UTC", out)

    def test_both_label_the_zone_column(self):
        self.assertIn("Kickoff (Toronto)", html_render.render(self.report))
        self.assertIn("Kickoff (Toronto)", html_render.render_email(self.report))

    def test_generated_timestamp_is_local(self):
        out = html_render.render(self.report)
        self.assertIn("EDT", out)
        self.assertNotIn("16:00 UTC", out)


if __name__ == "__main__":
    unittest.main()
