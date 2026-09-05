"""Stage 6: markdown and HTML renderers. No network, no LLM."""

from __future__ import annotations

import re
import unittest
from datetime import datetime, timezone

from nfl_football.models import (
    Game, GameReport, Injury, NewsItem, Odds, Team, TeamBrief, Thesis, WinProbability,
)
from nfl_football.pipeline import WeekReport
from nfl_football.render import html, markdown

HOME = Team("26", "SEA", "Seattle Seahawks", "Seattle", "Seahawks")
AWAY = Team("17", "NE", "New England Patriots", "New England", "Patriots")
ODDS = Odds("DraftKings", "espn", -3.5, 44.5, -175, 145, "SEA -3.5")
GAME = Game("401", 2026, 1, 2, datetime(2026, 9, 10, 0, 20, tzinfo=timezone.utc),
            HOME, AWAY, "Lumen Field", False, ODDS)
PROB = WinProbability(0.609, 0.391, "moneyline", 0.045)

NEWS = [NewsItem("Seahawks reshuffle the line", "ESPN", "https://espn.com/a",
                 datetime(2026, 9, 1, tzinfo=timezone.utc))]


def make_report(thesis=None, probability=PROB, news=True):
    gr = GameReport(
        game=GAME, probability=probability, thesis=thesis,
        home_brief=TeamBrief("SEA", injuries=["Tory Horton (WR) — Questionable"],
                             storylines=["Line reshuffled"], momentum="Steady"),
        away_brief=TeamBrief("NE", injuries=["Milton Williams (DT) — Out"],
                             storylines=["New scheme"], momentum="Rebuilding"),
        home_news=NEWS if news else [], away_news=NEWS if news else [],
        home_injuries=[Injury("Tory Horton", "WR", "Questionable")],
        away_injuries=[Injury("Milton Williams", "DT", "Out")],
    )
    return WeekReport(season=2026, week=1,
                      generated_at=datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc),
                      games=[gr])


AGREE = Thesis("401", "Seattle Seahawks", "medium", "Seattle holds serve at home.",
               ["Home field", "Depth", "Scheme change"], "News supports the market.",
               is_upset_call=False, news_lean="Seattle Seahawks",
               lean_reason="Healthier roster.")

UPSET = Thesis("401", "New England Patriots", "low", "New England's scheme travels.",
               ["Fewer injuries", "New scheme", "Road form"],
               "News contradicts the market.", is_upset_call=True,
               news_lean="New England Patriots", lean_reason="Fewer injuries.")


class TestMarkdown(unittest.TestCase):
    def test_header_and_overview(self):
        out = markdown.render(make_report(AGREE), width=100)
        self.assertIn("# NFL 2026 — Week 1", out)
        self.assertIn("Week at a glance", out)
        self.assertIn("NE@SEA", out)

    def test_shows_pick_market_and_news_lean(self):
        out = markdown.render(make_report(AGREE), width=100)
        self.assertIn("Pick: Seattle Seahawks", out)
        self.assertIn("Market favours: **Seattle Seahawks**", out)
        self.assertIn("News alone favoured: **Seattle Seahawks**", out)

    def test_upset_call_is_flagged(self):
        out = markdown.render(make_report(UPSET), width=100)
        self.assertIn("UPSET CALL", out)
        self.assertIn("Pick: New England Patriots", out)

    def test_agreeing_pick_has_no_upset_flag(self):
        self.assertNotIn("UPSET CALL", markdown.render(make_report(AGREE), width=100))

    def test_probabilities_rendered_as_percentages(self):
        out = markdown.render(make_report(AGREE), width=100)
        self.assertIn("61%", out)
        self.assertIn("39%", out)

    def test_sources_are_linked(self):
        out = markdown.render(make_report(AGREE), width=100)
        self.assertIn("[Seahawks reshuffle the line](https://espn.com/a)", out)

    def test_sources_can_be_suppressed(self):
        out = markdown.render(make_report(AGREE), width=100, sources=False)
        self.assertNotIn("https://espn.com/a", out)

    def test_injuries_listed(self):
        out = markdown.render(make_report(AGREE), width=100)
        self.assertIn("Milton Williams (DT) — Out", out)

    def test_no_line_posted(self):
        out = markdown.render(make_report(AGREE, probability=None), width=100)
        self.assertIn("No betting line posted", out)

    def test_missing_thesis_is_explicit(self):
        out = markdown.render(make_report(None), width=100)
        self.assertIn("No thesis generated", out)

    def test_failed_pick_is_explicit(self):
        failed = Thesis("401", None, "medium", "No thesis available — the model did not return a valid response.")
        out = markdown.render(make_report(failed), width=100)
        self.assertIn("No pick", out)

    def test_summary_counts_upsets(self):
        self.assertIn("1 of 1 picks go against the market",
                      markdown.render(make_report(UPSET), width=100))


class TestHtml(unittest.TestCase):
    def test_renders_and_is_wellformed(self):
        out = html.render(make_report(AGREE))
        self.assertTrue(out.lstrip().startswith("<!doctype html>"))
        self.assertIn("</html>", out)

    def test_self_contained_no_external_assets(self):
        """Must open offline: no scripts, no remote CSS, no remote images."""
        out = html.render(make_report(AGREE))
        self.assertIn("<style>", out)
        self.assertNotIn("<script", out)
        self.assertNotIn("<link", out)
        self.assertNotIn("<img", out)

    def test_shows_pick_market_and_news_lean(self):
        out = html.render(make_report(AGREE))
        self.assertIn("Pick: Seattle Seahawks", out)
        self.assertIn("News alone:", out)

    def test_upset_badge(self):
        self.assertIn("UPSET CALL", html.render(make_report(UPSET)))
        self.assertNotIn("UPSET CALL", html.render(make_report(AGREE)))

    def test_probability_bars_use_percent_widths(self):
        out = html.render(make_report(AGREE))
        self.assertIn("width: 60.9%", out)
        self.assertIn("width: 39.1%", out)

    def test_probability_bars_are_block_level(self):
        """Regression: as inline spans .track/.fill collapse to 0x0 and every
        bar renders empty, even though the percent widths are in the markup.
        Verified in a browser — computed display was `inline`, width 0px."""
        out = html.render(make_report(AGREE))
        css = out[out.index("<style>"):out.index("</style>")]
        fill_rule = re.search(r"\.fill\s*\{[^}]*\}", css).group(0)
        track_rule = re.search(r"\.track\s*\{[^}]*\}", css).group(0)
        self.assertIn("display: block", fill_rule)
        self.assertIn("display: block", track_rule)

    def test_sources_linked_with_noopener(self):
        out = html.render(make_report(AGREE))
        self.assertIn('href="https://espn.com/a"', out)
        self.assertIn("noopener", out)

    def test_autoescaping_is_on(self):
        """Headline text is third-party; it must not be able to inject markup."""
        report = make_report(AGREE)
        report.games[0].home_news = [
            NewsItem("<script>alert(1)</script>", "evil", "https://e.com", None)
        ]
        out = html.render(report)
        self.assertNotIn("<script>alert(1)</script>", out)
        self.assertIn("&lt;script&gt;", out)

    def test_no_line_posted(self):
        self.assertIn("No betting line posted",
                      html.render(make_report(AGREE, probability=None)))

    def test_overview_links_to_each_game(self):
        out = html.render(make_report(AGREE))
        self.assertIn('href="#g401"', out)
        self.assertIn('id="g401"', out)

    def test_write_creates_file(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = html.write(make_report(AGREE), Path(tmp) / "out" / "r.html")
            self.assertTrue(path.exists())
            self.assertIn("NFL 2026", path.read_text())


if __name__ == "__main__":
    unittest.main()
