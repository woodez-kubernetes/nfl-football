"""Email-body rendering: client compatibility, size, and attachments.

Mail clients are not browsers. These tests pin the constraints that the
standalone report deliberately ignores.
"""

from __future__ import annotations

import email
import email.policy
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nfl_football import mailer
from nfl_football.models import (
    Game, GameReport, NewsItem, Odds, Team, TeamBrief, Thesis, WinProbability,
)
from nfl_football.pipeline import WeekReport
from nfl_football.render import html as html_render


def make_week(count=16, thesis_len=520, factors=4, factor_len=110):
    games = []
    for i in range(count):
        home = Team(str(i), f"H{i:02d}", f"Home Team {i} Longname", "City", f"N{i}")
        away = Team(str(50 + i), f"A{i:02d}", f"Away Team {i} Longname", "City", f"N{i}")
        odds = Odds("DraftKings", "espn", -3.5, 44.5, -175, 145, "HOME -3.5")
        game = Game(str(400 + i), 2026, 1, 2,
                    datetime(2026, 9, 10, tzinfo=timezone.utc) + timedelta(hours=i),
                    home, away, "Some Big Stadium", False, odds)
        thesis = Thesis(game.espn_id, away.display_name, "medium", "word " * (thesis_len // 5),
                        [("factor " * (factor_len // 7)).strip()] * factors,
                        "news vs market " * 12, is_upset_call=(i % 3 == 0),
                        news_lean=away.display_name, lean_reason="reason " * 15)
        games.append(GameReport(
            game=game, probability=WinProbability(0.609, 0.391, "moneyline", 0.045),
            thesis=thesis,
            home_brief=TeamBrief(home.abbreviation,
                                 injuries=[f"Player {j} (POS) — Questionable" for j in range(6)]),
            away_brief=TeamBrief(away.abbreviation,
                                 injuries=[f"Player {j} (POS) — Out" for j in range(6)]),
            home_news=[NewsItem("H", "ESPN", "https://e.com/1", None)],
            away_news=[NewsItem("A", "ESPN", "https://e.com/2", None)],
        ))
    return WeekReport(2026, 1, datetime(2026, 9, 2, 12, tzinfo=timezone.utc), games)


class TestEmailClientCompatibility(unittest.TestCase):
    """Each of these is unsupported or stripped by a major mail client."""

    @classmethod
    def setUpClass(cls):
        cls.body = html_render.render_email(make_week())
        cls.compact = cls.body.replace(" ", "")

    def test_no_css_custom_properties(self):
        """Gmail and Outlook strip var(--x); the standalone report is built on it."""
        self.assertNotIn("var(--", self.body)

    def test_no_style_block(self):
        """Gmail mobile drops <style> for non-Gmail accounts."""
        self.assertNotIn("<style", self.body.lower())

    def test_no_class_selectors(self):
        self.assertNotIn('class="', self.body)

    def test_no_flexbox_or_grid(self):
        """Outlook's Word rendering engine supports neither."""
        self.assertNotIn("display:flex", self.compact)
        self.assertNotIn("display:grid", self.compact)

    def test_no_details_element(self):
        self.assertNotIn("<details", self.body.lower())

    def test_no_scripts_or_external_assets(self):
        self.assertNotIn("<script", self.body.lower())
        self.assertIsNone(re.search(r"(src=|<link)", self.body))

    def test_uses_inline_styles_and_tables(self):
        self.assertGreater(self.body.count("style="), 200)
        self.assertGreater(self.body.count("<table"), 20)

    def test_long_unbroken_text_cannot_blow_out_the_layout(self):
        """A 500-char token (e.g. a URL in a headline) must wrap, not widen the
        table — measured at 2289px before word-break was added."""
        week = make_week(count=1)
        week.games[0].thesis.thesis = "x" * 500
        body = html_render.render_email(week)
        self.assertIn("word-break:break-word", body)


class TestEmailSizeBudget(unittest.TestCase):
    def test_typical_week_fits_under_the_gmail_clip_limit(self):
        body = html_render.render_email(make_week(thesis_len=200, factors=3))
        self.assertIsNone(mailer.clipping_warning(body))

    def test_worst_case_week_still_fits(self):
        body = html_render.render_email(make_week(thesis_len=520, factors=4))
        self.assertIsNone(mailer.clipping_warning(body))

    def test_compact_mode_is_smaller(self):
        week = make_week()
        full = html_render.render_email(week, compact=False)
        compact = html_render.render_email(week, compact=True)
        self.assertLess(len(compact), len(full))

    def test_compact_drops_injuries_and_news_vs_market(self):
        week = make_week(count=1)
        compact = html_render.render_email(week, compact=True)
        self.assertNotIn("News vs market", compact)
        self.assertNotIn("Player 0 (POS)", compact)

    def test_auto_degrades_to_compact_when_full_would_clip(self):
        """Beyond a real NFL week (max 16 games) the body switches to compact.

        Compact is not a guarantee of fitting — at some size nothing does — but
        the switch must happen, and `clipping_warning` still tells the operator.
        """
        huge = make_week(count=40)
        body = html_render.render_email(huge)
        self.assertNotIn("News vs market", body)  # degraded
        self.assertLess(len(body), len(html_render.render_email(huge, compact=False)))
        self.assertIsNotNone(mailer.clipping_warning(body))  # and still reported

    def test_a_real_sized_week_never_needs_compact(self):
        """16 games at maximum content length must fit in the full layout."""
        body = html_render.render_email(make_week(count=16, thesis_len=520, factors=4))
        self.assertIsNone(mailer.clipping_warning(body))
        self.assertIn("News vs market", body)

    def test_clipping_warning_thresholds(self):
        self.assertIsNone(mailer.clipping_warning("x" * 1000))
        self.assertIsNotNone(mailer.clipping_warning("x" * (mailer.GMAIL_CLIP_BYTES + 1)))


class TestEmailContent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = html_render.render_email(make_week())

    def test_shows_pick_market_and_news_lean(self):
        self.assertIn("Pick:", self.body)
        self.assertIn("Market:", self.body)
        self.assertIn("News alone:", self.body)

    def test_upset_badge_present(self):
        self.assertIn("UPSET CALL", self.body)

    def test_probability_bars_use_percent_width_cells(self):
        self.assertIn('width="61%"', self.body)
        self.assertIn('width="39%"', self.body)

    def test_bars_set_bgcolor_attribute_not_just_css(self):
        """Outlook honours the bgcolor attribute more reliably than CSS."""
        self.assertIn('bgcolor="#1f6feb"', self.body)

    def test_autoescaping_still_on(self):
        week = make_week(count=1)
        week.games[0].thesis.thesis = "<script>alert(1)</script>"
        body = html_render.render_email(week)
        self.assertNotIn("<script>alert(1)</script>", body)


class TestAttachments(unittest.TestCase):
    def test_full_report_is_attached(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            path.write_text("<html><body>full report</body></html>")
            message = mailer.build_message(
                html_body="<div>body</div>", text_body="body", subject="s",
                to="a@b.com", sender="c@d.com", attachments=[path],
            )
            parsed = email.message_from_bytes(bytes(message), policy=email.policy.default)
            names = [p.get_filename() for p in parsed.iter_attachments()]
            self.assertIn("report.html", names)

    def test_missing_attachment_is_skipped_not_fatal(self):
        message = mailer.build_message(
            html_body="<div>b</div>", text_body="b", subject="s",
            to="a@b.com", sender="c@d.com", attachments=[Path("/nonexistent.html")],
        )
        parsed = email.message_from_bytes(bytes(message), policy=email.policy.default)
        self.assertEqual(list(parsed.iter_attachments()), [])

    def test_body_still_multipart_alternative_with_attachment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.html"
            path.write_text("<html></html>")
            message = mailer.build_message(
                html_body="<div>b</div>", text_body="plain", subject="s",
                to="a@b.com", sender="c@d.com", attachments=[path],
            )
            parsed = email.message_from_bytes(bytes(message), policy=email.policy.default)
            self.assertIn("plain", parsed.get_body(preferencelist=("plain",)).get_content())
            self.assertIn("<div>b</div>",
                          parsed.get_body(preferencelist=("html",)).get_content())


if __name__ == "__main__":
    unittest.main()
