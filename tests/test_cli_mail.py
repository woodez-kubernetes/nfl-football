"""Stage 7: CLI and mailer. No network, no LLM, no real SMTP."""

from __future__ import annotations

import contextlib
import email
import email.policy
import io
import smtplib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from nfl_football import cli, mailer
from nfl_football.models import Game, GameReport, Odds, Team, Thesis, WinProbability
from nfl_football.pipeline import WeekReport

HOME = Team("26", "SEA", "Seattle Seahawks", "Seattle", "Seahawks")
AWAY = Team("17", "NE", "New England Patriots", "New England", "Patriots")
ODDS = Odds("DraftKings", "espn", -3.5, 44.5, -175, 145, "SEA -3.5")
GAME = Game("401", 2026, 1, 2, datetime(2026, 9, 10, 0, 20, tzinfo=timezone.utc),
            HOME, AWAY, "Lumen Field", False, ODDS)
THESIS = Thesis("401", "New England Patriots", "medium", "Scheme travels.",
                ["A", "B", "C"], "Contradicts.", is_upset_call=True,
                news_lean="New England Patriots", lean_reason="Fewer injuries.")


@contextlib.contextmanager
def _quiet_stderr():
    """The CLI writes its run summary to stderr; keep test output readable."""
    with contextlib.redirect_stderr(io.StringIO()):
        yield


def week_report():
    return WeekReport(
        season=2026, week=1,
        generated_at=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        games=[GameReport(game=GAME,
                          probability=WinProbability(0.609, 0.391, "moneyline", 0.045),
                          thesis=THESIS)],
    )


class TestSmtpConfig(unittest.TestCase):
    def test_reads_environment(self):
        cfg = mailer.SmtpConfig.from_env({
            "SMTP_HOST": "smtp.gmail.com", "SMTP_PORT": "587",
            "SMTP_USER": "a@b.com", "SMTP_PASSWORD": "pw",
        })
        self.assertEqual(cfg.host, "smtp.gmail.com")
        self.assertEqual(cfg.port, 587)
        self.assertEqual(cfg.sender, "a@b.com")  # defaults to user
        self.assertTrue(cfg.is_configured)

    def test_sender_override(self):
        cfg = mailer.SmtpConfig.from_env({
            "SMTP_HOST": "h", "SMTP_USER": "a@b.com",
            "SMTP_PASSWORD": "pw", "SMTP_FROM": "reports@b.com",
        })
        self.assertEqual(cfg.sender, "reports@b.com")

    def test_port_465_implies_ssl(self):
        cfg = mailer.SmtpConfig.from_env({"SMTP_PORT": "465"})
        self.assertTrue(cfg.use_ssl)

    def test_starttls_can_be_disabled(self):
        cfg = mailer.SmtpConfig.from_env({"SMTP_STARTTLS": "false"})
        self.assertFalse(cfg.use_starttls)

    def test_unconfigured_reports_what_is_missing(self):
        cfg = mailer.SmtpConfig.from_env({})
        self.assertFalse(cfg.is_configured)
        self.assertEqual(cfg.missing(), ["SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"])

    def test_repr_never_leaks_the_password(self):
        cfg = mailer.SmtpConfig.from_env({
            "SMTP_HOST": "h", "SMTP_USER": "u", "SMTP_PASSWORD": "hunter2-secret",
        })
        self.assertNotIn("hunter2", repr(cfg))
        self.assertIn("***set***", repr(cfg))


class TestBuildMessage(unittest.TestCase):
    def setUp(self):
        self.msg = mailer.build_message(
            html_body="<!doctype html><html><body><h1>Report</h1></body></html>",
            text_body="# Report", subject="NFL 2026 Week 1",
            to="kevin.wood75@gmail.com", sender="bot@example.com",
        )

    def test_headers(self):
        self.assertEqual(self.msg["To"], "kevin.wood75@gmail.com")
        self.assertEqual(self.msg["From"], "bot@example.com")
        self.assertEqual(self.msg["Subject"], "NFL 2026 Week 1")
        self.assertTrue(self.msg["Date"])
        self.assertTrue(self.msg["Message-ID"])

    def test_multipart_alternative_with_both_bodies(self):
        parsed = email.message_from_bytes(bytes(self.msg), policy=email.policy.default)
        self.assertEqual(parsed.get_content_type(), "multipart/alternative")
        self.assertIn("Report", parsed.get_body(preferencelist=("plain",)).get_content())
        self.assertIn("<h1>", parsed.get_body(preferencelist=("html",)).get_content())

    def test_subject_pluralisation(self):
        self.assertIn("1 game,", mailer.subject_for(2026, 1, 0, 1))
        self.assertIn("16 games,", mailer.subject_for(2026, 1, 7, 16))


class TestSend(unittest.TestCase):
    CONFIGURED = {"SMTP_HOST": "h", "SMTP_USER": "u", "SMTP_PASSWORD": "p"}

    def _message(self):
        return mailer.build_message(html_body="<p>x</p>", text_body="x",
                                    subject="s", to="a@b.com", sender="c@d.com")

    def test_refuses_when_unconfigured(self):
        with self.assertRaises(mailer.MailError) as ctx:
            mailer.send(self._message(), mailer.SmtpConfig.from_env({}))
        self.assertIn("SMTP_HOST", str(ctx.exception))

    def test_starttls_flow(self):
        server = MagicMock()
        server.__enter__ = MagicMock(return_value=server)
        server.__exit__ = MagicMock(return_value=False)
        with patch("smtplib.SMTP", return_value=server) as ctor:
            mailer.send(self._message(), mailer.SmtpConfig.from_env(self.CONFIGURED))
        ctor.assert_called_once()
        server.starttls.assert_called_once()
        server.login.assert_called_once_with("u", "p")
        server.send_message.assert_called_once()

    def test_ssl_flow_skips_starttls(self):
        server = MagicMock()
        server.__enter__ = MagicMock(return_value=server)
        server.__exit__ = MagicMock(return_value=False)
        env = dict(self.CONFIGURED, SMTP_PORT="465")
        with patch("smtplib.SMTP_SSL", return_value=server):
            mailer.send(self._message(), mailer.SmtpConfig.from_env(env))
        server.starttls.assert_not_called()
        server.send_message.assert_called_once()

    def test_auth_failure_mentions_app_password(self):
        server = MagicMock()
        server.__enter__ = MagicMock(return_value=server)
        server.__exit__ = MagicMock(return_value=False)
        server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"denied")
        with patch("smtplib.SMTP", return_value=server):
            with self.assertRaises(mailer.MailError) as ctx:
                mailer.send(self._message(), mailer.SmtpConfig.from_env(self.CONFIGURED))
        self.assertIn("App Password", str(ctx.exception))

    def test_network_failure_wrapped(self):
        with patch("smtplib.SMTP", side_effect=OSError("no route")):
            with self.assertRaises(mailer.MailError):
                mailer.send(self._message(), mailer.SmtpConfig.from_env(self.CONFIGURED))


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, argv):
        with patch.object(cli.pipeline, "build_week", return_value=week_report()):
            with _quiet_stderr():
                return cli.main(argv)

    def test_writes_html(self):
        out = self.dir / "r.html"
        self.assertEqual(self._run(["--quiet", "--html", str(out)]), 0)
        self.assertTrue(out.exists())
        self.assertIn("New England Patriots", out.read_text())

    def test_no_html_skips_the_file(self):
        out = self.dir / "r.html"
        self._run(["--quiet", "--no-html", "--html", str(out)])
        self.assertFalse(out.exists())

    def test_dry_run_writes_eml_and_sends_nothing(self):
        eml = self.dir / "m.eml"
        with patch("smtplib.SMTP") as smtp:
            code = self._run(["--quiet", "--no-html", "--email-dry-run", str(eml)])
        self.assertEqual(code, 0)
        self.assertTrue(eml.exists())
        smtp.assert_not_called()
        parsed = email.message_from_bytes(eml.read_bytes(), policy=email.policy.default)
        self.assertEqual(parsed["To"], "kevin.wood75@gmail.com")
        self.assertIn("1 against the market", parsed["Subject"])

    def test_nothing_is_sent_without_the_email_flag(self):
        with patch.object(cli.mailer, "send") as send:
            self._run(["--quiet", "--no-html"])
        send.assert_not_called()

    def test_email_flag_sends(self):
        with patch.object(cli.mailer, "send") as send:
            code = self._run(["--quiet", "--no-html", "--email"])
        self.assertEqual(code, 0)
        send.assert_called_once()

    def test_email_failure_returns_nonzero(self):
        with patch.object(cli.mailer, "send", side_effect=mailer.MailError("nope")):
            code = self._run(["--quiet", "--no-html", "--email"])
        self.assertEqual(code, 1)

    def test_custom_recipient(self):
        eml = self.dir / "m.eml"
        self._run(["--quiet", "--no-html", "--email-to", "other@example.com",
                   "--email-dry-run", str(eml)])
        parsed = email.message_from_bytes(eml.read_bytes(), policy=email.policy.default)
        self.assertEqual(parsed["To"], "other@example.com")

    def test_empty_week_is_an_error(self):
        empty = WeekReport(season=2026, week=1,
                           generated_at=datetime.now(timezone.utc), games=[])
        with patch.object(cli.pipeline, "build_week", return_value=empty):
            with _quiet_stderr():
                self.assertEqual(cli.main(["--quiet", "--no-html", "--game", "XX@YY"]), 1)

    def test_pipeline_failure_returns_nonzero(self):
        with patch.object(cli.pipeline, "build_week", side_effect=RuntimeError("boom")):
            with _quiet_stderr():
                self.assertEqual(cli.main(["--quiet", "--no-html"]), 1)

    def test_flags_reach_the_pipeline(self):
        with patch.object(cli.pipeline, "build_week",
                          return_value=week_report()) as build:
            with _quiet_stderr():
                cli.main(["--quiet", "--no-html", "--season", "2025", "--week", "9",
                          "--game", "DEN@KC", "--refresh", "--no-llm"])
        kwargs = build.call_args.kwargs
        self.assertEqual(kwargs["season"], 2025)
        self.assertEqual(kwargs["week"], 9)
        self.assertEqual(kwargs["only"], "DEN@KC")
        self.assertTrue(kwargs["refresh"])
        self.assertFalse(kwargs["use_llm"])


if __name__ == "__main__":
    unittest.main()
