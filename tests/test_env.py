"""Stage 7: .env loading, file permissions and credential hygiene."""

from __future__ import annotations

import contextlib
import io
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import dotenv_values, load_dotenv

from nfl_football import cli, config, mailer


class TestEnvFilePermissions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / ".env"
        self.path.write_text("SMTP_PASSWORD=secret\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_warning_when_owner_only(self):
        self.path.chmod(0o600)
        self.assertIsNone(config.env_file_warning(self.path))

    def test_warns_when_group_or_world_readable(self):
        self.path.chmod(0o644)
        warning = config.env_file_warning(self.path)
        self.assertIsNotNone(warning)
        self.assertIn("chmod 600", warning)

    def test_warns_when_world_writable(self):
        self.path.chmod(0o666)
        self.assertIsNotNone(config.env_file_warning(self.path))

    def test_warning_never_contains_the_secret(self):
        self.path.chmod(0o644)
        self.assertNotIn("secret", config.env_file_warning(self.path))

    def test_missing_file_is_not_a_warning(self):
        self.assertIsNone(config.env_file_warning(Path(self.tmp.name) / "absent"))


class TestEnvLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / ".env"

    def tearDown(self):
        self.tmp.cleanup()

    def test_values_are_read_from_the_file(self):
        self.path.write_text(
            "SMTP_HOST=smtp.gmail.com\nSMTP_USER=a@b.com\nSMTP_PASSWORD=pw\n"
        )
        values = dotenv_values(self.path)
        cfg = mailer.SmtpConfig.from_env(values)
        self.assertEqual(cfg.host, "smtp.gmail.com")
        self.assertTrue(cfg.is_configured)

    def test_exported_variable_wins_over_the_file(self):
        """override=False is deliberate: a one-off export beats the file."""
        self.path.write_text("NFL_TEST_KEY=from_file\n")
        with patch.dict(os.environ, {"NFL_TEST_KEY": "from_shell"}, clear=False):
            load_dotenv(self.path, override=False)
            self.assertEqual(os.environ["NFL_TEST_KEY"], "from_shell")

    def test_file_fills_an_unset_variable(self):
        self.path.write_text("NFL_TEST_UNSET=from_file\n")
        os.environ.pop("NFL_TEST_UNSET", None)
        try:
            load_dotenv(self.path, override=False)
            self.assertEqual(os.environ["NFL_TEST_UNSET"], "from_file")
        finally:
            os.environ.pop("NFL_TEST_UNSET", None)

    def test_comments_and_blank_lines_ignored(self):
        self.path.write_text("# a comment\n\nSMTP_HOST=h\n")
        self.assertEqual(dotenv_values(self.path), {"SMTP_HOST": "h"})

    def test_quoted_value_with_spaces(self):
        """Gmail App Passwords are shown as four space-separated groups."""
        self.path.write_text("SMTP_PASSWORD='abcd efgh ijkl mnop'\n")
        self.assertEqual(dotenv_values(self.path)["SMTP_PASSWORD"], "abcd efgh ijkl mnop")


class TestCheckEmail(unittest.TestCase):
    def _run(self, env):
        """Run --check-email against a controlled environment.

        clear=True so a developer's real exported SMTP_* cannot make the
        'missing configuration' case pass or fail spuriously.
        """
        buf = io.StringIO()
        with patch.dict(os.environ, env, clear=True):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["--check-email"])
        return code, buf.getvalue()

    def test_reports_missing_configuration(self):
        code, out = self._run({})
        self.assertEqual(code, 1)
        self.assertIn("NOT configured", out)
        self.assertIn("SMTP_HOST", out)

    def test_reports_ready_when_configured(self):
        code, out = self._run({"SMTP_HOST": "h", "SMTP_USER": "u@x.com",
                               "SMTP_PASSWORD": "pw"})
        self.assertEqual(code, 0)
        self.assertIn("ready", out)

    def test_never_prints_the_password(self):
        code, out = self._run({"SMTP_HOST": "h", "SMTP_USER": "u@x.com",
                               "SMTP_PASSWORD": "hunter2-topsecret"})
        self.assertNotIn("hunter2", out)
        self.assertIn("***set***", out)

    def test_does_not_run_the_pipeline(self):
        with patch.object(cli.pipeline, "build_week") as build:
            self._run({})
        build.assert_not_called()


class TestExampleFile(unittest.TestCase):
    def test_example_exists_and_holds_no_real_secret(self):
        example = config.PROJECT_ROOT / ".env.example"
        self.assertTrue(example.exists())
        text = example.read_text()
        self.assertIn("SMTP_PASSWORD=", text)
        self.assertIn("xxxx", text)  # placeholder, not a credential

    def test_env_is_gitignored(self):
        ignore = (config.PROJECT_ROOT / ".gitignore").read_text().splitlines()
        self.assertIn(".env", [line.strip() for line in ignore])


if __name__ == "__main__":
    unittest.main()
