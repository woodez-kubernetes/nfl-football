"""Command line entry point.

    python -m nfl_football --week 1 --html report.html
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from nfl_football import config, llm, mailer, pipeline
from nfl_football.render import html as html_render
from nfl_football.render import markdown as md_render
from nfl_football.sources import cache as cache_mod


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nfl-thesis",
        description="Weekly NFL matchup theses from odds and news, "
                    "reasoned by a local LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Email setup — put credentials in a .env file in the project root:\n"
            "  cp .env.example .env && chmod 600 .env\n"
            "Then edit .env (Gmail needs an App Password, not your password) and\n"
            "add --email. Use --email-dry-run first to inspect the message.\n"
            ".env is gitignored; exported environment variables override it."
        ),
    )
    parser.add_argument("--season", type=int, help="season year (default: current)")
    parser.add_argument("--week", type=int, help="week number (default: current)")
    parser.add_argument("--game", metavar="AWAY@HOME",
                        help='limit to one matchup, e.g. "DEN@KC"')
    parser.add_argument("--refresh", action="store_true",
                        help="bypass all caches and refetch/regenerate")
    parser.add_argument("--no-llm", action="store_true",
                        help="odds and news only; skip briefs and theses")
    parser.add_argument("--html", metavar="PATH", default="report.html",
                        help="write the HTML report here (default: report.html)")
    parser.add_argument("--no-html", action="store_true", help="skip the HTML file")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print the markdown report to stdout")
    parser.add_argument("--no-sources", action="store_true",
                        help="omit source links from the markdown report")

    email = parser.add_argument_group("email")
    email.add_argument("--email", action="store_true",
                       help="send the HTML report by email")
    email.add_argument("--email-to", metavar="ADDR", default=config.EMAIL_TO,
                       help=f"recipient (default: {config.EMAIL_TO})")
    email.add_argument("--email-dry-run", metavar="PATH", nargs="?",
                       const="report.eml",
                       help="write the message to a .eml file instead of sending")
    email.add_argument("--check-email", action="store_true",
                       help="report SMTP configuration and exit without running")
    return parser


def _check_email(args) -> int:
    """Show what is configured, without ever printing the password."""
    smtp = mailer.SmtpConfig.from_env()
    print(f"env file:  {config.ENV_FILE}"
          f"{'' if config.ENV_FILE.exists() else '  (not found)'}")
    print(f"config:    {smtp!r}")
    print(f"recipient: {args.email_to}")
    warning = config.env_file_warning()
    if warning:
        print(f"warning:   {warning}")
    if smtp.is_configured:
        print("status:    ready — add --email to send")
        return 0
    print(f"status:    NOT configured — missing {', '.join(smtp.missing())}")
    print(f"           copy .env.example to .env and fill it in")
    return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_email:
        return _check_email(args)

    started = time.time()
    if args.email or args.email_dry_run:
        warning = config.env_file_warning()
        if warning:
            print(f"warning: {warning}", file=sys.stderr)

    def progress(index: int, total: int, game) -> None:
        if not args.quiet:
            print(f"[{index}/{total}] {game.matchup}", file=sys.stderr, flush=True)

    try:
        report = pipeline.build_week(
            season=args.season,
            week=args.week,
            refresh=args.refresh,
            use_llm=not args.no_llm,
            only=args.game,
            progress=progress,
        )
    except Exception as exc:  # noqa: BLE001 - top level guard
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not report.games:
        target = args.game or f"{args.season} week {args.week}"
        print(f"error: no games found for {target}", file=sys.stderr)
        return 1

    markdown = md_render.render(report, sources=not args.no_sources)
    if not args.quiet:
        print(markdown)

    html_path: Path | None = None
    if not args.no_html:
        html_path = html_render.write(report, args.html)

    exit_code = 0
    if args.email or args.email_dry_run:
        exit_code = _deliver(args, report, markdown, html_path)

    _summary(report, html_path, started)
    return exit_code


def _deliver(args, report, markdown: str, html_path: Path | None) -> int:
    # The email body uses the mail-safe template, not the standalone report:
    # clients have no flexbox/grid, strip CSS variables, and clip large bodies.
    body = html_render.render_email(report)
    warning = mailer.clipping_warning(body)
    if warning:
        print(f"warning: {warning}", file=sys.stderr)

    # The full report — dark mode, every source link — rides along as a file.
    attachments = [html_path] if html_path and html_path.exists() else []

    message = mailer.build_message(
        html_body=body,
        text_body=markdown,
        subject=mailer.subject_for(
            report.season, report.week, len(report.upset_calls), len(report.games)
        ),
        to=args.email_to,
        sender=mailer.SmtpConfig.from_env().sender or args.email_to,
        attachments=attachments,
    )

    if args.email_dry_run:
        path = mailer.save_eml(message, args.email_dry_run)
        print(f"email not sent (dry run) — message written to {path}", file=sys.stderr)
        return 0

    try:
        mailer.send(message, mailer.SmtpConfig.from_env())
    except mailer.MailError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"report emailed to {args.email_to}", file=sys.stderr)
    return 0


def _summary(report, html_path: Path | None, started: float) -> None:
    count = len(report.games)
    parts = [
        f"{count} {'game' if count == 1 else 'games'}",
        f"{len(report.upset_calls)} against the market",
    ]
    if report.games_without_a_pick:
        parts.append(f"{len(report.games_without_a_pick)} without a pick")
    if report.odds_filled_from_nflverse:
        parts.append(f"{report.odds_filled_from_nflverse} lines via nflverse")

    print(
        f"\n{' · '.join(parts)}\n"
        f"http: {cache_mod.STATS['hits']} cached, {cache_mod.STATS['misses']} fetched\n"
        f"llm:  {llm.STATS.summary()}\n"
        f"wall: {time.time() - started:.1f}s"
        + (f"\nhtml: {html_path}" if html_path else ""),
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
