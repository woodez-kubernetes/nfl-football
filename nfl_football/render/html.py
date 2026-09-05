"""Standalone HTML report.

Self-contained by design: CSS is inlined in the template and nothing is fetched
at view time, so the file works offline and can be mailed around as one artefact.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from nfl_football import config
from nfl_football.models import GameReport
from nfl_football.pipeline import WeekReport
from nfl_football.render.markdown import _line_summary

TEMPLATE_DIR = Path(__file__).parent


def _abbr(gr: GameReport, display_name: str | None) -> str:
    if display_name == gr.game.home.display_name:
        return gr.game.home.abbreviation
    if display_name == gr.game.away.display_name:
        return gr.game.away.abbreviation
    return "?"


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # Kickoffs are stored in UTC and displayed in the report timezone.
    env.globals["local"] = config.to_local
    env.globals["tz_label"] = config.REPORT_TZ_NAME.split("/")[-1].replace("_", " ")
    return env


def render(report: WeekReport) -> str:
    """The full standalone report: modern CSS, dark mode, source links."""
    template = _environment().get_template("template.html.j2")
    return template.render(report=report, abbr=_abbr, line_summary=_line_summary)


def render_email(report: WeekReport, *, compact: bool | None = None) -> str:
    """The email body — a different template, not the same HTML.

    Mail clients are not browsers: Outlook has no flexbox or grid, Gmail strips
    CSS custom properties and (on mobile, for non-Gmail accounts) whole <style>
    blocks, and no client supports <details>. Gmail also clips messages over
    ~102 KB, so source links are left to the attached full report.

    Size degrades automatically rather than gambling on content length: the full
    body is rendered first, and if it would be clipped it is re-rendered without
    the per-game injury lines and "news vs market" notes. Both remain in the
    attached report. Pass ``compact`` explicitly to force either mode.
    """
    from nfl_football.mailer import clipping_warning  # local: avoids a cycle

    template = _environment().get_template("template_email.html.j2")

    def _render(is_compact: bool) -> str:
        return template.render(report=report, abbr=_abbr,
                               line_summary=_line_summary, compact=is_compact)

    if compact is not None:
        return _render(compact)

    body = _render(False)
    if clipping_warning(body) is None:
        return body
    return _render(True)


def write(report: WeekReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(report), encoding="utf-8")
    return path
