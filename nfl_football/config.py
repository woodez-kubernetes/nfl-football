"""Runtime configuration and season/week resolution.

Settings come from the environment. A ``.env`` file in the project root is
loaded first, so credentials live in one gitignored file rather than in shell
history. Real environment variables always win over ``.env`` — handy for a
one-off override without editing the file.
"""

from __future__ import annotations

import os
import stat
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = Path(os.environ.get("NFL_ENV_FILE") or PROJECT_ROOT / ".env")

# override=False: an exported variable beats the file.
load_dotenv(ENV_FILE, override=False)


def env_file_warning(path: Path | None = None) -> str | None:
    """Warn if the .env file is readable by anyone but its owner.

    It holds an SMTP password; 0600 is the right mode.
    """
    path = path or ENV_FILE
    try:
        mode = path.stat().st_mode
    except OSError:
        return None
    if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
        return (
            f"{path} is readable by other users (mode {stat.filemode(mode)}). "
            f"It holds an SMTP password — run: chmod 600 {path}"
        )
    return None


# --- LLM -------------------------------------------------------------------
# Ollama defaults num_ctx to 4096, which is too small for the Stage-1 briefs.
# It must be set explicitly on every call.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://192.168.2.167:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_NUM_CTX = 16384
# Wall time on this host is ~10 tok/s of *generation*; prompt size and num_ctx
# barely matter (measured: 2048/4096/16384 all ~9-10 tok/s). Capping output is
# therefore the main runtime control. Concurrency does not help — the host
# serialises (2 workers = 1.10x, 4 workers = 0.83x).
OLLAMA_NUM_PREDICT = 400
NUM_PREDICT_BRIEF = 300  # Stage-1 team briefs: bounded arrays, ~150 tok typical
NUM_PREDICT_THESIS = 450  # Stage-2 game theses: prose, needs a little more room
NUM_PREDICT_LEAN = 160  # Stage-2 pass 1: one team name plus one sentence
OLLAMA_TEMPERATURE = 0.2
OLLAMA_TIMEOUT = 300  # generous: a cold model load measured ~8s
OLLAMA_MAX_RETRIES = 2

# --- Paths -----------------------------------------------------------------
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DB = CACHE_DIR / "nfl.db"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

# --- Cache TTLs (seconds) --------------------------------------------------
# TTL_ODDS must comfortably exceed a full cold run. Measured cold run: ~33 min.
# At the original 15 min the odds entry expired *while the run was still going*,
# so an immediate rerun refetched moved lines, which changed every thesis prompt
# and invalidated all 32 cached LLM responses — a "warm" rerun cost the full
# 30+ minutes again. Use --refresh when you deliberately want current numbers.
TTL_ODDS = 3 * 60 * 60
TTL_NEWS = 2 * 60 * 60
TTL_INJURIES = 6 * 60 * 60
TTL_SCHEDULE = 24 * 60 * 60
# LLM responses are content-addressed: the cache key is a hash of the model,
# schema and full prompt, so changed headlines invalidate a brief automatically.
# The TTL is therefore only a backstop and can be long.
TTL_LLM = 7 * 24 * 60 * 60

# --- Sources ---------------------------------------------------------------
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_SCOREBOARD = f"{ESPN_BASE}/scoreboard"
ESPN_INJURIES = f"{ESPN_BASE}/injuries"
ESPN_NEWS = f"{ESPN_BASE}/news"
GNEWS_RSS = "https://news.google.com/rss/search"
NFLVERSE_GAMES = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

# --- User-Agent: do not set one. ------------------------------------------
# ESPN's edge returns 403 for browser-like User-Agents *and* for unrecognised
# custom ones ("nfl-thesis/0.1" is rejected). It accepts the default UAs of
# programmatic clients — "python-requests/x.y" and "curl/x.y" both return 200.
# Google News and nflverse accept the same default.
#
# Verified 2026-09-02:
#   ESPN + "Mozilla/5.0 ... Chrome/124"  -> 403
#   ESPN + no User-Agent header          -> 403
#   ESPN + "nfl-thesis/0.1"              -> 403
#   ESPN + "python-requests/2.34.2"      -> 200
#
# So: send no custom User-Agent and let requests use its own. Adding a
# realistic browser UA here will break every ESPN call.

HTTP_TIMEOUT = 20
MAX_NEWS_PER_TEAM = 40
NEWS_WINDOW = "7d"

SEASON_TYPE_REGULAR = 2

# --- Display timezone ------------------------------------------------------
# Kickoffs arrive from ESPN in UTC. Everything shown to a reader is converted to
# this zone. It matters more than it looks: a 00:20 UTC Thursday kickoff is
# 20:20 on Wednesday in Toronto, so the *day* changes, not just the clock.
REPORT_TZ_NAME = os.environ.get("NFL_TZ", "America/Toronto")
REPORT_TZ = ZoneInfo(REPORT_TZ_NAME)


def to_local(moment: datetime) -> datetime:
    """Convert an aware datetime into the report timezone.

    A naive datetime is assumed to be UTC, which is what every source returns.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(REPORT_TZ)

# --- Email -----------------------------------------------------------------
# Credentials are read from the environment at send time (see mailer.py) and are
# never stored here. Nothing is sent unless --email is passed explicitly.
EMAIL_TO = os.environ.get("NFL_REPORT_TO", "kevin.wood75@gmail.com")


def current_season_week(today: date | None = None) -> tuple[int, int]:
    """Best-effort season year and regular-season week from the calendar.

    This is the offline fallback. ``sources.schedule.resolve_current_week()``
    asks ESPN directly and should be preferred when the network is available.

    Week 1 opens on the Thursday following the first Monday of September.
    """
    today = today or datetime.now(timezone.utc).date()

    def opener(year: int) -> date:
        sept = date(year, 9, 1)
        # weekday(): Monday == 0
        first_monday = sept + timedelta(days=(0 - sept.weekday()) % 7)
        return first_monday + timedelta(days=3)

    if today >= opener(today.year):
        season = today.year
    elif today.month >= 7:
        # Jul-early Sep: preseason. Point at the upcoming opener.
        return today.year, 1
    else:
        # Jan-Jun: playoffs or offseason, still the prior season's calendar.
        season = today.year - 1

    week = (today - opener(season)).days // 7 + 1
    return season, max(1, min(week, 18))
