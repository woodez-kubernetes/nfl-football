"""Re-derive SPREAD_LOGIT_PER_POINT from nflverse history.

Not a unit test — a reproducible analysis. Run it if the historical data is
refreshed and paste the fitted constant into nfl_football/probability.py.

    source .venv/bin/activate && python -m tests.calibrate_spread
"""

from __future__ import annotations

import csv
import io
import math

from nfl_football import config
from nfl_football.probability import SPREAD_LOGIT_PER_POINT, american_to_implied, devig
from nfl_football.sources.cache import cached_get


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    raw = cached_get(config.NFLVERSE_GAMES, ttl=config.TTL_SCHEDULE)
    rows = list(csv.DictReader(io.StringIO(raw)))

    # --- Fit: spread -> de-vigged market probability -----------------------
    samples = []
    for r in rows:
        spread, home_ml, away_ml = (
            _f(r["spread_line"]), _f(r["home_moneyline"]), _f(r["away_moneyline"])
        )
        if None in (spread, home_ml, away_ml) or 0 in (home_ml, away_ml):
            continue
        p_home, _, overround = devig(
            american_to_implied(int(home_ml)), american_to_implied(int(away_ml))
        )
        samples.append((spread, p_home, overround))

    def logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    num = sum(s * logit(p) for s, p, _ in samples)
    den = sum(s * s for s, _, _ in samples)
    k = num / den

    n = len(samples)
    sx = sum(s for s, _, _ in samples)
    sy = sum(logit(p) for _, p, _ in samples)
    sxy = num
    sxx = den
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n

    vigs = sorted(o for _, _, o in samples)
    errs = sorted(abs(1 / (1 + math.exp(-k * s)) - p) for s, p, _ in samples)

    print(f"games with spread + both moneylines : {n}")
    print(f"median overround                    : {vigs[n // 2]:.4f}")
    print(f"fitted k (through origin)           : {k:.5f}")
    print(f"free-intercept check                : slope={slope:.5f} intercept={intercept:.5f}")
    print(f"fit vs market  mean|err|={sum(errs)/n:.4f}  median={errs[n//2]:.4f}  p95={errs[int(n*.95)]:.4f}")

    # --- Validate: does the curve match real outcomes? ---------------------
    brier = brier_naive = 0.0
    total = 0
    weighted_err = 0.0
    buckets: dict[int, list] = {}
    for r in rows:
        spread, result = _f(r["spread_line"]), _f(r["result"])
        if spread is None or result is None or r["game_type"] != "REG" or result == 0:
            continue
        y = 1.0 if result > 0 else 0.0
        pred = 1 / (1 + math.exp(-k * spread))
        brier += (pred - y) ** 2
        brier_naive += (0.5 - y) ** 2
        total += 1
        b = buckets.setdefault(round(spread), [0, 0.0, 0.0])
        b[0] += 1
        b[1] += y
        b[2] += pred

    for count, wins, pred_sum in buckets.values():
        if count >= 60:
            weighted_err += abs(wins / count - pred_sum / count) * count

    graded = sum(c for c, _, _ in buckets.values() if c >= 60)
    print(f"\nvalidation games (REG, ties dropped): {total}")
    print(f"n-weighted mean abs calibration error: {weighted_err / graded:.4f}")
    print(f"Brier fitted={brier / total:.4f}  coin-flip={brier_naive / total:.4f}")
    print(f"\nconstant currently in probability.py : {SPREAD_LOGIT_PER_POINT}")


if __name__ == "__main__":
    main()
