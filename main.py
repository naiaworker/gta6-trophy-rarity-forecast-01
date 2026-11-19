"""gta6-trophy-rarity-forecast-01 — prototype forecaster (chunk 1).

Model: earners/players decomposition with hierarchical shrinkage and robust
trends, in logit space for the spread.

    rate_i(target) = E_i(target) / P(target)

where P is Exophase's players-tracked count and E_i trophy i's earner count
(recovered exactly from the printed rates: E = round(rate * P / 100); on the
practice cases the recovered earners sum exactly to the page's trophies_earned).

Pipeline per case:
  1. Compress the read history into distinct page states (the page only changes
     when Exophase refreshes it; consecutive identical reads are one state).
  2. Forecast P at the target:
     - >= 2 states: Theil-Sen slope of ln P over states, extrapolated with
       exponential decay of the growth rate (saturating growth).
     - 1 state (a frozen page): a refresh prior — the page is stale and will be
       refreshed before the target; the documented first refresh multiplied the
       player count by 3868/638 ~ 6.06 while earners grew far less.
  3. Forecast each E_i the same way (Theil-Sen slope of ln(E + 0.5), shrunk
     hierarchically toward the tier-group and global median slopes, saturating
     extrapolation, capped at P). In the frozen-page case, earners get a modest
     growth prior instead (they grew ~25% over the documented stale window,
     the growth implied by the house baseline's official practice loss ~1.51).
  4. Median rate from the ratio; spread on the logit scale combines the player
     uncertainty, the per-trophy earner uncertainty (shrunk toward its group),
     and a rounding term; quantiles are inv_logit(median_logit + z * sigma).

Deterministic: no randomness anywhere. Standard library only.
Reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR (/output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- Tunable priors / parameters (chunk 1 values; chunk 2 calibrates) --------
GROWTH_HALF_LIFE_DAYS = 1.0      # ln-growth rate of players/earners decays with this half-life
SAT_TAU_DAYS = 2.5               # saturating-extrapolation time constant for earners
SHRINK_K = 3.0                   # hierarchical shrinkage strength (pseudo-observations)
MAX_TS_STATES = 40               # cap on states used by the Theil-Sen fits
# Frozen-page refresh prior (from the refresh documented in RULES):
FLAT_LN_PLAYER_MULT = math.log(3868.0 / 638.0)   # ~1.802
FLAT_SIGMA_LN_PLAYERS = 0.27
FLAT_LN_EARNER_GROWTH = 0.25
FLAT_SIGMA_LN_EARNERS = 0.20
SIGMA_FLOOR = 0.12               # minimum logit spread (printing/rounding jitter)
SIGMA_ROUND_CAP = 0.15


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit_pct(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s: str) -> float:
    """ISO 8601 UTC string -> epoch seconds."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def median(xs: list) -> float:
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def theil_sen_slope(ts: list, ys: list) -> float:
    """Median pairwise slope; ts strictly increasing, len(ts) == len(ys) >= 2."""
    slopes = []
    n = len(ts)
    for a in range(n - 1):
        for b in range(a + 1, n):
            dt = ts[b] - ts[a]
            if dt > 0:
                slopes.append((ys[b] - ys[a]) / dt)
    return median(slopes) if slopes else 0.0


def thin(states: list, cap: int) -> list:
    """Deterministically thin a list to at most `cap` entries, keeping the last."""
    if len(states) <= cap:
        return states
    stride = (len(states) - 1) / (cap - 1)
    picked = [states[round(i * stride)] for i in range(cap)]
    picked[-1] = states[-1]
    return picked


class PageState:
    __slots__ = ("t", "players", "earners")

    def __init__(self, t: float, players: int, earners: list):
        self.t = t                # epoch seconds of the state's first read
        self.players = players
        self.earners = earners    # per-trophy earner counts (None where absent)


def page_states(case: dict) -> list:
    """Compress consecutive identical reads into distinct page states."""
    states = []
    prev_key = None
    for read in case["history"]:
        players = max(1, int(read["players_tracked"]))
        earners = [
            None if r is None else int(round(r * players / 100.0))
            for r in read["rates"]
        ]
        key = (players, tuple(earners))
        if key != prev_key:
            states.append(PageState(parse_time(read["at"]), players, earners))
            prev_key = key
    return states


def saturating_delta(rate_per_day: float, days: float, tau_days: float) -> float:
    """Integral of a growth rate that itself decays exponentially with time
    constant tau_days: the log-increment over `days` if the current log-growth
    rate is rate_per_day. Approaches rate*tau as days -> infinity (saturating).
    """
    if days <= 0:
        return 0.0
    return rate_per_day * tau_days * (1.0 - math.exp(-days / tau_days))


def forecast_players(states: list, horizon_days: float) -> tuple:
    """Return (median players at target, sigma of ln players)."""
    last = states[-1]
    if len(states) == 1:
        # Frozen page: it will refresh before the target. Prior centred on the
        # documented first-refresh multiplier, widening slowly with horizon.
        ln_mult = FLAT_LN_PLAYER_MULT + 0.35 * math.log1p(horizon_days)
        return last.players * math.exp(ln_mult), FLAT_SIGMA_LN_PLAYERS
    pts = thin([(s.t, math.log(s.players)) for s in states], MAX_TS_STATES)
    ts = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    slope_per_sec = theil_sen_slope(ts, ys)
    slope_per_day = slope_per_sec * 86400.0
    if slope_per_day <= 0:
        return float(last.players), 0.10
    tau = GROWTH_HALF_LIFE_DAYS / math.log(2.0)
    d_ln = saturating_delta(slope_per_day, horizon_days, tau)
    sigma = min(0.60, 0.08 + 0.15 * abs(d_ln))
    return last.players * math.exp(d_ln), sigma


def trophy_slopes(states: list, trophies: list) -> tuple:
    """Per-trophy Theil-Sen slopes of ln(E + 0.5) per day, hierarchically shrunk.

    Returns (shrunk slopes per trophy, raw slope or None per trophy, counts).
    Shrinkage: slope_i pulled toward its tier-group median, which is pulled
    toward the global median; weight from the number of usable states.
    """
    n_t = len(trophies)
    raw = [None] * n_t
    counts = [0] * n_t
    for i in range(n_t):
        pts = [
            (s.t, math.log(s.earners[i] + 0.5))
            for s in states
            if s.earners[i] is not None
        ]
        counts[i] = len(pts)
        if len(pts) >= 2:
            pts = thin(pts, MAX_TS_STATES)
            raw[i] = theil_sen_slope([p[0] for p in pts], [p[1] for p in pts]) * 86400.0
    have = [r for r in raw if r is not None]
    global_med = median(have) if have else 0.0
    group_med = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [
            raw[i]
            for i in range(n_t)
            if trophies[i]["tier"] == tier and raw[i] is not None
        ]
        group_med[tier] = median(vals) if vals else global_med
    shrunk = []
    for i in range(n_t):
        g = group_med[trophies[i]["tier"]]
        # Group median itself shrunk toward the global median.
        g = 0.5 * g + 0.5 * global_med
        if raw[i] is None:
            shrunk.append(g)
        else:
            w = counts[i] / (counts[i] + SHRINK_K)
            shrunk.append(w * raw[i] + (1.0 - w) * g)
    return shrunk, raw, counts


def forecast_case(case: dict) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"unexpected levels {case['levels']}")
    if not case["history"]:
        raise ValueError("case has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    states = page_states(case)
    last = states[-1]

    players_med, sigma_ln_players = forecast_players(states, horizon_days)
    flat = len(states) == 1
    if flat:
        slopes = [0.0] * len(trophies)
    else:
        slopes, _raw, _counts = trophy_slopes(states, trophies)

    rows = []
    for i, trophy in enumerate(trophies):
        # Most recent observed earners for this trophy.
        e_last = None
        for s in reversed(states):
            if s.earners[i] is not None:
                e_last = s.earners[i]
                break
        if e_last is None:
            # Never observed: fall back to a near-zero rate with a wide spread.
            med_rate, sigma = 0.05, 1.0
            rows.append(
                {"id": trophy["id"],
                 "q": [round4(inv_logit_pct(logit_pct(med_rate) + z * sigma)) for z in Z]}
            )
            continue
        if e_last >= last.players:
            # Universal trophy (every tracked player has it): it stayed at 100%
            # through the documented refresh; keep it pinned with a tight band.
            med_rate, sigma = 99.99, 0.10
            rows.append(
                {"id": trophy["id"],
                 "q": [round4(inv_logit_pct(logit_pct(med_rate) + z * sigma)) for z in Z]}
            )
            continue
        if flat:
            ln_g = FLAT_LN_EARNER_GROWTH
            sigma_ln_e = FLAT_SIGMA_LN_EARNERS
            e_med = e_last * math.exp(ln_g)
        else:
            d_ln_e = saturating_delta(slopes[i], horizon_days, SAT_TAU_DAYS)
            d_ln_e = max(-1.5, min(2.5, d_ln_e))
            e_med = (e_last + 0.5) * math.exp(d_ln_e) - 0.5
            sigma_ln_e = min(0.65, 0.10 + 0.20 * abs(d_ln_e) + 0.10 * math.sqrt(horizon_days))
        e_med = max(0.0, min(e_med, players_med * 0.999))
        med_rate = clamp_pct(100.0 * e_med / players_med)
        sigma_round = min(SIGMA_ROUND_CAP, 0.5 / max(1.0, e_med + 1.0))
        sigma = math.sqrt(sigma_ln_players ** 2 + sigma_ln_e ** 2 + sigma_round ** 2)
        sigma = max(SIGMA_FLOOR, sigma)
        lg = logit_pct(med_rate)
        rows.append(
            {"id": trophy["id"],
             "q": [round4(inv_logit_pct(lg + z * sigma)) for z in Z]}
        )
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
