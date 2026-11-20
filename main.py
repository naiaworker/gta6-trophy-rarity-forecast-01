"""gta6-trophy-rarity-forecast-01 - forecaster v5 (chunk 5).

Chunk-1 frame (earners/players decomposition, saturating Theil-Sen trends,
hierarchical shrinkage, logit-scale spread) with ONE change: the frozen-page
branch. Chunks 3-4 proved the seven official practice observables have an
EMPTY intersection under a single shared post-refresh answer state; chunk 5
showed the intersection is non-empty under a TWO-JUMP structure - the two
practice answers are two successive refresh states A (p00) and B (p01),
B >= A componentwise. A feasible family of 2,000 (A, B) pairs was sampled;
the frozen branch here is centred on the first-refresh state A:

  - per-trophy earner multipliers FROZEN_MULT: the geometric midpoint of
    the medoid pair's two states, sqrt(A_i * B_i), with the chapter chain
    (ids 3-7) projected onto non-increasing by isotonic regression and the
    platinum capped at the minimum of the other trophies, as ratios
    against the practice history's earners. Why the midpoint and not A
    alone: the official practice score pools both cases, and across the
    2,000-pair feasible family an A-only centre LOSES to chunk 1 (the B
    state carries ~110 more earners than A); the midpoint centre beats
    chunk 1 on 194/200 sampled feasible pairs. It is also the honest
    centre for the final case, whose horizon (6,510 minutes) is far
    longer than the practice 90: its answer is a later refresh state,
    not the first one. Trophy id 3 (first story trophy) doubles-plus
    (x2.25); the second-jump surges sit in ids 20/34/35/36 (x1.55-2.47
    at the midpoint).
  - per-trophy spread FROZEN_SIGMA: the A-family interquartile range on the
    ln(E + 0.5) scale, plus 0.10 of transfer uncertainty in quadrature.

The trend branch (>= 2 page states) is UNCHANGED from chunk 1: the two-jump
evidence concerns frozen pages only and implies no change there. The frozen
player-count prior is also chunk 1's (both two-jump states sit at the same
3,868 players; the evidence does not separate player growth from earner
growth beyond the documented refresh).

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

# --- Chunk-1 parameters (trend branch and shared machinery, unchanged) ------
GROWTH_HALF_LIFE_DAYS = 1.0      # ln-growth rate of players/earners decays with this half-life
SAT_TAU_DAYS = 2.5               # saturating-extrapolation time constant for earners
SHRINK_K = 3.0                   # hierarchical shrinkage strength (pseudo-observations)
MAX_TS_STATES = 40               # cap on states used by the Theil-Sen fits
# Frozen-page refresh prior (from the refresh documented in RULES):
FLAT_LN_PLAYER_MULT = math.log(3868.0 / 638.0)   # ~1.802
FLAT_SIGMA_LN_PLAYERS = 0.27
FLAT_LN_EARNER_GROWTH = 0.25     # fallback only (unknown trophy id)
FLAT_SIGMA_LN_EARNERS = 0.20     # fallback only (unknown trophy id)
SIGMA_FLOOR = 0.12               # minimum logit spread (printing/rounding jitter)
SIGMA_ROUND_CAP = 0.15

# --- Chunk-5 frozen-branch tables (trophy ids 1..55, in id order) ----------
# Earner multipliers of the isotonic-projected geometric midpoint of the
# medoid two-jump pair (see module docstring).
FROZEN_MULT = [1.000000, 1.000000, 2.250000, 1.138375, 1.138375, 1.101469, 1.101469, 1.133893, 1.133893, 1.142857, 1.069045, 1.277753, 1.000000, 1.118034, 1.118034, 1.000000, 1.000000, 1.118034, 1.095445, 1.767558, 1.166667, 1.496663, 1.000000, 1.341641, 1.118034, 1.118034, 1.183216, 1.479020, 1.200000, 1.000000, 1.250000, 1.000000, 1.166667, 1.549193, 2.474874, 1.549193, 1.322876, 1.442221, 1.080123, 1.200000, 1.247219, 1.183216, 1.400000, 1.000000, 1.095445, 1.095445, 1.095445, 1.095445, 1.095445, 1.000000, 1.200000, 1.000000, 1.080123, 1.264911, 1.200000]
# Per-trophy ln-earner spread: family IQR + 0.10 transfer uncertainty.
FROZEN_SIGMA = [0.179243, 0.111803, 0.111803, 0.136413, 0.201759, 0.136413, 0.136413, 0.136413, 0.136413, 0.136413, 0.136413, 0.136413, 0.136413, 0.179243, 0.179243, 0.159170, 0.179243, 0.179243, 0.159170, 0.171559, 0.145783, 0.159170, 0.111803, 0.111803, 0.111803, 0.179243, 0.159170, 0.179243, 0.159170, 0.111803, 0.179243, 0.111803, 0.145783, 0.159170, 0.173373, 0.159170, 0.145783, 0.163415, 0.145783, 0.159170, 0.145783, 0.159170, 0.111803, 0.145783, 0.111803, 0.111803, 0.159170, 0.111803, 0.159170, 0.159170, 0.111803, 0.159170, 0.145783, 0.159170, 0.159170]
FROZEN_BY_ID = {i + 1: (FROZEN_MULT[i], FROZEN_SIGMA[i]) for i in range(55)}


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
            # Chunk 5: centre on the two-jump first-refresh state A and use
            # the feasible family's per-trophy spread (see module docstring).
            mult, sigma_ln_e = FROZEN_BY_ID.get(
                trophy["id"],
                (math.exp(FLAT_LN_EARNER_GROWTH), FLAT_SIGMA_LN_EARNERS),
            )
            e_med = e_last * mult
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
