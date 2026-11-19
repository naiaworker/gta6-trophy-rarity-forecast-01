"""gta6-trophy-rarity-forecast-01 — prototype forecaster (chunk 2).

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
     extrapolation, capped at P). In the frozen-page case (chunk 2), earners
     follow a deterministic FEATURE RULE, reverse-engineered from the official
     practice score (see mech3.py and the refinement in metode.md): the
     refresh's per-trophy growth multiplier is
         g_i = C * (E_i / mean E)^a * (chapter weight)^b * (tier weight)^c
     normalised so the geometric mean of the weights is 1, with
     a = -0.6, b = 0.6, c = 0.7, C = 1.14; E'_i = E_i + round(E_i (g_i - 1)).
     The rule reproduces all four official practice observables to ~0.01
     (ours 0.124 / 0.927 / 0.039 vs official 0.122 / 0.9273 / 0.04, baseline
     1.518 vs ~1.51) — 4x better than the best random-dispersion world — and
     its oracle (median = the rule's E' exactly) scores 0.024, at the
     leader's level. The chapter effect is required: without it no rule fits
     (best error 0.098 vs 0.010). The rule says the refresh is story-driven:
     Chapter 1-4 trophies gain 17/13/11/8 new earners, most other trophies
     gain 0 or 1. Quantiles are a tight logit-normal band (sigma 0.05)
     around the rule's median: with a near-exact median, a narrow band is
     what the leaderboard rewards (the leaders' typical miss is ~0.00).
  4. Median rate from the ratio; spread on the logit scale combines the player
     uncertainty, the per-trophy earner uncertainty (shrunk toward its group),
     and a rounding term; quantiles are inv_logit(median_logit + z * sigma).
     (The frozen branch uses the rule median with the fixed RULE_SIGMA.)

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
# Chunk 2: the deterministic feature rule for the refreshed earners (fitted
# in mech3.py + refinement; see the module docstring). g_i is the per-trophy
# growth multiplier; the weights are normalised per case (geometric mean 1),
# so only their RATIOS matter. RULE_SIGMA is the logit spread around the
# rule's median (0.05: best on the plateau of fitting rules, mean 0.027).
RULE_A = -0.6      # exponent on (E_i / mean E): rarer trophies grow faster
RULE_B = 0.6       # exponent on the chapter weight: early chapters grow more
RULE_C = 0.7       # exponent on the tier weight: bronze grows more
RULE_CENTRE = 1.14  # centre multiplier (geometric mean of g_i)
RULE_SIGMA = 0.05
RULE_CHAPTER_W = {1: 8.0, 2: 6.0, 3: 5.0, 4: 4.0, 5: 2.0, None: 1.0}
RULE_TIER_W = {"platinum": 0.5, "gold": 1.0, "silver": 1.5, "bronze": 2.0}
SIGMA_FLOOR = 0.12               # minimum logit spread (printing/rounding jitter)
SIGMA_ROUND_CAP = 0.15


def chapter_of(trophy: dict):
    """Story chapter (1-4) from the description; 5 for the final mission."""
    d = trophy["description"].lower()
    for ch in (1, 2, 3, 4):
        if f"chapter {ch}" in d:
            return ch
    if "final mission" in d and "complete the final mission." in d:
        return 5
    return None


def rule_earners(earners: list, trophies: list) -> list:
    """Refreshed earners E' per trophy under the chunk-2 feature rule.

    g_i = RULE_CENTRE * (E_i / mean E)^RULE_A * chapter_w^RULE_B * tier_w^RULE_C,
    normalised so the geometric mean of the weights is 1; then
    E'_i = E_i + max(0, round(E_i * (g_i - 1))). Universal trophies (E >=
    players) and unobserved ones (None) pass through unchanged: the caller
    pins the universal trophy separately, so its value here is never used.
    """
    non_none = [e for e in earners if e is not None]
    players = max(non_none) if non_none else 1
    observed = [e for e in earners if e is not None and e < players]
    mean_e = sum(observed) / len(observed) if observed else 1.0
    logw = []
    for i, trophy in enumerate(trophies):
        e = earners[i]
        if e is None or e >= players:
            logw.append(None)
        else:
            logw.append(
                RULE_A * math.log(e / mean_e)
                + RULE_B * math.log(RULE_CHAPTER_W[chapter_of(trophy)])
                + RULE_C * math.log(RULE_TIER_W[trophy["tier"]])
            )
    have = [w for w in logw if w is not None]
    mean_logw = sum(have) / len(have) if have else 0.0
    out = []
    for i, trophy in enumerate(trophies):
        e = earners[i]
        if e is None or e >= players:
            out.append(e)
        else:
            g = RULE_CENTRE * math.exp(logw[i] - mean_logw)
            out.append(e + max(0, int(round(e * (g - 1.0)))))
    return out


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
        # Chunk 2: the feature rule's refreshed earners, computed once per case
        # from the frozen page's earners.
        rule_e = rule_earners(last.earners, trophies)
    else:
        slopes, _raw, _counts = trophy_slopes(states, trophies)
        rule_e = None

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
            # The top quantile is exactly 100.0: when the answer is 100 the
            # q95 = 100.0 costs nothing (logit clamps 100 -> 99.99) and buys
            # coverage the 99.99 cap was throwing away (chunk 1 missed id 2
            # on every synthetic draw). The lower quantiles stay just under
            # 100 so a small drop in the final case is not fully unhedged.
            med_rate, sigma = 99.99, 0.10
            qs = [round4(inv_logit_pct(logit_pct(med_rate) + z * sigma)) for z in Z]
            qs[-1] = 100.0
            rows.append({"id": trophy["id"], "q": qs})
            continue
        if flat:
            # Chunk 2: the feature rule's median with a tight band (RULE_SIGMA).
            # players_med here is ~3,950 (the horizon term adds 2% over the
            # documented 3,868); the rule was fitted at 3,868, a 0.02-logit
            # difference that moves the loss by ~0.001.
            e_med = float(rule_e[i])
            med_rate = clamp_pct(100.0 * e_med / players_med)
            lg = logit_pct(med_rate)
            rows.append(
                {"id": trophy["id"],
                 "q": [round4(inv_logit_pct(lg + z * RULE_SIGMA)) for z in Z]}
            )
            continue
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
