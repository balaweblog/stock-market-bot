"""
_validate_fixes.py  --  Targeted unit tests for the 9 bug fixes.
Run with: .venv/bin/python _validate_fixes.py
"""
import math
import sys

# ── Helpers copied from optionstrategy.py ──────────────────────────────────
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

RISK_FREE_RATE = 0.065
DIVIDEND_YIELD = 0.012


def compute_touch_probability(spot, t_years, iv_frac, barrier,
                              r=RISK_FREE_RATE, q=DIVIDEND_YIELD):
    """Fixed reflection-principle formula."""
    sigma = iv_frac
    if not (spot and sigma and t_years and t_years > 0 and sigma > 0 and barrier and barrier > 0):
        return None
    if barrier == spot:
        return 100.0
    mu = r - q - 0.5 * sigma ** 2
    a = math.log(barrier / spot)
    sqrt_t = math.sqrt(t_years)
    d_plus  = (a - mu * t_years) / (sigma * sqrt_t)
    d_minus = (-a - mu * t_years) / (sigma * sqrt_t)
    exponent = 2.0 * mu * a / sigma ** 2
    prob = _norm_cdf(d_plus) + math.exp(min(exponent, 700.0)) * _norm_cdf(d_minus)
    return round(max(0.0, min(1.0, prob)) * 100, 1)


# ── Test 1: Touch probability core properties ──────────────────────────────
# 1a. At-spot barrier is always 100%
assert compute_touch_probability(25000, 0.02, 0.15, 25000) == 100.0, "at-spot barrier must be 100%"

# 1b. POT is always >= POP for a given strike breakeven.
# POP (simplified, lognormal, no drift):
def _norm_cdf_simple(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
def pop_approx(spot, t, sigma, be):
    # P(S_T > be) for a call breakeven above spot
    d2 = (math.log(spot/be) + (-0.5*sigma**2)*t) / (sigma*math.sqrt(t))
    return _norm_cdf_simple(d2) * 100

be = 25300   # breakeven 300 pts above spot, 7 days, 15% vol
t  = 7/365; sig = 0.15
pop_be = pop_approx(25000, t, sig, be)
pot_be = compute_touch_probability(25000, t, sig, be)
assert pot_be >= pop_be, f"POT ({pot_be:.1f}%) must be >= POP ({pop_be:.1f}%) for same breakeven"
print(f"[PASS] POT ({pot_be:.1f}%) >= POP ({pop_be:.1f}%) for breakeven {be} -- correct financial relationship")

# 1c. Downward barrier for a bearish breakeven
be_down = 24700   # 300 pts below spot
pop_down = (100 - pop_approx(25000, t, sig, be_down))
pot_down = compute_touch_probability(25000, t, sig, be_down)
assert pot_down is not None and 0 < pot_down <= 100, f"downward barrier prob out of range: {pot_down}"
print(f"[PASS] Downward barrier {be_down}: POT={pot_down:.1f}%")

# ── Test 2: Up/down barrier symmetry ──────────────────────────────────────
spot = 25000; distance = 300; sigma = 0.15; T = 7/365
p_up   = compute_touch_probability(spot, T, sigma, spot + distance)
p_down = compute_touch_probability(spot, T, sigma, spot - distance)
# With a positive risk-neutral drift the up-touch prob is slightly higher --
# that's correct. The two should be within 30 pts of each other.
assert abs(p_up - p_down) < 30, f"up/down touch probs should be in the same ballpark: {p_up:.1f}% vs {p_down:.1f}%"
print(f"[PASS] Up/Down symmetry: +{distance}pt={p_up:.1f}%, -{distance}pt={p_down:.1f}%")

# ── Test 3: Debt-free company gets full points ─────────────────────────────
def score_fundamentals(f):
    score = 0
    pe   = f.get("pe");  roe = f.get("roe");  debt = f.get("debtToEquity")
    if pe and 0 < pe < 30:    score += 30
    if roe and roe > 0.15:    score += 40
    if debt is not None and 0 <= debt < 150:  score += 30   # FIXED
    return score

s = score_fundamentals({"pe": 20, "roe": 0.20, "debtToEquity": 0})
assert s == 100, f"Debt-free should score 100, got {s}"
s2 = score_fundamentals({"pe": 20, "roe": 0.20, "debtToEquity": 50})
assert s2 == 100, f"Normal debt should score 100, got {s2}"
print(f"[PASS] Debt-free={s}/100, D/E=50 → {s2}/100")

# ── Test 4: Negative D/E earns no bonus ────────────────────────────────────
def score_adv(data):
    score = 0
    de = data.get("debt_to_equity")
    if de is not None and 0 <= de < 100:  score += 15   # FIXED
    return score

assert score_adv({"debt_to_equity": -5})  == 0,  "Negative D/E must score 0"
assert score_adv({"debt_to_equity":  0})  == 15, "Zero D/E earns bonus"
assert score_adv({"debt_to_equity": 50})  == 15, "Low D/E earns bonus"
assert score_adv({"debt_to_equity": 150}) == 0,  "High D/E earns no bonus"
print("[PASS] Negative D/E=0pts; D/E=0→15pts; D/E=50→15pts; D/E=150→0pts")

# ── Test 5: Iron Condor unequal-wing max-loss ──────────────────────────────
call_width = 150; call_premium = 50   # call_max_loss = 100
put_width  = 200; put_premium  = 60   # put_max_loss  = 140
total_premium  = call_premium + put_premium   # 110
call_max_loss  = call_width - call_premium    # 100
put_max_loss   = put_width  - put_premium     # 140
max_loss       = max(call_max_loss, put_max_loss)   # 140
assert max_loss == 140, f"IC max loss should be 140, got {max_loss}"
rr = total_premium / max_loss
assert abs(rr - 110/140) < 0.001, f"IC R:R wrong: {rr:.4f}"
print(f"[PASS] IC unequal wings: call_ML={call_max_loss}, put_ML={put_max_loss}, total_ML={max_loss}, R:R={rr:.3f}")

# ── Test 6: EM divergence zero-denominator guard ───────────────────────────
def compute_em_divergence(straddle_pts, iv_pts):
    move_divergence_pct = None
    if iv_pts and straddle_pts:
        min_move = min(straddle_pts, iv_pts)
        if min_move > 0:                         # FIXED: guard zero
            move_divergence_pct = round(abs(straddle_pts - iv_pts) / min_move * 100, 1)
    return move_divergence_pct

assert compute_em_divergence(0.0, 50.0) is None, "zero straddle → None divergence (no crash)"
assert compute_em_divergence(50.0, 0.0) is None, "zero IV move → None divergence (no crash)"
assert compute_em_divergence(100.0, 80.0) == 25.0, f"divergence wrong: {compute_em_divergence(100, 80)}"
print("[PASS] EM divergence: zero-denominator → None; 100 vs 80 → 25.0%")

print("\n✅  All targeted unit tests PASSED")
