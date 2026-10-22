"""Tests of the Module 5 check: correct learner solutions must pass, typical wrong ones must fail with a message.
Run from the repository root: python checks/test_dt_m5_check.py"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m5"), str(ROOT / "checks")]
from dt_m5_check import check_dt_module5  # noqa: E402


def _nan(y):
    return y is None or (isinstance(y, float) and math.isnan(y))


# ---------------------------------------------------------------- EWMA
def e_good1(d, y, lam):
    if np.isnan(y):
        return d
    return lam * y + (1 - lam) * d
def e_good2(d, y, lam):
    if y != y:
        return d
    return d + lam * (y - d)
def e_good3(d, y, lam):
    return d if math.isnan(y) else np.float64(lam * y + (1 - lam) * d)
E_GOOD = [e_good1, e_good2, e_good3]
E_BAD = {
    "swapped": lambda d, y, lam: d if np.isnan(y) else lam * d + (1 - lam) * y,
    "returns y": lambda d, y, lam: d if np.isnan(y) else y,
    "nan not handled": lambda d, y, lam: lam * y + (1 - lam) * d,
    "nan gives 0": lambda d, y, lam: 0.0 if np.isnan(y) else lam * y + (1 - lam) * d,
    "nan gives None": lambda d, y, lam: None if np.isnan(y) else lam * y + (1 - lam) * d,
    "tuple": lambda d, y, lam: (d if np.isnan(y) else lam * y + (1 - lam) * d, lam),
    "lam fixed": lambda d, y, lam: d if np.isnan(y) else 0.3 * y + 0.7 * d,
}
def e_todo(d, y, lam):
    raise NotImplementedError("Complete ewma_update() first.")
E_BAD["not written"] = e_todo


# ---------------------------------------------------------------- Kalman
def k_good1(d, P, y, Q, R):
    Pm = P + Q
    if np.isnan(y):
        return d, Pm
    K = Pm / (Pm + R)
    return d + K * (y - d), (1 - K) * Pm
def k_good2(d, P, y, Q, R):
    Pm = P + Q
    if y != y:
        return [d, Pm]
    K = Pm / (Pm + R)
    return [(1 - K) * d + K * y, Pm * R / (Pm + R)]
def k_good3(d, P, y, Q, R):
    Pm = np.array([P + Q])
    if math.isnan(y):
        return np.array([d]), Pm
    K = Pm / (Pm + R)
    return d + K * (y - d), (1 - K) * Pm
K_GOOD = [k_good1, k_good2, k_good3]
def _k(fn_gain, fn_P, missing=None):
    def f(d, P, y, Q, R):
        Pm = P + Q
        if np.isnan(y):
            return missing(d, P, Q) if missing else (d, Pm)
        K = fn_gain(P, Pm, R)
        return d + K * (y - d), fn_P(K, P, Pm, Q)
    return f
K_BAD = {
    "gain without Q": _k(lambda P, Pm, R: P / (P + R), lambda K, P, Pm, Q: (1 - K) * Pm),
    "P from P_prev": _k(lambda P, Pm, R: Pm / (Pm + R), lambda K, P, Pm, Q: (1 - K) * P),
    "Q added twice": _k(lambda P, Pm, R: Pm / (Pm + R), lambda K, P, Pm, Q: (1 - K) * Pm + Q),
    "missing keeps P": _k(lambda P, Pm, R: Pm / (Pm + R), lambda K, P, Pm, Q: (1 - K) * Pm, missing=lambda d, P, Q: (d, P)),
    "missing nan": _k(lambda P, Pm, R: Pm / (Pm + R), lambda K, P, Pm, Q: (1 - K) * Pm, missing=lambda d, P, Q: (float("nan"), P + Q)),
    "gain R over": _k(lambda P, Pm, R: R / (Pm + R), lambda K, P, Pm, Q: (1 - K) * Pm),
    "swapped": lambda d, P, y, Q, R: tuple(reversed(k_good1(d, P, y, Q, R))),
    "only d": lambda d, P, y, Q, R: k_good1(d, P, y, Q, R)[0],
}


# ---------------------------------------------------------------- next_time
def t_good1(d, tgt, B, BA, xi, lim):
    x = tgt - d
    A = B / BA
    t = (x ** 2 - xi ** 2 + A * (x - xi)) / B
    return min(max(t, lim[0]), lim[1])
def t_good2(d, tgt, B, BA, xi, lim):
    A = B / BA
    tau = (xi ** 2 + A * xi) / B
    x = tgt - d
    return float(np.clip((x * x + A * x) / B - tau, *lim))
def t_good3(d, tgt, B, BA, xi, lim):
    x = tgt - d
    t = (x - xi) * (x + xi) / B + (x - xi) / BA
    lo, hi = lim
    return lo if t < lo else (hi if t > hi else t)
T_GOOD = [t_good1, t_good2, t_good3]
def _t(expr, clip=True, fixed=False):
    def f(d, tgt, B, BA, xi, lim):
        t = expr(d, tgt, B, BA, xi)
        if not clip:
            return t
        lo, hi = (80.0, 100.0) if fixed else lim
        return min(max(t, lo), hi)
    return f
_free = lambda d, tgt, B, BA, xi: ((tgt - d) ** 2 - xi ** 2 + B / BA * ((tgt - d) - xi)) / B
T_BAD = {
    "no clip": _t(_free, clip=False),
    "clip 80/100": _t(_free, fixed=True),
    "target plus d": _t(lambda d, tgt, B, BA, xi: _free(-d, tgt, B, BA, xi)),
    "no xi": _t(lambda d, tgt, B, BA, xi: ((tgt - d) ** 2 + B / BA * (tgt - d)) / B),
    "A inverted": _t(lambda d, tgt, B, BA, xi: ((tgt - d) ** 2 - xi ** 2 + BA / B * ((tgt - d) - xi)) / B),
    "seconds": lambda *a: 60.0 * t_good1(*a),
    "rounded": lambda *a: round(t_good1(*a)),
    "tuple": lambda *a: (t_good1(*a), False),
}


# ---------------------------------------------------------------- alarm_run
def a_good1(offsets, limit, k):
    c = 0
    for i, v in enumerate(offsets):
        c = c + 1 if abs(v) > limit else 0
        if c == k:
            return i + 1
    return None
def a_good2(offsets, limit, k):
    over = np.abs(np.asarray(offsets, dtype=float)) > limit
    run = 0
    for i in range(len(over)):
        run = run + 1 if over[i] else 0
        if run >= k:
            return i + 1
def a_good3(offsets, limit, k):
    vals = list(offsets)
    for end in range(k, len(vals) + 1):
        if all(abs(v) > limit for v in vals[end - k:end]):
            return end
    return None
A_GOOD = [a_good1, a_good2, a_good3]
def _a(cmp, reset=True, base=1, first=False, none=None):
    def f(offsets, limit, k):
        c = 0
        for i, v in enumerate(offsets):
            if cmp(v, limit):
                c += 1
            elif reset:
                c = 0
            if c >= k:
                return (i - k + 1 + base) if first else (i + base)
        return none
    return f
A_BAD = {
    "greater or equal": _a(lambda v, L: abs(v) >= L),
    "no reset": _a(lambda v, L: abs(v) > L, reset=False),
    "0-based": _a(lambda v, L: abs(v) > L, base=0),
    "first of streak": _a(lambda v, L: abs(v) > L, first=True),
    "no abs": _a(lambda v, L: v > L),
    "0 for none": _a(lambda v, L: abs(v) > L, none=0),
    "-1 for none": _a(lambda v, L: abs(v) > L, none=-1),
}


def main():
    good = list(itertools.product(E_GOOD, K_GOOD, T_GOOD, A_GOOD))
    wrong_verdicts = 0
    for e, k, t, a in good:
        ok, msgs, val = check_dt_module5(e, k, t, a, 417)
        if not ok or val is None:
            wrong_verdicts += 1
            print("GOOD SOLUTION FAILED:", e.__name__, k.__name__, t.__name__, a.__name__, msgs)
    n_bad = 0
    sets = [(E_GOOD[0], K_GOOD[0], T_GOOD[0], A_GOOD[0]), (E_GOOD[1], K_GOOD[1], T_GOOD[1], A_GOOD[1]), (E_GOOD[2], K_GOOD[2], T_GOOD[2], A_GOOD[2])]
    for slot, bad in enumerate([E_BAD, K_BAD, T_BAD, A_BAD]):
        for name, fn in bad.items():
            for base in sets:
                fns = list(base)
                fns[slot] = fn
                ok, msgs, val = check_dt_module5(*fns, 417)
                n_bad += 1
                if ok or val is not None:
                    wrong_verdicts += 1
                    print("WRONG SOLUTION PASSED:", slot, name)
                elif base is sets[0]:
                    print(f"  [{['ewma', 'kalman', 'next_time', 'alarm'][slot]}: {name}]", [m for m in msgs if m.startswith("NOT")][0][:170])
    print(f"{len(good)} good and {n_bad} wrong combinations, {wrong_verdicts} wrong verdicts")
    for bad_key in (None, "abc", 50, 2026):
        ok, msgs, val = check_dt_module5(*sets[0], bad_key)
        assert not ok and val is None, bad_key
    print("KEY guards OK")
    return wrong_verdicts


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
