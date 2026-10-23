"""Tests of the Module 6 check: correct learner solutions must pass, typical wrong ones must fail with a message.
Run from the repository root: python checks/test_dt_m6_check.py"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m6"), str(ROOT / "checks")]
from dt_m6_check import check_dt_module6  # noqa: E402

NAN = float("nan")


# ---------------------------------------------------------------- throughput(t_out, t_start, t_end)
def t_good1(t_out, a, b):
    t = np.asarray(t_out, dtype=float)
    return np.sum((t >= a) & (t < b)) / (b - a)
def t_good2(t_out, a, b):
    n = 0
    for t in t_out:
        if t >= a and t < b:
            n += 1
    return n / (b - a)
def t_good3(t_out, a, b):
    s = t_out[t_out.notna()] if hasattr(t_out, "notna") else t_out[~np.isnan(t_out)]
    return float(((s >= a) & (s < b)).sum()) / (b - a)
T_GOOD = [t_good1, t_good2, t_good3]
T_BAD = {
    "count only": lambda t, a, b: np.sum((np.asarray(t) >= a) & (np.asarray(t) < b)),
    "divides by t_end": lambda t, a, b: np.sum((np.asarray(t) >= a) & (np.asarray(t) < b)) / b,
    "includes t_end": lambda t, a, b: np.sum((np.asarray(t) >= a) & (np.asarray(t) <= b)) / (b - a),
    "excludes t_start": lambda t, a, b: np.sum((np.asarray(t) > a) & (np.asarray(t) < b)) / (b - a),
    "all exits": lambda t, a, b: np.sum(~np.isnan(np.asarray(t, dtype=float))) / (b - a),
    "len / window": lambda t, a, b: len(t) / (b - a),
    "returns None": lambda t, a, b: None,
}
def t_todo(t, a, b):
    raise NotImplementedError("Complete throughput() first.")
T_BAD["not written"] = t_todo


# ---------------------------------------------------------------- mean_cycle_time(t_in, t_out, t_start, t_end)
def c_good1(ti, to, a, b):
    ti, to = np.asarray(ti, dtype=float), np.asarray(to, dtype=float)
    m = (to >= a) & (to < b)
    return float(np.mean(to[m] - ti[m])) if m.any() else float("nan")
def c_good2(ti, to, a, b):
    ct = [o - i for i, o in zip(ti, to) if o >= a and o < b]
    return sum(ct) / len(ct) if ct else float("nan")
def c_good3(ti, to, a, b):
    ti, to = np.asarray(ti, dtype=float), np.asarray(to, dtype=float)
    m = (to >= a) & (to < b)
    return np.nan if m.sum() == 0 else (to[m] - ti[m]).mean()
C_GOOD = [c_good1, c_good2, c_good3]
def _cv(sel, empty=float("nan")):
    def f(ti, to, a, b):
        ti, to = np.asarray(ti, dtype=float), np.asarray(to, dtype=float)
        with np.errstate(invalid="ignore"):
            m = sel(ti, to, a, b)
        return float(np.mean(to[m] - ti[m])) if m.any() else empty
    return f
C_BAD = {
    "by arrival": _cv(lambda ti, to, a, b: (ti >= a) & (ti < b) & ~np.isnan(to)),
    "ignores window": _cv(lambda ti, to, a, b: ~np.isnan(to)),
    "includes t_end": _cv(lambda ti, to, a, b: (to >= a) & (to <= b)),
    "excludes t_start": _cv(lambda ti, to, a, b: (to > a) & (to < b)),
    "nan not dropped": lambda ti, to, a, b: float(np.mean(np.asarray(to, dtype=float)[(np.asarray(ti, dtype=float) < b)] - np.asarray(ti, dtype=float)[(np.asarray(ti, dtype=float) < b)])),
    "0 for empty": _cv(lambda ti, to, a, b: (to >= a) & (to < b), empty=0.0),
    "mean exit times": lambda ti, to, a, b: float(np.mean(np.asarray(to, dtype=float)[(np.asarray(to, dtype=float) >= a) & (np.asarray(to, dtype=float) < b)])) if ((np.asarray(to, dtype=float) >= a) & (np.asarray(to, dtype=float) < b)).any() else NAN,
}
def c_todo(ti, to, a, b):
    raise NotImplementedError("Complete mean_cycle_time() first.")
C_BAD["not written"] = c_todo


# ---------------------------------------------------------------- time_average_wip(t_in, t_out, t_start, t_end)
def w_good1(ti, to, a, b):
    ti, to = np.asarray(ti, dtype=float), np.asarray(to, dtype=float)
    to = np.where(np.isnan(to), np.inf, to)
    return float(np.clip(np.minimum(to, b) - np.maximum(ti, a), 0, None).sum() / (b - a))
def w_good2(ti, to, a, b):
    total = 0.0
    for i, o in zip(ti, to):
        end = b if np.isnan(o) else min(o, b)
        start = max(i, a)
        if end > start:
            total += end - start
    return total / (b - a)
def w_good3(ti, to, a, b):
    ti, to = np.asarray(ti, dtype=float), np.asarray(to, dtype=float)
    end = np.minimum(np.nan_to_num(to, nan=np.inf), b)
    inside = np.maximum(end - np.maximum(ti, a), 0.0)
    return inside.sum() / (b - a)
W_GOOD = [w_good1, w_good2, w_good3]
def _w_bad_finished_only(ti, to, a, b):
    ti, to = np.asarray(ti, dtype=float), np.asarray(to, dtype=float)
    f = ~np.isnan(to)
    return float(np.clip(np.minimum(to[f], b) - np.maximum(ti[f], a), 0, None).sum() / (b - a))
def _w_bad_unclipped(ti, to, a, b):
    ti, to = np.asarray(ti, dtype=float), np.asarray(to, dtype=float)
    to = np.where(np.isnan(to), np.inf, to)
    return float((np.minimum(to, b) - np.maximum(ti, a)).sum() / (b - a))
def _w_bad_whole_cycle(ti, to, a, b):
    ti, to = np.asarray(ti, dtype=float), np.asarray(to, dtype=float)
    m = (to >= a) & (to < b)
    return float((to[m] - ti[m]).sum() / (b - a))
def _w_bad_at_start(ti, to, a, b):
    ti, to = np.asarray(ti, dtype=float), np.asarray(to, dtype=float)
    to = np.where(np.isnan(to), np.inf, to)
    return float(np.sum((ti <= a) & (to > a)))
W_BAD = {
    "nan not handled": lambda ti, to, a, b: float((np.clip(np.minimum(np.asarray(to, dtype=float), b) - np.maximum(np.asarray(ti, dtype=float), a), 0, None)).sum() / (b - a)),
    "finished lots only": _w_bad_finished_only,
    "unclipped overlap": _w_bad_unclipped,
    "whole cycle times": _w_bad_whole_cycle,
    "lots at the start": _w_bad_at_start,
    "divides by t_end": lambda ti, to, a, b: w_good1(ti, to, a, b) * (b - a) / b,
    "returns None": lambda ti, to, a, b: None,
}
def w_todo(ti, to, a, b):
    raise NotImplementedError("Complete time_average_wip() first.")
W_BAD["not written"] = w_todo


# ---------------------------------------------------------------- ci95(values)
def i_good1(v):
    x = np.asarray(v, dtype=float)
    n = len(x)
    return x.mean(), stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
def i_good2(v):
    x = list(v)
    n = len(x)
    m = sum(x) / n
    s = math.sqrt(sum((a - m) ** 2 for a in x) / (n - 1))
    return m, float(stats.t.ppf(0.975, df=n - 1)) * s / math.sqrt(n)
def i_good3(v):
    x = np.array(v, dtype=float)
    se = stats.sem(x)
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=se)
    return float(x.mean()), float((hi - lo) / 2)
I_GOOD = [i_good1, i_good2, i_good3]
def _ci(fn):
    def f(v):
        x = np.asarray(v, dtype=float)
        return float(x.mean()), float(fn(x, len(x)))
    return f
I_BAD = {
    "ddof 0": _ci(lambda x, n: stats.t.ppf(0.975, n - 1) * x.std() / np.sqrt(n)),
    "1.96": _ci(lambda x, n: 1.96 * x.std(ddof=1) / np.sqrt(n)),
    "no sqrt(n)": _ci(lambda x, n: stats.t.ppf(0.975, n - 1) * x.std(ddof=1)),
    "standard error only": _ci(lambda x, n: x.std(ddof=1) / np.sqrt(n)),
    "standard deviation": _ci(lambda x, n: x.std(ddof=1)),
    "0.95 quantile": _ci(lambda x, n: stats.t.ppf(0.95, n - 1) * x.std(ddof=1) / np.sqrt(n)),
    "n degrees of freedom": _ci(lambda x, n: stats.t.ppf(0.975, n) * x.std(ddof=1) / np.sqrt(n)),
    "interval ends": lambda v: (float(np.mean(v)) - float(stats.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v))),
                                float(np.mean(v)) + float(stats.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v)))),
    "half-width only": lambda v: float(stats.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v))),
}
def i_todo(v):
    raise NotImplementedError("Complete ci95() first.")
I_BAD["not written"] = i_todo


# ---------------------------------------------------------------- calibrate_now(recent, limit, k)
def k_good1(r, limit, k):
    return len(r) >= k and all(abs(v) > limit for v in r[-k:])
def k_good2(r, limit, k):
    if len(r) < k:
        return False
    for v in list(r)[-k:]:
        if abs(v) <= limit:
            return False
    return True
def k_good3(r, limit, k):
    r = np.asarray(r, dtype=float)
    return bool(r.size >= k and np.all(np.abs(r[-k:]) > limit))
K_GOOD = [k_good1, k_good2, k_good3]
K_BAD = {
    "any": lambda r, limit, k: len(r) >= k and any(abs(v) > limit for v in r[-k:]),
    "no abs": lambda r, limit, k: len(r) >= k and all(v > limit for v in r[-k:]),
    "greater or equal": lambda r, limit, k: len(r) >= k and all(abs(v) >= limit for v in r[-k:]),
    "first k": lambda r, limit, k: len(r) >= k and all(abs(v) > limit for v in r[:k]),
    "too few values": lambda r, limit, k: all(abs(v) > limit for v in r[-k:]) and len(r) > 0,
    "last value only": lambda r, limit, k: len(r) >= k and abs(r[-1]) > limit,
    "returns None": lambda r, limit, k: None,
    "returns 1 or 0": lambda r, limit, k: 1 if (len(r) >= k and all(abs(v) > limit for v in r[-k:])) else 0,
}
def k_todo(r, limit, k):
    raise NotImplementedError("Complete calibrate_now() first.")
K_BAD["not written"] = k_todo

NAMES = ["throughput", "mean_cycle_time", "time_average_wip", "ci95", "calibrate_now"]


def main():
    slots = [T_GOOD, C_GOOD, W_GOOD, I_GOOD, K_GOOD]
    good = list(itertools.product(*slots))
    wrong_verdicts = 0
    for combo in good:
        ok, msgs, val = check_dt_module6(*combo, 417)
        if not ok or val is None or abs(val - __import__('dt_m6_check').verification_code(417)) > 1e-9:
            wrong_verdicts += 1
            print("GOOD SOLUTION FAILED:", [f.__name__ for f in combo], msgs)
    n_bad = 0
    sets = [tuple(g[i] for g in slots) for i in range(3)]
    for slot, bad in enumerate([T_BAD, C_BAD, W_BAD, I_BAD, K_BAD]):
        for name, fn in bad.items():
            for base in sets:
                fns = list(base)
                fns[slot] = fn
                ok, msgs, val = check_dt_module6(*fns, 417)
                n_bad += 1
                if ok or val is not None:
                    wrong_verdicts += 1
                    print("WRONG SOLUTION PASSED:", NAMES[slot], name)
                elif base is sets[0]:
                    print(f"  [{NAMES[slot]}: {name}]", [m for m in msgs if m.startswith("NOT")][0][:200])
    print(f"{len(good)} good and {n_bad} wrong combinations, {wrong_verdicts} wrong verdicts")
    for bad_key in (None, "abc", 50, 2026):
        ok, msgs, val = check_dt_module6(*sets[0], bad_key)
        assert not ok and val is None, bad_key
    print("KEY guards OK")
    return wrong_verdicts


def bypass_test():
    """The verification value is not the result the notebook prints earlier."""
    from dt_m6_check import personal_value as _result, verification_code as _code  # noqa: E402
    for _k in (417,):
        assert abs(_code(_k) - _result(_k)) > 0.0005, "the verification value must differ from the result printed before the check"
    print("bypass test passed: the verification value differs from the printed result for KEYs", (417,))


if __name__ == "__main__":
    bypass_test()
    sys.exit(1 if main() else 0)

