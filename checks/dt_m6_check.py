"""Check cell logic for the Module 6 lab (Digital Twins for Semiconductor Manufacturing).

check_dt_module6(throughput, mean_cycle_time, time_average_wip, ci95, calibrate_now, KEY) returns (passed, messages, verification_value).

The learner writes
  throughput(t_out, t_start, t_end)          lots that left the line in the window [t_start, t_end), per hour; t_out is a NumPy array or
                                             pandas Series of exit times in hours, NaN for lots still in the line;
  mean_cycle_time(t_in, t_out, t_start, t_end)
                                             the mean of t_out - t_in over the lots that left in the window [t_start, t_end);
                                             NaN when no lot did;
  time_average_wip(t_in, t_out, t_start, t_end)
                                             the average number of lots in the line over [t_start, t_end]: the hours every lot
                                             spends inside the window, summed, divided by the window length; NaN in t_out means
                                             the lot is still in the line;
  ci95(values)                               (mean, half_width) of a 95 % confidence interval of the mean of independent
                                             replications: half_width = t(0.975, n - 1) * s / sqrt(n), s with n - 1 in the divisor;
  calibrate_now(recent, limit, k)            True when the last k values of recent (the drift estimates since the last calibration,
                                             latest last) are ALL beyond the limit in size: abs(value) > limit; False when recent has
                                             fewer than k values.

The learner's result is the mean cycle-time reduction, in hours to 3 decimals, when lithography is 10 % faster (litho_h 1.0 -> 0.9),
over 20 replications with seeds KEY*1000 + 1 ... KEY*1000 + 20 (1,000 h each, the first 200 h discarded), the same seeds for both
lines. It is computed with the course's reference functions and printed only when every test passes.

The verification value that the check prints is not the result itself, which the notebook shows earlier as a learning
output. It is a code made from the result and the KEY: (A * round(1000 * result) + B * KEY + C) mod 9973, divided by
1000, to 3 decimals, with A = 19, B = 36, C = 601 for this module. It is printed only when every test passes, so the
result alone does not answer Question 1. It shows that the check passed for this KEY; it is not proof of authorship.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

import line_tools as lt
from line import line_config

HOURS, WARMUP_H = lt.HOURS, lt.WARMUP_H


# ---------------------------------------------------------------- reference functions
def reference_throughput(t_out, t_start, t_end):
    return lt.reference_throughput(t_out, t_start, t_end)


def reference_cycle_time(t_in, t_out, t_start, t_end):
    return lt.reference_cycle_time(t_in, t_out, t_start, t_end)


def reference_wip(t_in, t_out, t_start, t_end):
    return lt.reference_wip(t_in, t_out, t_start, t_end)


def reference_ci95(values):
    x = np.asarray(values, dtype=float)
    n = len(x)
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n))


def reference_calibrate(recent, limit, k):
    recent = list(recent)
    return len(recent) >= k and all(abs(v) > limit for v in recent[-k:])


def personal_seeds(key):
    return tuple(int(key) * 1000 + i for i in range(1, 21))


def personal_differences(key):
    """The 20 paired reductions in mean cycle time (h): the baseline line minus the line with a 10 % faster lithography tool."""
    seeds = personal_seeds(key)
    base = lt.run_replications(None, seeds)
    fast = lt.run_replications(line_config(litho_h=0.9), seeds)
    return [reference_cycle_time(b.lots["t_in"], b.lots["t_out"], WARMUP_H, HOURS) - reference_cycle_time(f.lots["t_in"], f.lots["t_out"], WARMUP_H, HOURS)
            for b, f in zip(base, fast)]


def personal_raw(key):
    return float(np.mean(personal_differences(key)))


def personal_value(key):
    return round(personal_raw(key), 3)


CODE_A, CODE_B, CODE_C = 19, 36, 601


def verification_code(key):
    """The verification value: a code made from the learner's result and the KEY, printed only when every test passes."""
    n = int(round(personal_value(key) * 1000))
    return ((CODE_A * n + CODE_B * int(key) + CODE_C) % 9973) / 1000


# ---------------------------------------------------------------- helpers
def _call(fn, name, *args, allow_none=False):
    try:
        got = fn(*args)
    except NotImplementedError:
        return False, f"{name}() is not written yet: replace the raise line with your code."
    except Exception as e:
        return False, f"{name}() stopped with {type(e).__name__}: {e}"
    if got is None and not allow_none:
        return False, f"{name}() returned nothing (None): end it with a return line that gives back the result."
    return True, got


def _num(x):
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.size != 1:
        return None
    return float(a.reshape(-1)[0])


def _close(a, b, tol=1e-9):
    if a is None or b is None:
        return False
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol * max(1.0, abs(b))


NAN = float("nan")


def _wrap(x, kind):
    return pd.Series(x) if kind == "series" else np.array(x, dtype=float)


# ---------------------------------------------------------------- tests
def _test_throughput(throughput):
    cases = [([5.0, 12.0, 15.0, 18.0, 25.0, NAN, NAN], 10.0, 20.0),
             ([10.0, 20.0, 15.0, NAN, 14.0], 10.0, 20.0),                    # one exit exactly at the start, one exactly at the end
             ([1.0, 2.0, 3.0], 10.0, 20.0),
             ([200.0, 200.5, 201.0, 650.2, 999.9, 1000.0, NAN, NAN], 200.0, 1000.0),
             ([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 2.0, 5.0),
             ([NAN, NAN, NAN], 0.0, 10.0),
             (list(np.arange(50.0, 150.0, 0.7)) + [NAN] * 5, 80.0, 120.0)]
    for vals, t0, t1 in cases:
        for kind in ("array", "series"):
            arg = _wrap(vals, kind)
            ok, got = _call(throughput, "throughput", arg, t0, t1)
            if not ok:
                return False, got
            g = _num(got)
            if g is None:
                return False, f"throughput() returned {got!r}: return one number, lots per hour."
            want = reference_throughput(arg, t0, t1)
            if _close(g, want):
                continue
            v = np.asarray(vals, dtype=float)
            with np.errstate(invalid="ignore"):
                n_in = float(np.sum((v >= t0) & (v < t1)))
                if _close(g, n_in):
                    return False, "throughput() returns the number of lots. Throughput is a rate: divide by the window length, t_end - t_start."
                if _close(g, n_in / t1):
                    return False, "throughput() divides by t_end. The window runs from t_start to t_end: divide by t_end - t_start."
                if _close(g, np.sum((v >= t0) & (v <= t1)) / (t1 - t0)):
                    return False, ("throughput() counts a lot that leaves exactly at t_end. The window is [t_start, t_end): "
                                   "t_out >= t_start and t_out < t_end.")
                if _close(g, np.sum((v > t0) & (v < t1)) / (t1 - t0)):
                    return False, ("throughput() leaves out a lot that leaves exactly at t_start. The window is [t_start, t_end): "
                                   "count t_out >= t_start.")
                if _close(g, np.sum(~np.isnan(v)) / (t1 - t0)):
                    return False, "throughput() counts every lot that has left, whenever it left. Count only the lots with t_start <= t_out < t_end."
            return False, (f"throughput({vals}, {t0}, {t1}) returned {g:.6g}; it should be {want:.6g}: the lots with t_start <= t_out < t_end, "
                           "divided by t_end - t_start.")
    return True, "throughput() passed 14 tests, including exits exactly at the start and end of the window, and lots still in the line (NaN)."


def _test_cycle(mean_cycle_time):
    cases = [([0.0, 2.0, 5.0, 8.0, 9.0], [4.0, 7.0, 11.0, NAN, 12.5], 5.0, 12.0),    # arrives before the window, leaves inside; one still in the line
             ([1.0, 3.0, 6.0, 11.0], [6.0, 9.0, 13.0, 15.0], 6.0, 14.0),             # leaves exactly at the start, and after the end
             ([0.0, 1.0, 2.0], [3.0, 4.0, 5.0], 10.0, 20.0),                         # no lot leaves in the window: NaN
             ([100.0, 150.2, 190.0, 210.0, 230.0, 260.0], [180.0, 205.5, 215.0, 260.5, NAN, 300.0], 200.0, 300.0),
             ([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [2.0, 2.5, 3.5, 5.0, 6.0, NAN], 2.0, 5.5),
             ([4.0, 5.0], [9.0, 10.0], 9.0, 10.0)]
    for tin, tout, t0, t1 in cases:
        for kind in ("array", "series"):
            a_in, a_out = _wrap(tin, kind), _wrap(tout, kind)
            ok, got = _call(mean_cycle_time, "mean_cycle_time", a_in, a_out, t0, t1)
            if not ok:
                return False, got
            g = _num(got)
            if g is None:
                return False, f"mean_cycle_time() returned {got!r}: return one number, the mean cycle time in hours."
            want = reference_cycle_time(a_in, a_out, t0, t1)
            if _close(g, want):
                continue
            ti, to = np.asarray(tin, dtype=float), np.asarray(tout, dtype=float)
            if math.isnan(want):
                if g == 0.0:
                    return False, ("mean_cycle_time() returns 0 when no lot left in the window. There is no mean to report: "
                                   "return float('nan'), so that nothing is mistaken for a cycle time of 0 hours.")
                return False, f"mean_cycle_time() returned {got!r} for a window in which no lot left the line; it should be NaN (float('nan'))."
            if math.isnan(g):
                return False, ("mean_cycle_time() returned NaN. A lot that has not left the line has no cycle time yet: "
                               "average only the lots with t_start <= t_out < t_end, and NaN exit times are not in that window.")
            with np.errstate(invalid="ignore"):
                arr_sel = (ti >= t0) & (ti < t1) & ~np.isnan(to)
                if arr_sel.any() and _close(g, float(np.mean(to[arr_sel] - ti[arr_sel]))):
                    return False, ("mean_cycle_time() selects the lots by their ARRIVAL time. Select the lots that LEFT the line in the window "
                                   "(t_start <= t_out < t_end): their times are what the window measured.")
                fin = ~np.isnan(to)
                if _close(g, float(np.mean(to[fin] - ti[fin]))):
                    return False, "mean_cycle_time() ignores the window. Average only the lots with t_start <= t_out < t_end."
                inc = (to >= t0) & (to <= t1)
                if inc.any() and _close(g, float(np.mean(to[inc] - ti[inc]))):
                    return False, "mean_cycle_time() counts a lot that leaves exactly at t_end. The window is [t_start, t_end): t_out < t_end."
                exc = (to > t0) & (to < t1)
                if exc.any() and _close(g, float(np.mean(to[exc] - ti[exc]))):
                    return False, "mean_cycle_time() leaves out a lot that leaves exactly at t_start. The window is [t_start, t_end): t_out >= t_start."
            return False, (f"mean_cycle_time({tin}, {tout}, {t0}, {t1}) returned {g:.6g} h; it should be {want:.6g} h: "
                           "the mean of t_out - t_in over the lots with t_start <= t_out < t_end.")
    return True, "mean_cycle_time() passed 12 tests, including lots still in the line, lots that arrived before the window, and a window with no exits."


def _test_wip(time_average_wip):
    cases = [([0.0, 2.0, 5.0, 8.0], [4.0, 7.0, 11.0, NAN], 4.0, 9.0),
             ([0.0, 1.0, 2.0], [3.0, 4.0, 5.0], 10.0, 20.0),                           # nothing in the line in the window
             ([0.0, 1.0, 2.0], [30.0, 40.0, 50.0], 10.0, 20.0),                        # three lots in the line all the time
             ([5.0, 12.0, 14.0, 22.0], [15.0, 18.0, 30.0, NAN], 10.0, 20.0),
             ([100.0, 150.2, 190.0, 210.0, 230.0, 260.0], [180.0, 205.5, 215.0, 260.5, NAN, 300.0], 200.0, 300.0),
             ([0.5, 1.0, 3.0], [NAN, NAN, NAN], 2.0, 4.0),
             ([2.0, 4.0], [4.0, 8.0], 2.0, 8.0)]
    for tin, tout, t0, t1 in cases:
        for kind in ("array", "series"):
            a_in, a_out = _wrap(tin, kind), _wrap(tout, kind)
            ok, got = _call(time_average_wip, "time_average_wip", a_in, a_out, t0, t1)
            if not ok:
                return False, got
            g = _num(got)
            if g is None:
                return False, f"time_average_wip() returned {got!r}: return one number, the average number of lots in the line."
            want = reference_wip(a_in, a_out, t0, t1)
            if _close(g, want):
                continue
            ti, to = np.asarray(tin, dtype=float), np.asarray(tout, dtype=float)
            w = t1 - t0
            if math.isnan(g):
                return False, ("time_average_wip() returned NaN. A lot with no exit time is still in the line: "
                               "treat its exit time as later than t_end, for example with np.where(np.isnan(t_out), np.inf, t_out).")
            fin = ~np.isnan(to)
            hits = []
            with np.errstate(invalid="ignore"):
                o_fin = np.clip(np.minimum(to[fin], t1) - np.maximum(ti[fin], t0), 0, None).sum() / w
                if _close(g, o_fin):
                    hits.append("time_average_wip() leaves out the lots that are still in the line at the end of the run (NaN exit). "
                                "They are in the line until the end of the window: count their hours too.")
                m = (to >= t0) & (to < t1)
                if _close(g, float(np.sum(to[m] - ti[m])) / w):
                    hits.append("time_average_wip() adds up whole cycle times of the lots that left in the window. Count only the hours "
                                "each lot spends INSIDE the window: from max(t_in, t_start) to min(t_out, t_end), never below 0.")
                o_unclipped = (np.minimum(np.where(np.isnan(to), np.inf, to), t1) - np.maximum(ti, t0)).sum() / w
                if _close(g, o_unclipped):
                    hits.append("time_average_wip() lets a lot that is not in the window at all count a negative number of hours. "
                                "The hours inside the window are never below 0: clip each lot's overlap at 0.")
                if _close(g, float(np.sum((ti <= t0) & (np.where(np.isnan(to), np.inf, to) > t0)))):
                    hits.append("time_average_wip() counts the lots in the line at t_start only. Average over the whole window, "
                                "lot-hours divided by t_end - t_start.")
            if len(hits) == 1:
                return False, hits[0]
            return False, (f"time_average_wip({tin}, {tout}, {t0}, {t1}) returned {g:.6g}; it should be {want:.6g}: the hours every lot spends "
                           "inside the window, summed, divided by t_end - t_start.")
    return True, "time_average_wip() passed 14 tests, including lots that are still in the line, lots that cross the window edges, and lots outside the window."


def _ci_variants(x):
    n = len(x)
    m, s1, s0 = float(np.mean(x)), float(np.std(x, ddof=1)), float(np.std(x))
    t975, t95 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    return {"ddof0": t975 * s0 / math.sqrt(n), "z": 1.96 * s1 / math.sqrt(n), "no_sqrt": t975 * s1, "se_only": s1 / math.sqrt(n),
            "onesided": t95 * s1 / math.sqrt(n), "df_n": stats.t.ppf(0.975, n) * s1 / math.sqrt(n), "sd": s1}


def _test_ci95(ci95):
    cases = [[7.2, 8.1, 6.9, 7.7, 8.4], [10.0, 12.0], [3.1, 2.9, 3.6, 3.3, 2.7, 3.0, 3.4, 3.2], [7.56, 8.12, 6.94, 7.71, 7.30, 8.55, 6.88, 7.42, 7.95, 7.18, 8.30, 7.05, 7.66, 7.88, 6.60, 7.37, 7.75, 8.01, 7.24, 7.49],
             [-0.4, -0.1, -0.7, -0.3, -0.5, -0.2]]
    for vals in cases:
        for kind in ("array", "series", "list"):
            arg = list(vals) if kind == "list" else _wrap(vals, kind)
            ok, got = _call(ci95, "ci95", arg)
            if not ok:
                return False, got
            if not isinstance(got, (tuple, list, np.ndarray)) or len(got) != 2:
                return False, f"ci95() returned {got!r}: return two numbers, the mean and the half-width of the interval, as mean, half_width."
            gm, gh = _num(got[0]), _num(got[1])
            if gm is None or gh is None:
                return False, f"ci95() returned {got!r}: return two numbers, mean and half_width."
            wm, wh = reference_ci95(vals)
            if _close(gm, wm) and _close(gh, wh):
                continue
            if _close(gm, wm - wh) and _close(gh, wm + wh):
                return False, "ci95() returns the two ends of the interval. Return the mean and the HALF-width: mean, half_width."
            if not _close(gm, wm):
                return False, f"ci95() gave a mean of {gm:.6g}; the mean of the values is {wm:.6g}."
            if _close(gh, _ci_variants(vals)["ddof0"]):
                return False, ("ci95() divides by n in the standard deviation. For replications use the sample standard deviation, "
                               "with n - 1 in the divisor: np.std(values, ddof=1).")
            if _close(gh, _ci_variants(vals)["z"]):
                return False, ("ci95() uses 1.96. With 20 replications or fewer, use the t distribution with n - 1 degrees of freedom: "
                               "stats.t.ppf(0.975, n - 1).")
            if _close(gh, _ci_variants(vals)["no_sqrt"]):
                return False, "ci95() leaves out the division by the square root of n: the half-width is t * s / sqrt(n)."
            if _close(gh, _ci_variants(vals)["se_only"]):
                return False, "ci95() returns the standard error of the mean. The 95 % half-width is the standard error times t(0.975, n - 1)."
            if _close(gh, _ci_variants(vals)["sd"]):
                return False, "ci95() returns the standard deviation of the values. The half-width of the interval for the MEAN is t * s / sqrt(n)."
            if _close(gh, _ci_variants(vals)["onesided"]):
                return False, "ci95() uses the 0.95 quantile. A two-sided 95 % interval uses the 0.975 quantile: stats.t.ppf(0.975, n - 1)."
            if _close(gh, _ci_variants(vals)["df_n"]):
                return False, "ci95() uses n degrees of freedom. The t distribution for the mean of n values has n - 1."
            return False, f"ci95({vals}) returned a half-width of {gh:.6g}; it should be t(0.975, n - 1) * s / sqrt(n) = {wh:.6g}."
    return True, "ci95() passed 15 tests, with lists, arrays and pandas Series, two to 20 values."


def _test_calibrate(calibrate_now):
    cases = [([0.4, 0.9, 2.2, 2.5, 2.6], 2.0, 3, True), ([0.4, 2.1, 2.2, 1.9, 2.5, 2.6], 2.0, 3, False), ([2.3, 2.5], 2.0, 3, False),
             ([2.0, 2.0, 2.0], 2.0, 3, False), ([-2.2, -2.4, -2.6], 2.0, 3, True), ([0.1, 3.5, 3.6, 0.2], 2.0, 3, False),
             ([2.5, 2.6, 2.7, 0.3], 2.0, 3, False), ([0.2, 0.5, 2.4], 2.0, 1, True), ([], 2.0, 3, False), ([1.4, 1.6, 1.7, 1.8], 1.5, 3, True),
             ([0.5, 2.5, 2.6, 2.7, 2.8], 2.0, 4, True), ([2.5, 2.6, 2.7, 2.8, 0.5], 2.0, 4, False)]
    for vals, limit, k, want in cases:
        for kind in ("list", "array"):
            arg = list(vals) if kind == "list" else np.array(vals, dtype=float)
            ok, got = _call(calibrate_now, "calibrate_now", arg, limit, k, allow_none=True)
            if not ok:
                return False, got
            if got is None:
                return False, "calibrate_now() returned nothing (None): return True or False."
            if not isinstance(got, (bool, np.bool_)):
                return False, f"calibrate_now() returned {got!r}: return True or False."
            if bool(got) == want:
                continue
            v = list(vals)
            tail = v[-k:]
            if want and not got:
                if not all(x > limit for x in tail) and all(abs(x) > limit for x in tail):
                    return False, "calibrate_now() misses drift estimates below zero: compare abs(value) with the limit."
                return False, f"calibrate_now({v}, {limit}, {k}) returned False; it should return True: the last {k} values are all beyond {limit} in size."
            hits = []
            if len(v) < k:
                hits.append("calibrate_now() raises the alarm with fewer than k values. Fewer than k runs since the last calibration "
                            "cannot show k runs in a row beyond the limit: return False.")
            if any(abs(x) > limit for x in tail) and not all(abs(x) > limit for x in tail):
                hits.append("calibrate_now() raises the alarm when ANY of the last k values is beyond the limit. All k must be: use all(...), not any(...).")
            if all(abs(x) >= limit for x in tail) and not all(abs(x) > limit for x in tail):
                hits.append("calibrate_now() counts a value exactly at the limit. A value beyond the limit is abs(value) > limit.")
            if len(hits) == 1:
                return False, hits[0]
            return False, (f"calibrate_now({v}, {limit}, {k}) returned {got}; it should return {want}: True only when there are at least {k} values "
                           f"and the last {k} are all beyond {limit} in size.")
    return True, "calibrate_now() passed 24 tests, with lists and arrays, negative drift, values at the limit and too few values."


def check_dt_module6(throughput, mean_cycle_time, time_average_wip, ci95, calibrate_now, KEY):
    try:
        key = int(KEY)
    except (TypeError, ValueError):
        return False, ["Set KEY to the number Canvas gives you in Question 1 of the lab check, for example KEY = 417."], None
    if not 100 <= key <= 999:
        return False, [f"KEY should be a whole number from 100 to 999, the number in Question 1 of the lab check; it is {KEY}."], None
    results = [_test_throughput(throughput), _test_cycle(mean_cycle_time), _test_wip(time_average_wip), _test_ci95(ci95), _test_calibrate(calibrate_now)]
    msgs = [("PASSED  " if ok else "NOT PASSED  ") + m for ok, m in results]
    if not all(ok for ok, _ in results):
        return False, msgs, None
    return True, msgs, verification_code(key)
