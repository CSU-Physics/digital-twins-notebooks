"""Check cell logic for the Module 5 lab (Digital Twins for Semiconductor Manufacturing).

check_dt_module5(ewma_update, kalman_update, next_time, alarm_run, KEY) returns (passed, messages, verification_value).

The learner writes
  ewma_update(d_prev, y, lam)          the EWMA estimate of the offset after one more observation y:
                                       lam * y + (1 - lam) * d_prev; when y is NaN (no measurement), d_prev unchanged;
  kalman_update(d_prev, P_prev, y, Q, R)
                                       one step of the one-variable Kalman filter for a random-walk offset, returning
                                       (d, P): P_minus = P_prev + Q; K = P_minus / (P_minus + R); d = d_prev + K (y - d_prev);
                                       P = (1 - K) P_minus; when y is NaN, (d_prev, P_minus);
  next_time(d_hat, target_nm, B, BA, xi_nm, t_limits)
                                       the oxidation time (min) at which the twin's Deal-Grove thickness plus d_hat equals the
                                       target: with x = target_nm - d_hat and A = B / BA,
                                       t = (x^2 - xi^2 + A (x - xi)) / B, clipped to t_limits = (t_min, t_max);
  alarm_run(offsets, limit, k)         the run number (1 for the first value) at which abs(offset) has been greater than
                                       limit for k runs in a row; None if that never happens.

The verification value is the oxidation time, in minutes to 3 decimals, that the Kalman controller recommends for run 61
of the learner's furnace in the "drift" scenario, after the 60 production runs. It is computed with the course's
reference functions and printed only when every test passes.
"""

import math

import numpy as np

from furnace import R2R_RUNS, r2r_setup
from r2r_tools import Q_NM2, R_NM2, closed_loop


# ---------------------------------------------------------------- reference functions
def _is_missing(y):
    return y is None or (isinstance(y, (float, np.floating)) and math.isnan(y))


def reference_ewma(d_prev, y, lam):
    if _is_missing(y):
        return d_prev
    return lam * y + (1.0 - lam) * d_prev


def reference_kalman(d_prev, P_prev, y, Q, R):
    P_minus = P_prev + Q
    if _is_missing(y):
        return d_prev, P_minus
    K = P_minus / (P_minus + R)
    return d_prev + K * (y - d_prev), (1.0 - K) * P_minus


def reference_next_time(d_hat, target_nm, B, BA, xi_nm, t_limits):
    x = target_nm - d_hat
    A = B / BA
    t = (x * x - xi_nm ** 2 + A * (x - xi_nm)) / B
    return float(min(max(t, t_limits[0]), t_limits[1]))


def reference_alarm(offsets, limit, k):
    count = 0
    for i, v in enumerate(list(offsets)):
        count = count + 1 if abs(v) > limit else 0
        if count >= k:
            return i + 1
    return None


def personal_raw(key):
    log = closed_loop(key, "kalman", next_time=reference_next_time, kalman_update=reference_kalman, scenario="drift")
    s = r2r_setup(key)
    return reference_next_time(float(log["d_hat_nm"].iloc[-1]), s["target_nm"], s["B"], s["BA"], s["xi_nm"], s["t_limits"])


def personal_value(key):
    return round(personal_raw(key), 3)


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
    """A learner's number as a float, or None if it is not one number."""
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


# ---------------------------------------------------------------- tests
def _test_ewma(ewma_update):
    cases = [(1.0, 2.0, 0.3), (-0.8, -1.4, 0.3), (6.2, 5.1, 0.2), (5.0, 4.0, 1.0), (5.0, 4.0, 0.0), (2.5, 3.7, 0.5)]
    for d, y, lam in cases:
        ok, got = _call(ewma_update, "ewma_update", d, y, lam)
        if not ok:
            return False, got
        g = _num(got)
        if g is None:
            return False, (f"ewma_update({d}, {y}, {lam}) returned {got!r}: return one number, the new estimate.")
        want = reference_ewma(d, y, lam)
        if not _close(g, want):
            if _close(g, lam * d + (1 - lam) * y):
                return False, ("ewma_update() puts the weight lam on the old estimate. The new observation gets the weight lam: "
                               "d_new = lam * y + (1 - lam) * d_prev.")
            if _close(g, y):
                return False, "ewma_update() returns the new observation itself: blend it with the old estimate, lam * y + (1 - lam) * d_prev."
            return False, (f"ewma_update({d}, {y}, {lam}) returned {g:.6g}; it should be lam * y + (1 - lam) * d_prev = {want:.6g}.")
    for y_missing, label in ((float("nan"), "float('nan')"), (np.nan, "np.nan"), (np.float64("nan"), "np.float64('nan')")):
        for d, lam in ((2.5, 0.3), (-1.2, 1.0)):
            ok, got = _call(ewma_update, "ewma_update", d, y_missing, lam)
            if not ok:
                return False, got if "returned nothing" not in got else (
                    "ewma_update() returned nothing (None) for a run with no measurement: return d_prev unchanged.")
            g = _num(got)
            if g is None or not _close(g, d):
                return False, (f"For a run with no measurement (y = {label}), ewma_update() returned {got!r}. With no new "
                               "information the estimate stays where it was: return d_prev. Test with np.isnan(y).")
    return True, "ewma_update() passed 12 tests, including runs with no measurement."


def _test_kalman(kalman_update):
    cases = [(0.5, 0.16, 1.3, 0.01, 0.16), (-1.0, 0.04, -2.0, 0.02, 0.16), (6.0, 0.05, 5.2, 0.01, 0.16), (0.0, 1.0, 2.0, 0.0, 1.0)]
    for d, P, y, Q, R in cases:
        ok, got = _call(kalman_update, "kalman_update", d, P, y, Q, R)
        if not ok:
            return False, got
        if not isinstance(got, (tuple, list)) or len(got) != 2:
            return False, f"kalman_update() returned {got!r}: return two numbers, the new estimate and its variance, as d, P."
        gd, gP = _num(got[0]), _num(got[1])
        if gd is None or gP is None:
            return False, f"kalman_update() returned {got!r}: return two numbers, d and P."
        wd, wP = reference_kalman(d, P, y, Q, R)
        if _close(gd, wd) and _close(gP, wP):
            continue
        if _close(gd, wP) and _close(gP, wd):
            return False, "kalman_update() returns (P, d): return the estimate first, then its variance, as d, P."
        K0 = P / (P + R)
        if Q > 0 and _close(gd, d + K0 * (y - d)):
            return False, ("kalman_update() computes the gain from P_prev. First add the process noise, P_minus = P_prev + Q, "
                           "then K = P_minus / (P_minus + R).")
        if not _close(gd, wd):
            return False, (f"kalman_update({d}, {P}, {y}, {Q}, {R}) gives d = {gd:.6g}; it should be {wd:.6g}: "
                           "d = d_prev + K (y - d_prev) with K = P_minus / (P_minus + R) and P_minus = P_prev + Q.")
        if Q > 0 and _close(gP, (1 - (P + Q) / (P + Q + R)) * P):
            return False, "kalman_update() updates P from P_prev. Use P_minus: P = (1 - K) * P_minus, with P_minus = P_prev + Q."
        if _close(gP, (P + Q) * R / (P + Q + R) + Q):
            return False, "kalman_update() adds Q again after the update. Add Q once, before the gain: P = (1 - K) * P_minus."
        return False, (f"kalman_update({d}, {P}, {y}, {Q}, {R}) gives P = {gP:.6g}; it should be (1 - K) * P_minus = {wP:.6g}.")
    for d, P, Q, R in ((0.5, 0.05, 0.01, 0.16), (-2.0, 0.2, 0.03, 0.16)):
        for y_missing in (float("nan"), np.nan):
            ok, got = _call(kalman_update, "kalman_update", d, P, y_missing, Q, R)
            if not ok:
                return False, got
            if not isinstance(got, (tuple, list)) or len(got) != 2:
                return False, f"For a run with no measurement, kalman_update() returned {got!r}: still return d, P."
            gd, gP = _num(got[0]), _num(got[1])
            if gd is None or gP is None or (isinstance(gd, float) and math.isnan(gd)):
                return False, ("For a run with no measurement (y is NaN), kalman_update() returned a NaN estimate. With no new "
                               "information, keep d = d_prev, and let the uncertainty grow: P = P_prev + Q. Test with np.isnan(y).")
            if not _close(gd, d):
                return False, "For a run with no measurement, kalman_update() changes the estimate: keep d = d_prev."
            if _close(gP, P):
                return False, ("For a run with no measurement, kalman_update() keeps P unchanged. The offset can still move, so "
                               "its uncertainty grows by the process noise: P = P_prev + Q.")
            if not _close(gP, P + Q):
                return False, (f"For a run with no measurement, kalman_update() gives P = {gP:.6g}; it should be "
                               f"P_prev + Q = {P + Q:.6g}.")
    return True, "kalman_update() passed 8 tests, including runs with no measurement."


def _test_next_time(next_time):
    B, BA, xi = 176.75, 0.7307, 1.5
    cases = [(6.58, 61.2, B, BA, xi, (80.0, 100.0)), (4.10, 61.2, B, BA, xi, (80.0, 100.0)), (5.0, 62.0, 188.6, 0.796, xi, (80.0, 100.0)),
             (2.0, 60.0, 150.0, 0.6, 25.0, (60.0, 200.0)), (12.0, 61.2, B, BA, xi, (80.0, 100.0)), (-3.0, 61.2, B, BA, xi, (80.0, 100.0)),
             (12.0, 61.2, B, BA, xi, (70.0, 110.0)), (-3.0, 61.2, B, BA, xi, (70.0, 110.0))]
    for d, tgt, b, ba, x0, lim in cases:
        ok, got = _call(next_time, "next_time", d, tgt, b, ba, x0, lim)
        if not ok:
            return False, got
        g = _num(got)
        if g is None:
            return False, f"next_time() returned {got!r}: return one number, the oxidation time in minutes."
        want = reference_next_time(d, tgt, b, ba, x0, lim)
        if _close(g, want, 1e-7):
            continue
        A = b / ba
        free = lambda x, A=A: (x * x - x0 ** 2 + A * (x - x0)) / b
        raw = free(tgt - d)
        if _close(g, raw, 1e-7):
            return False, ("next_time() does not respect the time limits. Clip the time to t_limits: no shorter than t_limits[0] "
                           "and no longer than t_limits[1], for example min(max(t, t_min), t_max).")
        if _close(g, min(max(raw, 80.0), 100.0), 1e-7):
            return False, "next_time() always clips to 80 and 100 minutes: use the t_limits argument."
        clip = lambda t: min(max(t, lim[0]), lim[1])
        if _close(g, clip(free(tgt + d)), 1e-7):
            return False, ("next_time() aims at target + d_hat. The furnace grows the twin's prediction PLUS d, so the twin must "
                           "aim lower: x = target_nm - d_hat.")
        if _close(g, clip((tgt - d) ** 2 / b + (tgt - d) / ba), 1e-7) or _close(g, clip(((tgt - d) ** 2 + A * (tgt - d)) / b), 1e-7):
            return False, ("next_time() leaves out the initial oxide. The time from xi to x is t = (x^2 - xi^2 + A (x - xi)) / B, "
                           "with A = B / BA.")
        if _close(g, clip(((tgt - d) ** 2 - x0 ** 2 + (ba / b) * ((tgt - d) - x0)) / b), 1e-7):
            return False, "next_time() uses A = BA / B. The Deal-Grove A is B divided by B/A: A = B / BA."
        if _close(g * 60.0, want, 1e-6) or _close(g / 60.0, want, 1e-6):
            return False, "next_time() returns the time in other units. B is in nm^2/min, so t comes out in minutes: return it as it is."
        return False, (f"next_time({d}, {tgt}, {b}, {ba}, {x0}, {lim}) returned {g:.6g} min; it should be {want:.6g} min. "
                       "With x = target_nm - d_hat and A = B / BA: t = (x^2 - xi^2 + A (x - xi)) / B, then clip to t_limits.")
    return True, "next_time() passed 8 tests, including both time limits and a different initial oxide."


def _alarm_variant(vals, limit, k, ge=False, reset=True, use_abs=True):
    """The run a typical mistaken alarm_run() would return (None when it never fires)."""
    count = 0
    for i, v in enumerate(vals):
        a = abs(v) if use_abs else v
        if (a >= limit) if ge else (a > limit):
            count += 1
        elif reset:
            count = 0
        if count >= k:
            return i + 1
    return None


def _test_alarm(alarm_run):
    import pandas as pd
    cases = [([0.5, 2.5, 2.6, 2.7, 0.1], 2.0, 3), ([0.5, 2.1, 2.2, 1.9, 2.5, 2.6, 2.7, 0.1], 2.0, 3), ([-2.1, -2.3, -2.5, -1.0], 2.0, 3), ([2.0, 2.0, 2.0, 2.0, 2.0], 2.0, 3),
             ([0.0, 0.4, 3.1], 2.0, 1), ([0.1] * 10, 2.0, 3), ([2.5, 2.5, 1.0, 2.5, 2.5], 2.0, 3), ([0.3, -2.2, 2.4, -2.6, 0.0], 2.0, 3),
             ([1.2, 1.6, 1.7, 1.8, 0.2], 1.5, 3), ([0.0, 3.0, 3.0, 3.0, 3.0, 3.0], 2.0, 5)]
    for vals, limit, k in cases:
        for kind in ("list", "array", "series"):
            arg = list(vals) if kind == "list" else (np.array(vals) if kind == "array" else pd.Series(vals))
            ok, got = _call(alarm_run, "alarm_run", arg, limit, k, allow_none=True)
            if not ok:
                return False, got
            want = reference_alarm(vals, limit, k)
            g = None if got is None else _num(got)
            if got is not None and g is None:
                return False, f"alarm_run() returned {got!r}: return one whole number, the run of the alarm, or None."
            if g == want or (g is None and want is None):
                continue
            if g is not None and g == _alarm_variant(vals, limit, k, ge=True):
                return False, ("alarm_run() raises the alarm for values exactly at the limit. Count a run only when "
                               "abs(offset) is greater than the limit (>), not equal to it.")
            if g is not None and g == _alarm_variant(vals, limit, k, reset=False):
                return False, ("alarm_run() keeps counting across a run that is back inside the limit. The k runs must be "
                               "in a row: reset the count to 0 whenever abs(offset) is not above the limit.")
            if want is None:
                if g in (0, -1):
                    return False, (f"alarm_run() returns {got!r} when there is no alarm. Return None when no {k} runs in a row are "
                                   "beyond the limit, so that a run number is never mistaken for 'no alarm'.")
                return False, f"alarm_run({vals}, {limit}, {k}) returned {got!r}; it should return None: no {k} runs in a row are beyond the limit."
            if g is None:
                if _alarm_variant(vals, limit, k, use_abs=False) is None:
                    return False, "alarm_run() misses offsets beyond the limit on the negative side: compare abs(offset) with the limit."
                return False, f"alarm_run({vals}, {limit}, {k}) returned None; it should return run {want}."
            if g == want - 1:
                return False, ("alarm_run() returns a 0-based position. Runs are numbered from 1: the first value is run 1, "
                               "so return the position plus 1.")
            if g == want - k + 1 or g == want - k:
                return False, ("alarm_run() returns the first run of the streak. The alarm sounds on the run that completes the "
                               f"streak: the k-th run in a row beyond the limit (here run {want}).")
            return False, f"alarm_run({vals}, {limit}, {k}) returned {got!r}; it should return run {want}."
    return True, "alarm_run() passed 30 tests, with lists, arrays and pandas Series, values at the limit and broken streaks."


def check_dt_module5(ewma_update, kalman_update, next_time, alarm_run, KEY):
    try:
        key = int(KEY)
    except (TypeError, ValueError):
        return False, ["Set KEY to the number Canvas gives you in Question 1 of the lab check, for example KEY = 417."], None
    if not 100 <= key <= 999:
        return False, [f"KEY should be a whole number from 100 to 999, the number in Question 1 of the lab check; it is {KEY}."], None
    results = [_test_ewma(ewma_update), _test_kalman(kalman_update), _test_next_time(next_time), _test_alarm(alarm_run)]
    msgs = [("PASSED  " if ok else "NOT PASSED  ") + m for ok, m in results]
    if not all(ok for ok, _ in results):
        return False, msgs, None
    return True, msgs, personal_value(key)
