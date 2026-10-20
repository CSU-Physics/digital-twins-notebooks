"""Check cell logic for the Module 0 lab (Digital Twins for Semiconductor Manufacturing).

check_dt_module0(mean_and_sd, moving_average, KEY) returns (passed, messages, verification_value).

The learner writes mean_and_sd(values): the mean and the sample standard deviation (ddof = 1) of a set of
readings, ignoring missing values (NaN). The learner also repairs moving_average(x, window), which
the notebook gives with a planted bug: it adds up each run of `window` readings instead of averaging them.

The verification value is the sample standard deviation, in degrees C to 3 decimals, of the 5-point
moving average of the hold-period readings (time_s >= 1200, missing readings dropped) of the learner's
own simulated furnace log, furnace_log(KEY). It is computed with the course's reference functions and
printed only when every test passes.
"""
import math

import numpy as np

from furnace import HOLD_START_S, furnace_log

WINDOW = 5


def reference_mean_and_sd(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()), float(v.std(ddof=1))


def reference_moving_average(x, window):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def hold_readings(log):
    """The hold-period readings of a log (time_s >= 1200), with missing readings dropped."""
    return log.loc[log["time_s"] >= HOLD_START_S, "zone_temp_C"].dropna().to_numpy()


def personal_value(key):
    smooth = reference_moving_average(hold_readings(furnace_log(key)), WINDOW)
    return round(float(np.std(smooth, ddof=1)), 3)


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def _test_mean_and_sd(mean_and_sd):
    rng = np.random.default_rng(404)
    cases = [np.array([1.0, 2.0, 3.0, 4.0]), [999.6, 1000.4, 1000.1], np.array([5.0, np.nan, 7.0, 9.0]),
             np.array([2.0, 2.0, 2.0]), np.array([np.nan, 10.0, 12.0, np.nan])]
    for k in range(8):
        n = int(rng.integers(3, 40))
        v = rng.normal(1000, 0.5, n)
        v[rng.choice(n, size=int(rng.integers(0, 3)), replace=False)] = np.nan
        cases.append(v)
    try:
        import pandas as pd
        cases.append(pd.Series([3.0, np.nan, 4.0, 8.0]))
    except ImportError:
        pass
    for c in cases:
        want_m, want_s = reference_mean_and_sd(c)
        try:
            got = mean_and_sd(c.copy() if hasattr(c, "copy") else list(c))
        except NotImplementedError:
            return False, "mean_and_sd() is not written yet: replace the raise line with your code."
        except Exception as e:
            return False, f"mean_and_sd() stopped with {type(e).__name__}: {e}"
        try:
            got_m, got_s = (float(got[0]), float(got[1]))
        except Exception:
            return False, "mean_and_sd() must return two numbers, the mean and the standard deviation, for example: return m, s"
        if math.isnan(got_m) or math.isnan(got_s):
            return False, ("mean_and_sd() returned nan for readings that include a missing value. Leave the missing values out "
                           "first, for example with values[~np.isnan(values)] or np.nanmean and np.nanstd.")
        if not _close(got_m, want_m):
            return False, f"The mean is not right: for {np.round(np.asarray(c, float), 3).tolist()} it should be {want_m:.6g}, not {got_m:.6g}."
        if not _close(got_s, want_s):
            v = np.asarray(c, float)
            v = v[~np.isnan(v)]
            if _close(got_s, float(v.std(ddof=0))):
                return False, ("The standard deviation is the population value (ddof = 0). Use the sample standard deviation, "
                               "which divides by n - 1: np.std(values, ddof=1) or np.nanstd(values, ddof=1).")
            return False, f"The standard deviation is not right: for {np.round(v, 3).tolist()} it should be {want_s:.6g}, not {got_s:.6g}."
    return True, f"mean_and_sd() passed {len(cases)} tests, including readings with missing values."


def _test_moving_average(moving_average):
    rng = np.random.default_rng(505)
    cases = [(np.arange(1.0, 7.0), 3), (np.array([10.0, 10.0, 10.0, 10.0]), 2), (np.array([1.0, 5.0, 2.0]), 1)]
    for k in range(10):
        n = int(rng.integers(5, 40))
        cases.append((rng.normal(1000, 0.5, n), int(rng.integers(1, 8))))
    for x, w in cases:
        want = reference_moving_average(x, w)
        try:
            got = np.asarray(moving_average(x.copy(), w), dtype=float)
        except Exception as e:
            return False, f"moving_average() stopped with {type(e).__name__}: {e}"
        if got.shape != want.shape:
            return False, (f"moving_average() should return {want.size} values for {x.size} readings and window {w} "
                           f"(only the full windows, mode='valid'); it returned {got.size}. If you used pandas rolling() or "
                           "mode='same', go back to the np.convolve line with mode='valid'.")
        if not np.allclose(got, want, rtol=1e-12, atol=1e-9):
            if w > 1 and np.allclose(got, want * w, rtol=1e-12, atol=1e-9):
                return False, ("moving_average() still adds up each run of readings instead of averaging them: its values are "
                               f"{w} times too large for window {w}. Divide by the number of readings in the window.")
            return False, f"moving_average({np.round(x[:6], 3).tolist()}..., {w}) should start {np.round(want[:3], 4).tolist()}, not {np.round(got[:3], 4).tolist()}."
    return True, f"moving_average() passed {len(cases)} tests with windows from 1 to 7."


def check_dt_module0(mean_and_sd, moving_average, KEY):
    msgs = []
    try:
        key = int(KEY)
    except (TypeError, ValueError):
        return False, ["Set KEY to the number Canvas gives you in Question 1 of the lab check, for example KEY = 417."], None
    if not 100 <= key <= 999:
        return False, [f"KEY should be a whole number from 100 to 999, the number in Question 1 of the lab check; it is {KEY}."], None
    ok1, m1 = _test_mean_and_sd(mean_and_sd)
    msgs.append(("PASSED  " if ok1 else "NOT PASSED  ") + m1)
    ok2, m2 = _test_moving_average(moving_average)
    msgs.append(("PASSED  " if ok2 else "NOT PASSED  ") + m2)
    if not (ok1 and ok2):
        return False, msgs, None
    return True, msgs, personal_value(key)
