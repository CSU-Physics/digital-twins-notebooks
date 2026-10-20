"""Tests of the Module 0 check: good implementations must pass, wrong ones must fail, for several KEYs.
Run from the repository root: python3 checks/test_dt_m0_check.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m0"), str(ROOT / "checks")]
import pandas as pd  # noqa: E402
from dt_m0_check import check_dt_module0, personal_value  # noqa: E402

# ---------------------------------------------------------------- correct versions
def ms_mask(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return v.mean(), v.std(ddof=1)

def ms_nan(values):
    return np.nanmean(values), np.nanstd(values, ddof=1)

def ms_pandas(values):
    s = pd.Series(values, dtype=float).dropna()
    return s.mean(), s.std()

def ms_loop(values):
    v = [float(x) for x in values if not np.isnan(x)]
    m = sum(v) / len(v)
    return m, (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5

def ma_fixed(x, window):
    return np.convolve(np.asarray(x, dtype=float), np.ones(window) / window, mode="valid")

def ma_fixed2(x, window):
    return np.convolve(np.asarray(x, dtype=float), np.ones(window), mode="valid") / window

def ma_loop(x, window):
    x = list(x)
    return np.array([sum(x[i:i + window]) / window for i in range(len(x) - window + 1)])

def ma_cumsum(x, window):
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    return (c[window:] - c[:-window]) / window

GOOD_MS = [ms_mask, ms_nan, ms_pandas, ms_loop]
GOOD_MA = [ma_fixed, ma_fixed2, ma_loop, ma_cumsum]

# ---------------------------------------------------------------- wrong versions
def ms_not_written(values):
    raise NotImplementedError("Complete mean_and_sd() first.")

def ms_population(values):
    return np.nanmean(values), np.nanstd(values)

def ms_keeps_nan(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), v.std(ddof=1)

def ms_nan_as_zero(values):
    v = np.nan_to_num(np.asarray(values, dtype=float))
    return v.mean(), v.std(ddof=1)

def ms_swapped(values):
    m, s = ms_mask(values)
    return s, m

def ms_one_value(values):
    return np.nanmean(values)

def ms_median(values):
    return np.nanmedian(values), np.nanstd(values, ddof=1)

def ms_variance(values):
    return np.nanmean(values), np.nanvar(values, ddof=1)

def ma_bug(x, window):                       # the planted bug
    return np.convolve(np.asarray(x, dtype=float), np.ones(window), mode="valid")

def ma_same(x, window):
    return np.convolve(np.asarray(x, dtype=float), np.ones(window) / window, mode="same")

def ma_rolling(x, window):
    return pd.Series(x).rolling(window).mean().to_numpy()

def ma_off_by_one(x, window):
    return np.convolve(np.asarray(x, dtype=float), np.ones(window) / (window + 1), mode="valid")

def ma_full(x, window):
    return np.convolve(np.asarray(x, dtype=float), np.ones(window) / window, mode="full")

WRONG_MS = [ms_not_written, ms_population, ms_keeps_nan, ms_nan_as_zero, ms_swapped, ms_one_value, ms_median, ms_variance]
WRONG_MA = [ma_bug, ma_same, ma_rolling, ma_off_by_one, ma_full]

KEYS = [100, 417, 999]
bad = 0
for key in KEYS:
    want = personal_value(key)
    for ms in GOOD_MS:
        for ma in GOOD_MA:
            ok, msgs, val = check_dt_module0(ms, ma, key)
            if not ok or val != want:
                bad += 1
                print("WRONG VERDICT good", key, ms.__name__, ma.__name__, msgs)
    for ms in WRONG_MS:
        ok, msgs, val = check_dt_module0(ms, ma_fixed, key)
        if ok or val is not None:
            bad += 1
            print("WRONG VERDICT bad", key, ms.__name__, msgs)
    for ma in WRONG_MA:
        ok, msgs, val = check_dt_module0(ms_mask, ma, key)
        if ok or val is not None:
            bad += 1
            print("WRONG VERDICT bad", key, ma.__name__, msgs)
for k in [None, "abc", 99, 1000, 12.7]:
    ok, msgs, val = check_dt_module0(ms_mask, ma_fixed, k)
    if ok and not (isinstance(k, float) and 100 <= int(k) <= 999):
        bad += 1
        print("WRONG VERDICT key", k, msgs)

# messages a learner sees for the planted bug and the population SD
print(check_dt_module0(ms_mask, ma_bug, 417)[1][1])
print(check_dt_module0(ms_population, ma_fixed, 417)[1][0])
print(check_dt_module0(ms_keeps_nan, ma_fixed, 417)[1][0])
print(check_dt_module0(ms_mask, ma_rolling, 417)[1][1])
n_good = len(KEYS) * len(GOOD_MS) * len(GOOD_MA)
n_bad = len(KEYS) * (len(WRONG_MS) + len(WRONG_MA))
print(f"{n_good} good and {n_bad} wrong combinations over KEYs {KEYS}: {bad} wrong verdicts")
assert bad == 0
