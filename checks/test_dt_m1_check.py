"""Tests of the Module 1 check: good implementations must pass, wrong ones must fail, for several KEYs.
Run from the repository root: python3 checks/test_dt_m1_check.py"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m1"), str(ROOT / "checks")]
import pandas as pd  # noqa: E402
from dt_m1_check import check_dt_module1, personal_value, reference_tau  # noqa: E402
from furnace import temperature_stream  # noqa: E402
from twin_tools import fit_tau  # noqa: E402


# ---------------------------------------------------------------- correct versions
def up_isnan(T_pred, reading, gain):
    if np.isnan(reading):
        return T_pred
    return T_pred + gain * (reading - T_pred)

def up_math(T_pred, reading, gain):
    return T_pred if math.isnan(reading) else (1 - gain) * T_pred + gain * reading

def up_pandas(T_pred, reading, gain):
    if pd.isna(reading):
        return T_pred
    return T_pred + gain * (reading - T_pred)

def up_nan_to_num(T_pred, reading, gain):
    return float(np.where(np.isnan(reading), T_pred, T_pred + gain * (np.nan_to_num(reading) - T_pred)))

def up_selfcompare(T_pred, reading, gain):
    if reading != reading:          # NaN is the only value not equal to itself
        return T_pred
    return gain * reading + (1 - gain) * T_pred

GOOD = [up_isnan, up_math, up_pandas, up_nan_to_num, up_selfcompare]

# ---------------------------------------------------------------- wrong versions
def up_not_written(T_pred, reading, gain):
    raise NotImplementedError("Complete update_prediction() first.")

def up_no_nan(T_pred, reading, gain):
    return T_pred + gain * (reading - T_pred)

def up_sign(T_pred, reading, gain):
    return T_pred if np.isnan(reading) else T_pred - gain * (reading - T_pred)

def up_inverted(T_pred, reading, gain):
    return T_pred if np.isnan(reading) else reading + gain * (T_pred - reading)

def up_copy_reading(T_pred, reading, gain):
    return T_pred if np.isnan(reading) else reading

def up_ignore(T_pred, reading, gain):
    return T_pred

def up_nan_zero(T_pred, reading, gain):
    return T_pred + gain * (np.nan_to_num(reading) - T_pred)

def up_returns_none(T_pred, reading, gain):
    T_pred = T_pred + gain * (reading - T_pred)

def up_mean(T_pred, reading, gain):
    return T_pred if np.isnan(reading) else (T_pred + reading) / 2

WRONG = [up_not_written, up_no_nan, up_sign, up_inverted, up_copy_reading, up_ignore, up_nan_zero, up_returns_none, up_mean]


def tau_window(key, end=600, start=0):
    s = temperature_stream(key)
    w = s[(s["time_s"] < end) & (s["time_s"] >= start)]
    return fit_tau(w["time_s"], w["zone_temp_C"])[0]


KEYS = [100, 417, 999]
bad = 0
for key in KEYS:
    want = personal_value(key)
    good_taus = [reference_tau(key), tau_window(key)]
    for up in GOOD:
        for tau in good_taus:
            ok, msgs, val = check_dt_module1(up, tau, key)
            if not ok or val != want:
                bad += 1
                print("WRONG VERDICT good", key, up.__name__, msgs)
    for up in WRONG:
        ok, msgs, val = check_dt_module1(up, good_taus[0], key)
        if ok or val is not None:
            bad += 1
            print("WRONG VERDICT bad update", key, up.__name__, msgs)
    other = 417 if key != 417 else 100
    wrong_taus = [None, "abc", tau_window(key, end=1800), tau_window(key, end=1200), tau_window(other), 150.0]
    for tau in wrong_taus:
        ok, msgs, val = check_dt_module1(up_isnan, tau, key)
        if ok or val is not None:
            bad += 1
            print("WRONG VERDICT bad tau", key, tau, msgs)
for k in [None, "abc", 99, 1000]:
    ok, msgs, val = check_dt_module1(up_isnan, 150.0, k)
    if ok:
        bad += 1
        print("WRONG VERDICT key", k, msgs)

for up in [up_no_nan, up_sign, up_inverted, up_returns_none]:
    print(check_dt_module1(up, reference_tau(417), 417)[1][0])
print(check_dt_module1(up_isnan, tau_window(417, end=1800), 417)[1][1])
n_good = len(KEYS) * len(GOOD) * 2
n_bad = len(KEYS) * (len(WRONG) + 6)  # a fit of only the first 300 s gives nearly the same tau, so it is accepted
print(f"{n_good} good and {n_bad} wrong combinations over KEYs {KEYS}: {bad} wrong verdicts")
assert bad == 0
