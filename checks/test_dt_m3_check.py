"""Tests of the Module 3 check: good implementations must pass, wrong ones must fail, for several KEYs.
Run from the repository root: python3 checks/test_dt_m3_check.py"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m3"), str(ROOT / "checks")]
from dt_m3_check import check_dt_module3, personal_value, verification_code  # noqa: E402
from furnace import OX_SIGMA_NM, OX_XI_NM, THIN_XI_NM, oxidation_runs  # noqa: E402
from oxide_tools import fit_deal_grove, linear_form_fit  # noqa: E402


# ---------------------------------------------------------------- dg_tau
def tau_ref(xi, B, BA):
    A = B / BA
    return (xi ** 2 + A * xi) / B

def tau_alt(xi, B, BA):
    return xi ** 2 / B + xi / BA

def tau_np(xi, B, BA):
    return float(np.divide(np.square(xi) + (B / BA) * xi, B))
TAU_GOOD = [tau_ref, tau_alt, tau_np]

def tau_inverted_A(xi, B, BA):
    A = BA / B
    return (xi ** 2 + A * xi) / B

def tau_no_linear(xi, B, BA):
    return xi ** 2 / B

def tau_times_B(xi, B, BA):
    return (xi ** 2 + B / BA * xi) * B

def tau_xi_not_squared(xi, B, BA):
    return (xi + B / BA * xi) / B

def tau_zero(xi, B, BA):
    return 0.0

def tau_raise(xi, B, BA):
    raise NotImplementedError

def tau_none(xi, B, BA):
    tau = (xi ** 2 + B / BA * xi) / B
TAU_BAD = [tau_inverted_A, tau_no_linear, tau_times_B, tau_xi_not_squared, tau_zero, tau_raise, tau_none]


# ---------------------------------------------------------------- dg_thickness
def th_ref(t, B, BA, xi):
    A = B / BA
    tau = tau_ref(xi, B, BA)
    return A / 2 * (np.sqrt(1 + 4 * B * (t + tau) / A ** 2) - 1)

def th_quadratic(t, B, BA, xi):
    A = B / BA
    c = -B * (np.asarray(t, dtype=float) + tau_ref(xi, B, BA))
    return (-A + np.sqrt(A ** 2 - 4 * c)) / 2

def th_alt(t, B, BA, xi):
    A = B / BA
    return 0.5 * A * (np.sqrt(1.0 + (t + (xi * xi + A * xi) / B) / (A * A / (4.0 * B))) - 1.0)
TH_GOOD = [th_ref, th_quadratic, th_alt]

def th_no_tau(t, B, BA, xi):
    A = B / BA
    return A / 2 * (np.sqrt(1 + 4 * B * t / A ** 2) - 1)

def th_parabolic(t, B, BA, xi):
    return np.sqrt(B * (t + tau_ref(xi, B, BA)))

def th_squared(t, B, BA, xi):
    return th_ref(t, B, BA, xi) ** 2

def th_math(t, B, BA, xi):
    A = B / BA
    return A / 2 * (math.sqrt(1 + 4 * B * (t + tau_ref(xi, B, BA)) / A ** 2) - 1)

def th_inverted_A(t, B, BA, xi):
    A = BA / B
    return A / 2 * (np.sqrt(1 + 4 * B * (t + tau_ref(xi, B, BA)) / A ** 2) - 1)

def th_linear(t, B, BA, xi):
    return xi + BA * t

def th_plus_xi(t, B, BA, xi):
    return xi + th_ref(t, B, BA, xi)

def th_raise(t, B, BA, xi):
    raise NotImplementedError
TH_BAD = [th_no_tau, th_parabolic, th_squared, th_math, th_inverted_A, th_linear, th_plus_xi, th_raise]


# ---------------------------------------------------------------- fit_mine
def fits(key):
    r = oxidation_runs(key)
    cal, thin, new = r["calibration"], r["thin"], r["new"]
    good = [fit_deal_grove(cal["time_min"], cal["thickness_nm"], OX_SIGMA_NM, OX_XI_NM),
            fit_deal_grove(cal["time_min"].to_numpy(), cal["thickness_nm"].to_numpy(), 0.5, 25.0, p0=(170.0, 0.75)),
            fit_deal_grove(list(cal["time_min"]), list(cal["thickness_nm"]), OX_SIGMA_NM, OX_XI_NM,
                           p0=linear_form_fit(cal["time_min"].iloc[[0, 5, 11]], cal["thickness_nm"].iloc[[0, 5, 11]], OX_XI_NM))]
    ex = oxidation_runs(2026)["calibration"]
    both = cal.iloc[:8]
    bad = [None, {"B": 1.0},
           fit_deal_grove(ex["time_min"], ex["thickness_nm"], OX_SIGMA_NM, OX_XI_NM),                     # the example furnace
           fit_deal_grove(cal["time_min"], cal["thickness_nm"], OX_SIGMA_NM, THIN_XI_NM),                 # wrong starting oxide
           fit_deal_grove(cal["time_min"], cal["thickness_nm"], OX_SIGMA_NM, 0.0),
           fit_deal_grove(cal["time_min"], cal["thickness_nm"], 1.0, OX_XI_NM),                           # wrong sigma
           fit_deal_grove(both["time_min"], both["thickness_nm"], OX_SIGMA_NM, OX_XI_NM),                 # part of the runs
           fit_deal_grove(thin["time_min"], thin["thickness_nm"], OX_SIGMA_NM, THIN_XI_NM)]              # the thin-oxide runs
    other = oxidation_runs(417 if key != 417 else 100)["calibration"]
    bad.append(fit_deal_grove(other["time_min"], other["thickness_nm"], OX_SIGMA_NM, OX_XI_NM))           # another learner's furnace
    # the right B and B/A, with one other field broken: NaN, infinity, a wrong covariance
    import copy
    base = good[0]
    for field, value in [("xi", float("nan")), ("sigma", float("nan")), ("sigma", float("inf")), ("se_B", float("nan")),
                         ("cov", np.full((2, 2), np.nan)), ("cov", np.zeros((3, 3))), ("cov", np.array([[1.0, 2.0], [0.0, 1.0]])),
                         ("cov", np.array([[1.0, 0.0], [0.0, -1.0]])), ("cov", base["cov"] * 4), ("B", float("nan")), ("BA", -base["BA"]),
                         ("n", float("nan")), ("se_BA", "nan")]:
        f = copy.deepcopy(base)
        f[field] = value
        bad.append(f)
    return good, bad


KEYS = [100, 417, 999]
bad = n_good = n_bad = 0
for key in KEYS:
    want = verification_code(key)
    good_fits, bad_fits = fits(key)
    for i in range(max(len(TAU_GOOD), len(TH_GOOD), len(good_fits))):
        ok, msgs, val = check_dt_module3(TAU_GOOD[i % 3], TH_GOOD[i % 3], good_fits[i % 3], key)
        n_good += 1
        if not ok or val != want:
            bad += 1
            print("WRONG VERDICT good", key, i, msgs)
    for w in TAU_BAD:
        ok, msgs, val = check_dt_module3(w, th_ref, good_fits[0], key); n_bad += 1
        if ok or val is not None:
            bad += 1; print("WRONG VERDICT bad tau", key, w.__name__, msgs)
        elif key == 417:
            print(f"{w.__name__:>20}: {[m for m in msgs if m.startswith('NOT')][0]}")
    for w in TH_BAD:
        ok, msgs, val = check_dt_module3(tau_ref, w, good_fits[0], key); n_bad += 1
        if ok or val is not None:
            bad += 1; print("WRONG VERDICT bad thickness", key, w.__name__, msgs)
        elif key == 417:
            print(f"{w.__name__:>20}: {[m for m in msgs if m.startswith('NOT')][0]}")
    for j, f in enumerate(bad_fits):
        ok, msgs, val = check_dt_module3(tau_ref, th_ref, f, key); n_bad += 1
        if ok or val is not None:
            bad += 1; print("WRONG VERDICT bad fit", key, j, msgs)
        elif key == 417:
            print(f"{'fit ' + str(j):>20}: {[m for m in msgs if m.startswith('NOT')][0]}")
for k in [None, "abc", 99, 1000]:
    ok, msgs, val = check_dt_module3(tau_ref, th_ref, None, k); n_bad += 1
    if ok:
        bad += 1; print("WRONG VERDICT key", k)
print(check_dt_module3(tau_ref, th_ref, fits(417)[0][0], 417))
print(f"{n_good} good and {n_bad} wrong combinations over KEYs {KEYS}: {bad} wrong verdicts")
assert bad == 0

# ---------------------------------------------------------------- the verification value is not the result the notebook prints earlier
from dt_m3_check import personal_value as _result, verification_code as _code  # noqa: E402
for _k in (417, 100):
    assert abs(_code(_k) - _result(_k)) > 0.0005, "the verification value must differ from the result printed before the check"
print("bypass test passed: the verification value differs from the printed result for KEYs", (417, 100))
