"""Check cell logic for the Module 3 lab (Digital Twins for Semiconductor Manufacturing).

check_dt_module3(dg_tau, dg_thickness, fit_mine, KEY) returns (passed, messages, verification_value).

The learner writes
  dg_tau(xi, B, BA)            tau = (xi^2 + A xi) / B with A = B / BA: the time the model needs to grow the
                               starting oxide xi (minutes);
  dg_thickness(t, B, BA, xi)   the Deal-Grove thickness x(t) = (A/2) (sqrt(1 + 4 B (t + tau) / A^2) - 1) (nm), for a
                               number or an array of times t (minutes);
and calls the supplied fit_deal_grove() on the calibration runs of their own furnace, oxidation_runs(KEY), with the
stated starting oxide (25.0 nm) and measurement standard deviation (0.5 nm), and passes the result as fit_mine.

The learner's result is the thickness the learner's calibrated model predicts for the new run of 270 minutes,
in nm to 3 decimals, computed with the course's reference functions and printed only when every test passes.

The verification value that the check prints is not the result itself, which the notebook shows earlier as a learning
output. It is a code made from the result and the KEY: (A * round(1000 * result) + B * KEY + C) mod 9973, divided by
1000, to 3 decimals, with A = 11, B = 26, C = 307 for this module. It is printed only when every test passes, so the
result alone does not answer Question 1. It shows that the check passed for this KEY; it is not proof of authorship.
"""

import math

import numpy as np

from furnace import OX_NEW_MIN, OX_SIGMA_NM, OX_XI_NM, THIN_XI_NM, oxidation_runs
from oxide_tools import fit_deal_grove, predict


# ---------------------------------------------------------------- reference functions
def reference_tau(xi, B, BA):
    A = B / BA
    return (xi ** 2 + A * xi) / B


def reference_thickness(t, B, BA, xi):
    A = B / BA
    return A / 2.0 * (np.sqrt(1.0 + 4.0 * B * (np.asarray(t, dtype=float) + reference_tau(xi, B, BA)) / A ** 2) - 1.0)


def reference_fit(key):
    cal = oxidation_runs(key)["calibration"]
    return fit_deal_grove(cal["time_min"], cal["thickness_nm"], OX_SIGMA_NM, OX_XI_NM)


def personal_raw(key):
    return float(predict(reference_fit(key), OX_NEW_MIN)["x"][0])


def personal_value(key):
    return round(personal_raw(key), 3)


CODE_A, CODE_B, CODE_C = 11, 26, 307


def verification_code(key):
    """The verification value: a code made from the learner's result and the KEY, printed only when every test passes."""
    n = int(round(personal_value(key) * 1000))
    return ((CODE_A * n + CODE_B * int(key) + CODE_C) % 9973) / 1000


# ---------------------------------------------------------------- helpers
def _call(fn, name, *args):
    try:
        got = fn(*args)
    except NotImplementedError:
        return False, f"{name}() is not written yet: replace the raise line with your code."
    except Exception as e:
        return False, f"{name}() stopped with {type(e).__name__}: {e}"
    if got is None:
        return False, f"{name}() returned nothing (None): end it with a return line that gives back the result."
    return True, got


def _close(a, b, tol=1e-9):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= tol * np.maximum(1.0, np.abs(b))))


CASES = [(25.0, 173.98, 0.7482), (0.0, 173.98, 0.7482), (1.5, 160.0, 0.80), (20.0, 150.0, 0.6), (40.0, 420.0, 2.5), (8.0, 95.0, 0.31)]


# ---------------------------------------------------------------- tests
def _test_tau(dg_tau):
    for xi, B, BA in CASES:
        ok, got = _call(dg_tau, "dg_tau", xi, B, BA)
        if not ok:
            return False, got
        try:
            got = float(got)
        except Exception:
            return False, "dg_tau() should return one number, tau in minutes."
        want = reference_tau(xi, B, BA)
        if _close(got, want):
            continue
        A = B / BA
        if xi > 0 and _close(got, (xi ** 2 + (BA / B) * xi) / B):
            return False, "dg_tau() uses A = BA / B. The constant A is B divided by B/A: A = B / BA (in nm)."
        if xi > 0 and _close(got, xi ** 2 / B):
            return False, "dg_tau() leaves out the linear term: tau = (xi**2 + A * xi) / B."
        if xi > 0 and _close(got, (xi ** 2 + A * xi) * B):
            return False, "dg_tau() multiplies by B: divide by B, tau = (xi**2 + A * xi) / B."
        if xi > 0 and _close(got, (xi + A * xi) / B):
            return False, "dg_tau() uses xi instead of xi squared: tau = (xi**2 + A * xi) / B."
        return False, f"dg_tau({xi}, {B}, {BA}) should be {want:.6g} minutes, not {got:.6g}."
    return True, f"dg_tau() passed {len(CASES)} tests, including a wafer with no starting oxide."


def _test_thickness(dg_thickness):
    rng = np.random.default_rng(303)
    for xi, B, BA in CASES:
        t = np.array([0.0, 5.0, 30.0, 120.0, 480.0, 1440.0])
        ok, got = _call(dg_thickness, "dg_thickness", t.copy(), B, BA, xi)
        if not ok:
            if "scalar" in got or "math" in got:
                return False, got.rstrip(".") + ". Use NumPy functions such as np.sqrt (not math.sqrt), so that dg_thickness() works on an array of times."
            return False, got
        try:
            got = np.asarray(got, dtype=float)
        except Exception:
            return False, "dg_thickness() should return thicknesses as numbers (an array for an array of times)."
        if got.shape != t.shape:
            return False, ("dg_thickness() should work on an array of times and return one thickness per time. "
                           "Use NumPy functions, for example np.sqrt, instead of math.sqrt.")
        want = reference_thickness(t, B, BA, xi)
        if _close(got, want, 1e-8):
            continue
        A = B / BA
        if xi > 0 and _close(got, reference_thickness(t, B, BA, 0.0), 1e-8):
            return False, "dg_thickness() ignores the starting oxide: add tau = dg_tau(xi, B, BA) to t inside the square root."
        if _close(got, np.sqrt(B * (t + reference_tau(xi, B, BA))), 1e-8):
            return False, "dg_thickness() uses only the parabolic limit sqrt(B (t + tau)). Use the full Deal-Grove solution."
        if _close(got, reference_thickness(t, B, BA, xi) ** 2, 1e-8):
            return False, "dg_thickness() returns x squared: return x itself."
        if xi > 0 and _close(got, xi + reference_thickness(t, B, BA, xi), 1e-8):
            return False, "dg_thickness() adds xi a second time: the Deal-Grove solution with tau already includes the starting oxide."
        Ai = BA / B
        if _close(got, Ai / 2.0 * (np.sqrt(1.0 + 4.0 * B * (t + (xi ** 2 + Ai * xi) / B) / Ai ** 2) - 1.0), 1e-8):
            return False, "dg_thickness() uses A = BA / B. The constant A is B divided by B/A: A = B / BA (in nm)."
        if not _close(got[0], xi, 1e-6):
            return False, (f"dg_thickness() gives {got[0]:.4g} nm at t = 0, but the oxide starts at xi = {xi} nm. "
                           "Check tau and the formula x = (A/2) * (sqrt(1 + 4*B*(t + tau)/A**2) - 1).")
        return False, "dg_thickness() gives wrong thicknesses: x = (A/2) * (np.sqrt(1 + 4*B*(t + tau)/A**2) - 1), with A = B / BA."
    for _ in range(4):
        B, BA, xi, tt = float(rng.uniform(80, 400)), float(rng.uniform(0.3, 2.0)), float(rng.uniform(0, 40)), float(rng.uniform(0, 900))
        ok, got = _call(dg_thickness, "dg_thickness", tt, B, BA, xi)
        if not ok:
            return False, got
        try:
            got = float(np.asarray(got, dtype=float).reshape(-1)[0])
        except Exception:
            return False, "dg_thickness() should also accept a single time and return one number."
        if not _close(got, reference_thickness(tt, B, BA, xi), 1e-8):
            return False, f"dg_thickness({tt:.4g}, {B:.4g}, {BA:.4g}, {xi:.4g}) should be {float(reference_thickness(tt, B, BA, xi)):.6g} nm, not {got:.6g}."
    return True, "dg_thickness() passed its tests on arrays and single times, with and without a starting oxide."


def _finite(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _invalid_fit_fields(fit):
    """The name of the first field of a fit dict that is missing, not finite or not physically valid; None if all are valid."""
    for k in ("B", "BA", "xi", "sigma", "n", "se_B", "se_BA"):
        if k not in fit or not _finite(fit[k]):
            return f"value of {k}" if k in fit else f"dict: {k} is missing"
    if float(fit["B"]) <= 0 or float(fit["BA"]) <= 0:
        return "B or B/A (they must be above zero)"
    if float(fit["sigma"]) <= 0 or float(fit["xi"]) < 0 or float(fit["se_B"]) < 0 or float(fit["se_BA"]) < 0:
        return "sigma, xi or standard error (none can be negative, and sigma must be above zero)"
    try:
        cov = np.asarray(fit["cov"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return "covariance (cov)"
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return "covariance (cov): it must be a 2 x 2 matrix of finite numbers"
    if abs(cov[0, 1] - cov[1, 0]) > 1e-9 * max(1.0, abs(cov[0, 1])):
        return "covariance (cov): it must be symmetric"
    if np.min(np.linalg.eigvalsh(cov)) < -1e-12 * max(1.0, float(np.max(np.abs(cov)))):
        return "covariance (cov): it must be positive semidefinite"
    return None


def _test_fit(fit_mine, key):
    if fit_mine is None:
        return False, "fit_mine is still None: in Step 9, call fit_deal_grove() on your own calibration runs, as in Step 5."
    if not isinstance(fit_mine, dict) or "B" not in fit_mine or "BA" not in fit_mine:
        return False, "fit_mine should be the result of fit_deal_grove(...), a dict with B, BA and the other results."
    bad = _invalid_fit_fields(fit_mine)
    if bad:
        return False, (f"fit_mine has an invalid {bad}. Every number in it must be finite and physically valid (B, B/A and sigma above zero, "
                       "the covariance a symmetric 2 x 2 matrix with no negative variance). Use the dict that fit_deal_grove() returns, unchanged.")
    want = reference_fit(key)
    if abs(float(fit_mine.get("xi", -1)) - OX_XI_NM) > 1e-9:
        return False, (f"fit_mine was fitted with xi = {fit_mine.get('xi')} nm. Your calibration wafers start with OX_XI_NM = {OX_XI_NM} nm "
                       f"of oxide (the thin-oxide wafers start with {THIN_XI_NM} nm, but they are not calibration data).")
    if abs(float(fit_mine.get("sigma", -1)) - OX_SIGMA_NM) > 1e-9:
        return False, (f"fit_mine was fitted with sigma = {fit_mine.get('sigma')} nm. Use the stated measurement standard deviation, "
                       f"OX_SIGMA_NM = {OX_SIGMA_NM} nm: it sets the size of the uncertainties and the bands.")
    if int(fit_mine.get("n", 0)) != 12:
        return False, f"fit_mine was fitted to {fit_mine.get('n')} measurements; fit all 12 calibration runs of oxidation_runs(KEY)['calibration']."
    if not (_close(fit_mine["B"], want["B"], 1e-5) and _close(fit_mine["BA"], want["BA"], 1e-5)):
        ex = oxidation_runs(2026)["calibration"]
        f26 = fit_deal_grove(ex["time_min"], ex["thickness_nm"], OX_SIGMA_NM, OX_XI_NM)
        if _close(fit_mine["B"], f26["B"], 1e-5):
            return False, "fit_mine is the fit of the example furnace (key 2026). Fit the calibration runs of oxidation_runs(KEY), your own furnace."
        return False, ("fit_mine does not match the fit of your calibration runs. Use mine = oxidation_runs(KEY), then "
                       "fit_deal_grove(cal['time_min'], cal['thickness_nm'], OX_SIGMA_NM, OX_XI_NM) with cal = mine['calibration'].")
    if not np.allclose(np.asarray(fit_mine["cov"], dtype=float), want["cov"], rtol=1e-4, atol=0) or \
            not (_close(fit_mine["se_B"], want["se_B"], 1e-4) and _close(fit_mine["se_BA"], want["se_BA"], 1e-4)):
        return False, ("fit_mine has the right B and B/A but not the uncertainties of your fit. Use the dict that fit_deal_grove() returns, unchanged: "
                       "its covariance and standard errors set the bands.")
    return True, f"fit_mine matches the fit of your 12 calibration runs: B = {want['B']:.2f} nm^2/min, B/A = {want['BA']:.4f} nm/min."


def check_dt_module3(dg_tau, dg_thickness, fit_mine, KEY):
    try:
        key = int(KEY)
    except (TypeError, ValueError):
        return False, ["Set KEY to the number Canvas gives you in Question 1 of the lab check, for example KEY = 417."], None
    if not 100 <= key <= 999:
        return False, [f"KEY should be a whole number from 100 to 999, the number in Question 1 of the lab check; it is {KEY}."], None
    results = [_test_tau(dg_tau), _test_thickness(dg_thickness), _test_fit(fit_mine, key)]
    msgs = [("PASSED  " if ok else "NOT PASSED  ") + m for ok, m in results]
    if not all(ok for ok, _ in results):
        return False, msgs, None
    return True, msgs, verification_code(key)
