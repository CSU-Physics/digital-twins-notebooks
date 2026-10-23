"""Check cell logic for the Module 1 lab (Digital Twins for Semiconductor Manufacturing).

check_dt_module1(update_prediction, tau_hat, KEY) returns (passed, messages, verification_value).

The learner writes update_prediction(T_pred, reading, gain): move the twin's predicted temperature a fraction
`gain` of the way toward a new reading, and leave it unchanged when the reading is missing (NaN). The learner also
calls the supplied fit_tau() on the readings of their own stream, temperature_stream(KEY), from before 600 s, and
passes the fitted time constant as tau_hat.

The verification value is the root-mean-square one-step prediction error, in degrees C to 3 decimals, of the twin
that is synchronized with every reading (gain 0.5) over the readings from 600 s on, for the learner's own stream.
It is computed with the course's reference functions and printed only when every test passes.
"""
import math

import numpy as np

from furnace import LOAD_C, SETPOINT_C, STREAM_FIT_END_S, temperature_stream
from twin_tools import fit_tau, rms, run_twin

GAIN = 0.5


def reference_update(T_pred, reading, gain):
    if reading is None or (isinstance(reading, float) and math.isnan(reading)) or np.isnan(reading):
        return float(T_pred)
    return float(T_pred + gain * (reading - T_pred))


def reference_tau(key):
    s = temperature_stream(key)
    w = s[s["time_s"] < STREAM_FIT_END_S]
    return fit_tau(w["time_s"], w["zone_temp_C"], T0=LOAD_C, T_set=SETPOINT_C)[0]


def personal_value(key):
    s = temperature_stream(key)
    twin = run_twin(s, reference_tau(key), GAIN, reference_update)
    return round(rms(twin.loc[twin["time_s"] >= STREAM_FIT_END_S, "error"]), 3)


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def _test_update(update_prediction):
    rng = np.random.default_rng(606)
    cases = [(900.0, 910.0, 0.5), (900.0, 910.0, 0.0), (900.0, 910.0, 1.0), (1000.0, 990.0, 0.25),
             (950.0, float("nan"), 0.5), (980.0, 982.5, 0.8), (999.0, float("nan"), 1.0)]
    for _ in range(8):
        cases.append((float(rng.uniform(800, 1000)), float(rng.uniform(800, 1000)), float(rng.uniform(0.05, 0.95))))
    for T, y, g in cases:
        want = reference_update(T, y, g)
        try:
            got = update_prediction(T, y, g)
        except NotImplementedError:
            return False, "update_prediction() is not written yet: replace the raise line with your code."
        except Exception as e:
            return False, f"update_prediction() stopped with {type(e).__name__}: {e}"
        try:
            got = float(got)
        except Exception:
            return False, "update_prediction() must return one number, the corrected temperature, for example: return T_pred + ..."
        if math.isnan(y):
            if math.isnan(got):
                return False, ("update_prediction() returned nan for a missing reading. When the reading is NaN there is nothing "
                               "to learn from it: return T_pred unchanged (test it with np.isnan(reading)).")
            if not _close(got, want):
                return False, f"For a missing reading, update_prediction({T}, nan, {g}) should return T_pred = {T}, not {got:.6g}."
            continue
        if not _close(got, want):
            if _close(got, T - g * (y - T)):
                return False, "update_prediction() moves the prediction away from the reading: the correction is gain * (reading - T_pred)."
            if _close(got, y + g * (T - y)) and g != 0.5:
                return False, ("update_prediction() gives the reading a weight of 1 - gain. The gain is the weight of the new reading: "
                               "gain 0 keeps T_pred, gain 1 jumps to the reading.")
            return False, f"update_prediction({T:.4g}, {y:.4g}, {g:.3g}) should return {want:.6g}, not {got:.6g}."
    return True, f"update_prediction() passed {len(cases)} tests, including missing readings and gains of 0 and 1."


def _test_tau(tau_hat, key):
    if tau_hat is None:
        return False, "tau_mine is still None: in Step 7, fit tau to the readings of your own stream from before 600 s, as in Step 3."
    try:
        tau_hat = float(tau_hat)
    except Exception:
        return False, "tau_mine should be one number, the fitted time constant in seconds (the first value fit_tau returns)."
    if not math.isfinite(tau_hat) or tau_hat <= 0:
        return False, (f"tau_mine is {tau_hat}, but a time constant must be a finite, positive number of seconds. "
                       "Fit tau to the readings of your stream from before 600 s with fit_tau(), as in Step 3, and use the first value it returns.")
    want = reference_tau(key)
    if abs(tau_hat - want) > 0.05:
        return False, (f"tau_mine is {tau_hat:.2f} s, but the fit of the readings of your stream from before 600 s gives a different value. "
                       "Check that you fitted temperature_stream(KEY), only the rows with time_s < 600, with T0 = 800 and T_set = 1000.")
    return True, f"tau_mine = {tau_hat:.2f} s matches the fit of your stream's first 10 minutes."


def check_dt_module1(update_prediction, tau_hat, KEY):
    try:
        key = int(KEY)
    except (TypeError, ValueError):
        return False, ["Set KEY to the number Canvas gives you in Question 1 of the lab check, for example KEY = 417."], None
    if not 100 <= key <= 999:
        return False, [f"KEY should be a whole number from 100 to 999, the number in Question 1 of the lab check; it is {KEY}."], None
    ok1, m1 = _test_update(update_prediction)
    ok2, m2 = _test_tau(tau_hat, key)
    msgs = [("PASSED  " if ok1 else "NOT PASSED  ") + m1, ("PASSED  " if ok2 else "NOT PASSED  ") + m2]
    if not (ok1 and ok2):
        return False, msgs, None
    return True, msgs, personal_value(key)
