"""Check cell logic for the Module 7 capstone (Digital Twins for Semiconductor Manufacturing).

check_process_twin(predict_next, offset_observation, verdict, datasheet, KEY)
check_metrology_twin(vm_predict, offset_observation, verdict, datasheet, KEY)
check_toolset_twin(twin_estimates, verdict, datasheet, KEY)
each return (passed, messages, verification_value).

The check tests the learner's connections between the prepared components and the verdict rule on hidden cases, then
recomputes the track's validation results for the KEY with the course's reference functions and compares the data sheet
with them. The verification value is printed only when everything passes. It is a code made from the KEY and the three
validation numbers of the data sheet (held-out, challenging and baseline mean absolute error, in thousandths):
(7 H + 3 C + 11 B + KEY) mod 9973, divided by 1000, so a number from 0 to 9.972 with 3 decimals.

The verdict rule: "meets" when MAE + half-width <= tolerance; "misses" when MAE - half-width > tolerance; otherwise
"insufficient evidence".
"""

import math

import numpy as np
import pandas as pd

import capstone as C

NUM_TOL = 0.0006

# The data sheet answers. do_not_use: "always" must be listed; "challenging" and "heldout" must be listed exactly when
# that test's verdict is not "meets"; "tested" must not be listed when the held-out verdict is "meets".
REF_CHOICES = {
    "process": {"intended_use": "C", "data_sources": "A", "risk_mitigation": ("B", "D"), "low_confidence_action": "B",
                "do_not_use": {"always": "B", "challenging": "C", "heldout": "A", "tested": "D"}},
    "metrology": {"intended_use": "A", "data_sources": "D", "risk_mitigation": ("B", "C"), "low_confidence_action": "C",
                  "do_not_use": {"always": "C", "challenging": "B", "heldout": "D", "tested": "A"}},
    "toolset": {"intended_use": "B", "data_sources": "C", "risk_mitigation": ("A", "D"), "low_confidence_action": "D",
                "do_not_use": {"always": "A", "challenging": "D", "heldout": "C", "tested": "B"}},
}
UNITS = {"process": "nm", "metrology": "nm", "toolset": "h"}
NAMES = {"heldout": "held-out", "challenging": "challenging", "baseline": "baseline"}


# ---------------------------------------------------------------- reference functions
def reference_verdict(mae, half_width, tolerance):
    if mae + half_width <= tolerance:
        return "meets"
    if mae - half_width > tolerance:
        return "misses"
    return "insufficient evidence"


def reference_predict_next(t_min, fit, d_hat):
    from oxide_tools import predict
    return float(predict(fit, t_min)["x"][0]) + d_hat


def reference_offset_observation_process(measured_nm, t_min, fit):
    from oxide_tools import predict
    return measured_nm - float(predict(fit, t_min)["x"][0])


def reference_vm_predict(model, table):
    from vm_tools import FEATURES, physics_prediction
    return physics_prediction(table) + model.predict(table[FEATURES])


def reference_offset_observation_metrology(measured_nm, vm_nm):
    return measured_nm - vm_nm


def reference_twin_estimates(calib, hours):
    f = (calib["f_end"] - calib["f_start"]).dropna()
    l = (calib["l_end"] - calib["l_start"]).dropna()
    e = (calib["t_out"] - calib["e_start"]).dropna()
    return {"arrival_rate": len(calib) / hours, "furnace_h": float(f.mean()), "litho_h": float(l.mean()), "etch_h": float(e.mean()),
            "furnace_cv": float(f.std() / f.mean()), "litho_cv": float(l.std() / l.mean()), "etch_cv": float(e.std() / e.mean())}


_RESULTS = {}


def reference_results(track, key):
    """The track's validation results for a key: {"heldout": (mae, hw, n), "challenging": (...), "baseline": (...),
    "tolerance": t}, computed with the reference functions."""
    ck = (track, int(key))
    if ck in _RESULTS:
        return _RESULTS[ck]
    if track == "process":
        import process_twin as T
        d = T.run_twin(key, "drift", reference_predict_next, reference_offset_observation_process)
        f = T.run_twin(key, "failing", reference_predict_next, reference_offset_observation_process)
        res = {"heldout": C.mae_ci(T.runs_between(d, T.HELDOUT_RUNS)["error_nm"]),
               "challenging": C.mae_ci(T.runs_between(f, T.CHALLENGING_RUNS)["error_nm"]),
               "baseline": C.mae_ci(T.runs_between(d, T.HELDOUT_RUNS)["baseline_error_nm"]), "tolerance": T.TOLERANCE_NM}
    elif track == "metrology":
        import metrology_twin as T
        log = T.run_twin(key, reference_vm_predict, reference_offset_observation_metrology)
        res = {"heldout": C.mae_ci(T.per_run_mae(log, "error_nm", T.HELDOUT_RUNS)),
               "challenging": C.mae_ci(T.per_run_mae(log, "error_nm", T.CHALLENGING_RUNS)),
               "baseline": C.mae_ci(T.per_run_mae(log, "baseline_error_nm", T.HELDOUT_RUNS)), "tolerance": T.TOLERANCE_NM}
    elif track == "toolset":
        import toolset_twin as T
        log = T.fab_log(key)
        a, z = T.week_window(T.CAL_WEEKS)
        cfg = T.twin_config(reference_twin_estimates(log[(log["t_in"] >= a) & (log["t_in"] < z)], z - a))
        oh = T.weekly_cycle_time(log, T.HELDOUT_WEEKS)
        oc = T.weekly_cycle_time(log, T.CHALLENGING_WEEKS)
        base = float(T.weekly_cycle_time(log, T.CAL_WEEKS).mean())
        res = {"heldout": C.mae_ci(oh - T.replay(cfg, key, T.HELDOUT_WEEKS)),
               "challenging": C.mae_ci(oc - T.replay(cfg, key, T.CHALLENGING_WEEKS)),
               "baseline": C.mae_ci(oh - base), "tolerance": T.TOLERANCE_H}
    else:
        raise ValueError(f"track must be one of {C.TRACKS}")
    _RESULTS[ck] = res
    return res


def _thousandths(x):
    return int(round(float(x) * 1000.0))


def code_value(key, heldout_mae, challenging_mae, baseline_mae):
    """The verification code: (7 H + 3 C + 11 B + KEY) mod 9973, divided by 1000, with H, C and B the three mean absolute
    errors in thousandths (rounded)."""
    n = 7 * _thousandths(heldout_mae) + 3 * _thousandths(challenging_mae) + 11 * _thousandths(baseline_mae) + int(key)
    return (n % 9973) / 1000.0


def personal_value(track, key):
    r = reference_results(track, key)
    return code_value(key, r["heldout"][0], r["challenging"][0], r["baseline"][0])


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


# ---------------------------------------------------------------- tests of the verdict rule
def _test_verdict(verdict):
    cases = [(0.30, 0.10, 0.5, "meets"), (0.25, 0.25, 0.5, "meets"), (0.45, 0.10, 0.5, "insufficient evidence"),
             (0.75, 0.25, 0.5, "insufficient evidence"), (0.90, 0.20, 0.5, "misses"), (0.60, 0.05, 0.5, "misses"),
             (0.30, 0.60, 1.0, "meets"), (2.0, 1.5, 1.0, "insufficient evidence"), (5.2, 3.9, 1.0, "misses"),
             (0.98, 0.0, 1.0, "meets"), (1.02, 0.0, 1.0, "misses")]
    for m, h, tol, want in cases:
        ok, got = _call(verdict, "verdict", m, h, tol)
        if not ok:
            return False, got
        if not isinstance(got, str):
            return False, f"verdict() returned {got!r}: return one of the three strings " + ", ".join(repr(v) for v in C.VERDICTS) + "."
        g = got.strip().lower()
        if g == want:
            continue
        if g not in C.VERDICTS:
            return False, (f"verdict() returned {got!r}. Return exactly one of " + ", ".join(repr(v) for v in C.VERDICTS)
                           + ", spelled as here, so that the data sheet and the check can read it.")
        if want == "meets" and g == "insufficient evidence" and math.isclose(m + h, tol):
            return False, (f"verdict({m}, {h}, {tol}) returned {got!r}. MAE + half-width equals the tolerance exactly: the rule is "
                           "\"meets\" when MAE + half-width <= tolerance.")
        if want == "insufficient evidence" and g == "misses" and math.isclose(m - h, tol):
            return False, (f"verdict({m}, {h}, {tol}) returned {got!r}. MAE - half-width equals the tolerance exactly: \"misses\" needs "
                           "MAE - half-width > tolerance, so this case is \"insufficient evidence\".")
        if (want == "insufficient evidence" and g in ("meets", "misses") and ((m <= tol and g == "meets") or (m > tol and g == "misses"))):
            return False, (f"verdict({m}, {h}, {tol}) returned {got!r}, from the MAE alone. The interval matters: with MAE {m} and "
                           f"half-width {h}, the true error could lie anywhere from {m - h:g} to {m + h:g}, on both sides of the "
                           f"tolerance {tol}. That is \"insufficient evidence\".")
        return False, (f"verdict({m}, {h}, {tol}) returned {got!r}; it should return {want!r}. The rule: \"meets\" when MAE + half-width "
                       "<= tolerance, \"misses\" when MAE - half-width > tolerance, otherwise \"insufficient evidence\".")
    return True, "verdict() passed 11 tests, including both edges of the interval."


# ---------------------------------------------------------------- tests of the connections, by track
def _test_process(predict_next, offset_observation):
    import process_twin as T
    from oxide_tools import predict
    fits = [T.calibrate(k) for k in (113, 642)]
    cases = [(80.0, 0.0), (92.37, -1.234), (100.0, 2.5), (86.5, 0.731)]
    for fit in fits:
        for t, d in cases:
            ok, got = _call(predict_next, "predict_next", t, fit, d)
            if not ok:
                return False, got
            g = _num(got)
            if g is None:
                return False, f"predict_next() returned {got!r}: return one number, the predicted thickness in nm."
            base = float(predict(fit, t)["x"][0])
            want = base + d
            if _close(g, want):
                continue
            if d != 0 and _close(g, base):
                return False, ("predict_next() leaves out the offset estimate. The twin's prediction is the Deal-Grove thickness of the "
                               "calibrated model plus d_hat, the Module 5 estimate of how far this furnace now sits from the model.")
            if d != 0 and _close(g, base - d):
                return False, "predict_next() subtracts d_hat. The offset is measured minus model, so the prediction is the model plus d_hat."
            return False, (f"predict_next({t}, fit, {d}) returned {g:.6g} nm; it should be {want:.6g} nm: the calibrated Deal-Grove "
                           "thickness, predict(fit, t_min)[\"x\"][0], plus d_hat.")
    fit = fits[0]
    for y, t in [(61.84, 90.0), (59.20, 97.5), (63.01, 82.0), (float("nan"), 91.0), (np.nan, 88.0)]:
        ok, got = _call(offset_observation, "offset_observation", y, t, fit)
        if not ok:
            return False, got
        g = _num(got)
        if g is None:
            return False, f"offset_observation() returned {got!r}: return one number, in nm (NaN when there is no measurement)."
        base = float(predict(fit, t)["x"][0])
        want = y - base
        if _close(g, want):
            continue
        if math.isnan(want):
            return False, ("offset_observation() returned a number for a run with no measurement. measured_nm is NaN then, and so "
                           "is measured_nm minus anything: return that NaN, and the Kalman update keeps its estimate.")
        if _close(g, -want):
            return False, "offset_observation() has the sign reversed. The offset is measured minus the model's prediction."
        return False, (f"offset_observation({y}, {t}, fit) returned {g:.6g} nm; it should be {want:.6g} nm: the measured thickness "
                       "minus the calibrated Deal-Grove thickness for that oxidation time (without d_hat).")
    return True, "predict_next() and offset_observation() passed 13 tests, including runs with no measurement."


def _test_metrology(vm_predict, offset_observation):
    import metrology_twin as T
    from vm_tools import FEATURES, physics_prediction
    data = T.vm_production(311)
    model = T.train_correction(data["train"])
    for k in (47, 73, 104):
        tab = pd.concat([data["production"], data["after_replacement"]], ignore_index=True)
        tab = tab[tab["run_index"] == k].reset_index(drop=True)
        ok, got = _call(vm_predict, "vm_predict", model, tab)
        if not ok:
            if "feature names" in got.lower() or "features" in got.lower():
                return False, got + " Pass only the model's inputs: model.predict(table[FEATURES])."
            return False, got
        try:
            g = np.asarray(got, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return False, f"vm_predict() returned {type(got).__name__}: return one predicted thickness per wafer (an array or Series)."
        phys = physics_prediction(tab)
        corr = model.predict(tab[FEATURES])
        want = phys + corr
        if g.size != want.size:
            return False, f"vm_predict() returned {g.size} values for a run of {want.size} wafers: return one prediction per wafer."
        if np.allclose(g, want, rtol=1e-9, atol=1e-9):
            continue
        if np.allclose(g, phys, rtol=1e-9, atol=1e-9):
            return False, ("vm_predict() returns the Deal-Grove physics prediction alone. The hybrid model of Module 4 adds the learned "
                           "correction: physics_prediction(table) + model.predict(table[FEATURES]).")
        if np.allclose(g, corr, rtol=1e-9, atol=1e-9):
            return False, ("vm_predict() returns the learned correction alone. The model was fitted to the residual, measured minus "
                           "physics, so the prediction is physics_prediction(table) + model.predict(table[FEATURES]).")
        if np.allclose(g, phys - corr, rtol=1e-9, atol=1e-9):
            return False, "vm_predict() subtracts the correction. The correction is measured minus physics: add it to the physics prediction."
        return False, ("vm_predict() does not give the hybrid prediction physics_prediction(table) + model.predict(table[FEATURES]) "
                       f"for a hidden run (first wafer: {g[0]:.6g} nm instead of {want[0]:.6g} nm).")
    for y, v in [(55.31, 54.92), (56.04, 56.71), (54.20, 54.20), (57.33, 55.86)]:
        ok, got = _call(offset_observation, "offset_observation", y, v)
        if not ok:
            return False, got
        g = _num(got)
        if g is None:
            return False, f"offset_observation() returned {got!r}: return one number, in nm."
        want = y - v
        if _close(g, want):
            continue
        if _close(g, -want):
            return False, "offset_observation() has the sign reversed. The offset is measured minus the virtual-metrology prediction."
        return False, f"offset_observation({y}, {v}) returned {g:.6g} nm; it should be {want:.6g} nm: measured minus predicted."
    return True, "vm_predict() and offset_observation() passed 7 tests on hidden runs."


def _small_logs():
    nan = float("nan")
    a = pd.DataFrame({"lot": [1, 2, 3, 4, 5, 6],
                      "t_in": [0.5, 2.0, 3.1, 5.6, 7.2, 9.4],
                      "f_start": [0.5, 2.0, 3.1, 5.6, 7.9, 9.6],
                      "f_end": [2.3, 3.7, 5.0, 7.3, 9.6, nan],
                      "l_start": [2.3, 3.7, 5.2, 7.3, 9.6, nan],
                      "l_end": [3.4, 4.6, 6.1, 8.5, nan, nan],
                      "e_start": [3.4, 4.6, 6.1, 8.5, nan, nan],
                      "t_out": [4.7, 5.9, 7.2, nan, nan, nan]})
    b = a.copy()
    b["t_in"] = b["t_in"] * 2.0
    for c in ("f_start", "f_end", "l_start", "l_end", "e_start", "t_out"):
        b[c] = b[c] * 2.0 + 0.3
    return [(a, 10.0), (b, 20.0)]


def _test_toolset(twin_estimates):
    import toolset_twin as T
    log = T.fab_log(731)
    a, z = T.week_window(T.CAL_WEEKS)
    real = log[(log["t_in"] >= a) & (log["t_in"] < z)].reset_index(drop=True)
    names = ("arrival_rate", "furnace_h", "litho_h", "etch_h", "furnace_cv", "litho_cv", "etch_cv")
    for calib, hours in _small_logs() + [(real, z - a)]:
        ok, got = _call(twin_estimates, "twin_estimates", calib.copy(), hours)
        if not ok:
            return False, got
        if not isinstance(got, dict):
            return False, f"twin_estimates() returned {type(got).__name__}: return a dict with the seven estimates, " + ", ".join(names) + "."
        missing = [k for k in names if k not in got]
        if missing:
            return False, f"twin_estimates() returned a dict without {missing}. It needs " + ", ".join(names) + "."
        want = reference_twin_estimates(calib, hours)
        for k in names:
            g = _num(got[k])
            if g is None:
                return False, f"twin_estimates()[{k!r}] is {got[k]!r}: each estimate is one number."
            if _close(g, want[k]):
                continue
            if math.isnan(g):
                return False, (f"twin_estimates()[{k!r}] is NaN. Lots still in the line have NaN times: drop them (.dropna()) before "
                               "taking the mean and the standard deviation of a process time.")
            if k == "arrival_rate":
                n_done = calib["t_out"].notna().sum()
                if _close(g, n_done / hours):
                    return False, ("twin_estimates() counts only the lots that left the line. The release rate counts every lot "
                                   "released in the calibration window, finished or not: len(calib) / hours.")
                return False, f"twin_estimates()['arrival_rate'] is {g:.6g}; it should be {want[k]:.6g} lots per hour: len(calib) / hours."
            stage = {"furnace": ("f_end", "f_start"), "litho": ("l_end", "l_start"), "etch": ("t_out", "e_start")}[k.split("_")[0]]
            x = (calib[stage[0]] - calib[stage[1]]).dropna()
            if k.endswith("_cv"):
                if _close(g, float(x.std(ddof=0) / x.mean())):
                    return False, (f"twin_estimates()[{k!r}] uses the standard deviation with n in the divisor. Use the sample standard "
                                   "deviation, n - 1 in the divisor, as pandas' .std() does (with NumPy: np.std(x, ddof=1)).")
                if _close(g, float(x.std())):
                    return False, f"twin_estimates()[{k!r}] is the standard deviation. The coefficient of variation is the standard deviation divided by the mean."
                return False, f"twin_estimates()[{k!r}] is {g:.6g}; it should be {want[k]:.6g}: the standard deviation of the process time divided by its mean."
            if k == "etch_h" and _close(g, float((calib["t_out"] - calib["l_end"]).dropna().mean())):
                return False, ("twin_estimates()['etch_h'] measures from the end of lithography, which includes the wait for an etch "
                               "chamber. The etch process time runs from e_start to t_out.")
            if k == "furnace_h" and _close(g, float((calib["f_end"] - calib["t_in"]).dropna().mean())):
                return False, ("twin_estimates()['furnace_h'] measures from the release, which includes the wait for a tube. The furnace "
                               "process time runs from f_start to f_end.")
            if k == "litho_h" and _close(g, float((calib["l_end"] - calib["f_end"]).dropna().mean())):
                return False, ("twin_estimates()['litho_h'] measures from the end of the furnace, which includes the wait for the "
                               "lithography tool. Its process time runs from l_start to l_end.")
            return False, f"twin_estimates()[{k!r}] is {g:.6g}; it should be {want[k]:.6g} hours: the mean of {stage[0]} - {stage[1]} over the lots that have both."
    return True, "twin_estimates() passed 3 hidden logs, including lots still in the line."


# ---------------------------------------------------------------- the data sheet
def _letter(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    return s or None


def _check_datasheet(track, sheet, res):
    """A list of (ok, message) for the data sheet, against the reference results res."""
    out = []
    if not isinstance(sheet, dict):
        return [(False, "datasheet is not a dict. Start from datasheet = blank_datasheet() and fill in its fields.")]
    blank = [f for f in C.FIELDS if sheet.get(f) in (None, "", [], ())]
    if blank:
        return [(False, "The data sheet has blank fields: " + ", ".join(blank) + ". Fill every field (only \"note\" may stay empty).")]
    ref = REF_CHOICES[track]
    tol = res["tolerance"]
    unit = UNITS[track]
    verd = {t: reference_verdict(res[t][0], res[t][1], tol) for t in ("heldout", "challenging")}
    problems = []
    # 3. the numbers and verdicts
    for t in ("heldout", "challenging", "baseline"):
        g = _num(sheet.get(f"{t}_mae"))
        if g is None or isinstance(sheet.get(f"{t}_mae"), bool) or not math.isfinite(g) or g < 0:
            problems.append(f"{t}_mae is {sheet.get(f'{t}_mae')!r}: an MAE is a finite number of zero or more. "
                            "Type the number your notebook printed, for example 0.412.")
        elif abs(g - res[t][0]) > NUM_TOL:
            problems.append(f"{t}_mae is {g:g} {unit}, but your {NAMES[t]} test gives {res[t][0]:.3f} {unit} for KEY {res['key']}. "
                            "Run the step for your KEY again and copy its number (3 decimals).")
    for t in ("heldout", "challenging"):
        v = str(sheet.get(f"{t}_verdict")).strip().lower()
        if v not in C.VERDICTS:
            problems.append(f"{t}_verdict is {sheet.get(f'{t}_verdict')!r}: write one of " + ", ".join(repr(x) for x in C.VERDICTS) + ".")
        elif v != verd[t]:
            m, h = res[t][0], res[t][1]
            problems.append(f"{t}_verdict {v!r} does not match your result: MAE {m:.3f} +/- {h:.3f} {unit} against a tolerance of "
                            f"{tol:g} {unit}. Apply your verdict() rule to these numbers.")
    # 1, 2, 5, 6: single choices
    single = {"intended_use": "the intended use must describe what you built and validated: one asset, its recipe or line, and the "
                              "use your tests support, no wider",
              "data_sources": "this field must name the data the twin actually uses and how often they update it",
              "risk_mitigation": "the risk must be one that this twin's data path has, and the mitigation must reduce that risk",
              "low_confidence_action": "when confidence is low, the action must stop relying on the twin's predictions, get "
                                       "measurements instead, and bring the twin back only after it meets the tolerance again"}
    for f, why in single.items():
        g = _letter(sheet.get(f))
        if g not in C.OPTIONS[track][f]:
            problems.append(f"{f} is {sheet.get(f)!r}: choose one letter from show_options({track!r}).")
            continue
        okset = ref[f] if isinstance(ref[f], tuple) else (ref[f],)
        if g not in okset:
            problems.append(f"{f}: option {g} does not fit. Read the options again: {why}.")
    # 4. conditions where the predictions should not be used
    raw = sheet.get("do_not_use")
    items = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, (list, tuple, set)) else None
    if items is None:
        problems.append("do_not_use must be a list of letters, for example [\"A\", \"C\"].")
    else:
        got = {_letter(x) for x in items}
        bad = [x for x in got if x not in C.OPTIONS[track]["do_not_use"]]
        if bad:
            problems.append(f"do_not_use contains {bad}: use letters from show_options({track!r}).")
        else:
            d = ref["do_not_use"]
            if d["always"] not in got:
                problems.append("do_not_use misses a condition outside the range the twin was calibrated and tested in. The data sheet "
                                "claims only what the tests support, so every untested condition outside that range is listed.")
            for t in ("challenging", "heldout"):
                need = verd[t] != "meets"
                if need and d[t] not in got:
                    problems.append(f"do_not_use misses the condition of your {NAMES[t]} test, whose verdict was {verd[t]!r}. Every condition "
                                    "whose test did not meet the tolerance must be listed.")
                if not need and d[t] in got:
                    problems.append(f"do_not_use lists the condition of your {NAMES[t]} test, but that test met the tolerance. List only "
                                    "conditions your results do not support.")
            if verd["heldout"] == "meets" and d["tested"] in got:
                problems.append("do_not_use lists a condition that your held-out test included and that met the tolerance. List only "
                                "conditions your results do not support.")
    if problems:
        return [(False, "Data sheet: " + p) for p in problems]
    return [(True, "The data sheet is complete, and its numbers, verdicts and choices match your results.")]


def _check(track, connection_test, verdict, datasheet, KEY):
    results = [connection_test(), _test_verdict(verdict)]
    if not all(ok for ok, _ in results):
        msgs = [("PASSED  " if ok else "NOT PASSED  ") + m for ok, m in results]
        msgs.append("Fix the functions first; the data sheet is checked once they pass.")
        return False, msgs, None
    try:
        key = int(KEY)
    except (TypeError, ValueError):
        key = None
    if key is None or not 100 <= key <= 999 or (KEY != key):
        results.append((False, "KEY is not set: type the KEY from Question 1 of your project quiz (a whole number from 100 to 999)."))
        return False, [("PASSED  " if ok else "NOT PASSED  ") + m for ok, m in results], None
    res = dict(reference_results(track, key))
    res["key"] = key
    results += _check_datasheet(track, datasheet, res)
    msgs = [("PASSED  " if ok else "NOT PASSED  ") + m for ok, m in results]
    if all(ok for ok, _ in results):
        return True, msgs, personal_value(track, key)
    return False, msgs, None


def check_process_twin(predict_next, offset_observation, verdict, datasheet, KEY):
    return _check("process", lambda: _test_process(predict_next, offset_observation), verdict, datasheet, KEY)


def check_metrology_twin(vm_predict, offset_observation, verdict, datasheet, KEY):
    return _check("metrology", lambda: _test_metrology(vm_predict, offset_observation), verdict, datasheet, KEY)


def check_toolset_twin(twin_estimates, verdict, datasheet, KEY):
    return _check("toolset", lambda: _test_toolset(twin_estimates), verdict, datasheet, KEY)
