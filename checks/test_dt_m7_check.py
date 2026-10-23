"""Tests of dt_m7_check.py: correct and wrong functions and data sheets for the three capstone tracks.

Run from the repository root: python3 checks/test_dt_m7_check.py
Every correct combination must pass and print the stored value; every wrong one must be refused (no value)."""
import copy
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "content" / "m7"))
import dt_m7_check as K  # noqa: E402
from oxide_tools import predict  # noqa: E402
from r2r_tools import dg_thickness  # noqa: E402
from vm_tools import FEATURES, physics_prediction  # noqa: E402

KEYS = (417, 263, 905)


def good_sheet(track, key):
    r = K.reference_results(track, key)
    ref = K.REF_CHOICES[track]
    tol = r["tolerance"]
    v = {t: K.reference_verdict(r[t][0], r[t][1], tol) for t in ("heldout", "challenging")}
    d = ref["do_not_use"]
    dn = [d["always"]] + ([d["challenging"]] if v["challenging"] != "meets" else []) + ([d["heldout"]] if v["heldout"] != "meets" else [])
    return {"intended_use": ref["intended_use"], "data_sources": ref["data_sources"], "heldout_mae": round(r["heldout"][0], 3),
            "heldout_verdict": v["heldout"], "challenging_mae": round(r["challenging"][0], 3), "challenging_verdict": v["challenging"],
            "baseline_mae": round(r["baseline"][0], 3), "do_not_use": dn, "risk_mitigation": ref["risk_mitigation"][0],
            "low_confidence_action": ref["low_confidence_action"], "note": ""}


# ---------------------------------------------------------------- verdict functions
def v_ok(m, h, t):
    return K.reference_verdict(m, h, t)


def v_ok_caps(m, h, t):
    return K.reference_verdict(m, h, t).capitalize()


GOOD_VERDICTS = [v_ok, v_ok_caps]


def v_mae_only(m, h, t):
    return "meets" if m <= t else "misses"


def v_strict(m, h, t):
    return "meets" if m + h < t else "misses" if m - h > t else "insufficient evidence"


def v_ge(m, h, t):
    return "meets" if m + h <= t else "misses" if m - h >= t else "insufficient evidence"


def v_short(m, h, t):
    r = K.reference_verdict(m, h, t)
    return "insufficient" if r == "insufficient evidence" else r


def v_swapped(m, h, t):
    return "misses" if m + h <= t else "meets" if m - h > t else "insufficient evidence"


def v_double(m, h, t):
    return K.reference_verdict(m, 2 * h, t)


def v_bool(m, h, t):
    return m + h <= t


BAD_VERDICTS = [v_mae_only, v_strict, v_ge, v_short, v_swapped, v_double, v_bool]


# ---------------------------------------------------------------- process connections
def p_pred_ok(t, fit, d):
    return float(predict(fit, t)["x"][0]) + d


def p_pred_dg(t, fit, d):
    return float(dg_thickness(t, fit["B"], fit["BA"], fit["xi"])) + d


def p_pred_arr(t, fit, d):
    return predict(fit, t)["x"] + d


def p_obs_ok(y, t, fit):
    return y - float(predict(fit, t)["x"][0])


def p_obs_arr(y, t, fit):
    return y - predict(fit, t)["x"]


GOOD_P = [(p_pred_ok, p_obs_ok), (p_pred_dg, p_obs_ok), (p_pred_arr, p_obs_arr)]


def p_pred_nod(t, fit, d):
    return float(predict(fit, t)["x"][0])


def p_pred_minus(t, fit, d):
    return float(predict(fit, t)["x"][0]) - d


def p_pred_tuple(t, fit, d):
    return float(predict(fit, t)["x"][0]) + d, 0.0


def p_obs_rev(y, t, fit):
    return float(predict(fit, t)["x"][0]) - y


def p_obs_zero_nan(y, t, fit):
    return 0.0 if math.isnan(y) else y - float(predict(fit, t)["x"][0])


def p_obs_fixed90(y, t, fit):
    return y - float(predict(fit, 90.0)["x"][0])


def p_obs_none(y, t, fit):
    return None if math.isnan(y) else y - float(predict(fit, t)["x"][0])


def p_raise(*a):
    raise NotImplementedError("Complete it first.")


BAD_P = [(p_pred_nod, p_obs_ok), (p_pred_minus, p_obs_ok), (p_pred_tuple, p_obs_ok), (p_pred_ok, p_obs_rev),
         (p_pred_ok, p_obs_zero_nan), (p_pred_ok, p_obs_fixed90), (p_pred_ok, p_obs_none), (p_raise, p_obs_ok), (p_pred_ok, p_raise)]


# ---------------------------------------------------------------- metrology connections
def m_vm_ok(model, table):
    return physics_prediction(table) + model.predict(table[FEATURES])


def m_vm_list(model, table):
    return list(physics_prediction(table) + model.predict(table[FEATURES]))


def m_obs_ok(y, v):
    return y - v


GOOD_M = [(m_vm_ok, m_obs_ok), (m_vm_list, m_obs_ok)]


def m_vm_phys(model, table):
    return physics_prediction(table)


def m_vm_corr(model, table):
    return model.predict(table[FEATURES])


def m_vm_minus(model, table):
    return physics_prediction(table) - model.predict(table[FEATURES])


def m_vm_allcols(model, table):
    return physics_prediction(table) + model.predict(table)


def m_vm_first(model, table):
    return float((physics_prediction(table) + model.predict(table[FEATURES]))[0])


def m_obs_rev(y, v):
    return v - y


BAD_M = [(m_vm_phys, m_obs_ok), (m_vm_corr, m_obs_ok), (m_vm_minus, m_obs_ok), (m_vm_allcols, m_obs_ok), (m_vm_first, m_obs_ok),
         (m_vm_ok, m_obs_rev), (p_raise, m_obs_ok)]


# ---------------------------------------------------------------- toolset connections
def t_ok(calib, hours):
    return K.reference_twin_estimates(calib, hours)


def t_ok_nodrop(calib, hours):
    f = calib["f_end"] - calib["f_start"]
    l = calib["l_end"] - calib["l_start"]
    e = calib["t_out"] - calib["e_start"]
    return {"arrival_rate": len(calib) / hours, "furnace_h": f.mean(), "litho_h": l.mean(), "etch_h": e.mean(),
            "furnace_cv": f.std() / f.mean(), "litho_cv": l.std() / l.mean(), "etch_cv": e.std() / e.mean()}


def t_ok_numpy(calib, hours):
    out = {"arrival_rate": len(calib) / hours}
    for s, (a, b) in {"furnace": ("f_end", "f_start"), "litho": ("l_end", "l_start"), "etch": ("t_out", "e_start")}.items():
        x = (calib[a] - calib[b]).to_numpy()
        x = x[~np.isnan(x)]
        out[s + "_h"] = np.mean(x)
        out[s + "_cv"] = np.std(x, ddof=1) / np.mean(x)
    return out


GOOD_T = [t_ok, t_ok_nodrop, t_ok_numpy]


def _mod(**changes):
    def f(calib, hours):
        e = K.reference_twin_estimates(calib, hours)
        for k, fn in changes.items():
            e[k] = fn(calib, hours, e)
        return e
    return f


def _x(calib, a, b):
    return (calib[a] - calib[b]).dropna()


t_ddof0 = _mod(litho_cv=lambda c, h, e: float(_x(c, "l_end", "l_start").std(ddof=0) / _x(c, "l_end", "l_start").mean()))
t_finished = _mod(arrival_rate=lambda c, h, e: c["t_out"].notna().sum() / h)
t_etch_from_litho = _mod(etch_h=lambda c, h, e: float(_x(c, "t_out", "l_end").mean()))
t_furnace_from_release = _mod(furnace_h=lambda c, h, e: float(_x(c, "f_end", "t_in").mean()))
t_litho_from_furnace = _mod(litho_h=lambda c, h, e: float(_x(c, "l_end", "f_end").mean()))
t_sd_not_cv = _mod(etch_cv=lambda c, h, e: float(_x(c, "t_out", "e_start").std()))
t_numpy_nan = _mod(furnace_h=lambda c, h, e: float(np.mean((c["f_end"] - c["f_start"]).to_numpy())))
t_hours_wrong = _mod(arrival_rate=lambda c, h, e: len(c) / float(c["t_in"].max()))


def t_missing(calib, hours):
    e = K.reference_twin_estimates(calib, hours)
    del e["etch_cv"]
    return e


BAD_T = [t_ddof0, t_finished, t_etch_from_litho, t_furnace_from_release, t_litho_from_furnace, t_sd_not_cv, t_numpy_nan,
         t_hours_wrong, t_missing, p_raise]


# ---------------------------------------------------------------- data sheets
def bad_sheets(track, key, other_key):
    g = good_sheet(track, key)
    d = K.REF_CHOICES[track]["do_not_use"]
    letters = list(K.REF_CHOICES[track]["intended_use"])
    out = []

    def mk(name, **changes):
        s = copy.deepcopy(g)
        s.update(changes)
        out.append((name, s))

    for f in K.C.FIELDS:
        mk(f"blank {f}", **{f: None})
    wrong = lambda field: next(x for x in "ABCD" if x not in (K.REF_CHOICES[track][field] if isinstance(K.REF_CHOICES[track][field], tuple) else (K.REF_CHOICES[track][field],)))
    for f in ("intended_use", "data_sources", "risk_mitigation", "low_confidence_action"):
        mk(f"wrong {f}", **{f: wrong(f)})
    mk("intended_use E", intended_use="E")
    for t in ("heldout", "challenging", "baseline"):
        mk(f"{t} +0.002", **{f"{t}_mae": round(g[f"{t}_mae"] + 0.002, 3)})
        mk(f"{t} as text", **{f"{t}_mae": "about 0.4"})
        for bad_value, label in [(float("nan"), "NaN"), ("nan", "text nan"), (float("inf"), "infinity"), (float("-inf"), "minus infinity"),
                                 (-g[f"{t}_mae"], "negative"), (True, "True"), ("", "empty text")]:
            mk(f"{t} {label}", **{f"{t}_mae": bad_value})
    mk("all three MAEs NaN", heldout_mae=float("nan"), challenging_mae=float("nan"), baseline_mae=float("nan"))
    mk("all three MAEs text nan", heldout_mae="nan", challenging_mae="nan", baseline_mae="nan")
    for t in ("heldout", "challenging"):
        other = next(v for v in K.C.VERDICTS if v != g[f"{t}_verdict"])
        mk(f"{t} verdict wrong", **{f"{t}_verdict": other})
        mk(f"{t} verdict misspelled", **{f"{t}_verdict": "pass"})
    dn = list(g["do_not_use"])
    mk("do_not_use without always", do_not_use=[x for x in dn if x != d["always"]] or ["Z"])
    if d["challenging"] in dn:
        mk("do_not_use without challenging", do_not_use=[x for x in dn if x != d["challenging"]])
    else:
        mk("do_not_use with challenging that met", do_not_use=dn + [d["challenging"]])
    if g["heldout_verdict"] == "meets":
        mk("do_not_use with heldout that met", do_not_use=dn + [d["heldout"]])
        mk("do_not_use with tested condition", do_not_use=dn + [d["tested"]])
    else:
        mk("do_not_use without heldout", do_not_use=[x for x in dn if x != d["heldout"]])
    mk("do_not_use letter Q", do_not_use=dn + ["Q"])
    mk("do_not_use as a number", do_not_use=3)
    o = good_sheet(track, other_key)
    mk("another learner's numbers", heldout_mae=o["heldout_mae"], challenging_mae=o["challenging_mae"], baseline_mae=o["baseline_mae"],
       heldout_verdict=o["heldout_verdict"], challenging_verdict=o["challenging_verdict"], do_not_use=o["do_not_use"])
    out.append(("not a dict", "C, A, 0.4"))
    return out


def run(track, conn, verdict, sheet, key):
    if track == "process":
        return K.check_process_twin(conn[0], conn[1], verdict, sheet, key)
    if track == "metrology":
        return K.check_metrology_twin(conn[0], conn[1], verdict, sheet, key)
    return K.check_toolset_twin(conn, verdict, sheet, key)


def main():
    good = {"process": GOOD_P, "metrology": GOOD_M, "toolset": GOOD_T}
    bad = {"process": BAD_P, "metrology": BAD_M, "toolset": BAD_T}
    n_good = n_bad = wrong = 0
    for track in ("process", "metrology", "toolset"):
        for i, key in enumerate(KEYS):
            other = KEYS[(i + 1) % len(KEYS)]
            sheet = good_sheet(track, key)
            want = K.personal_value(track, key)
            # correct functions and data sheet: must pass with the stored value
            for conn in good[track]:
                for v in GOOD_VERDICTS:
                    ok, msgs, val = run(track, conn, v, sheet, key)
                    n_good += 1
                    if not ok or val != want:
                        wrong += 1
                        print("WRONG (refused a correct one):", track, key, getattr(conn, "__name__", conn), v.__name__, msgs)
            # sheet with lowercase letters and a single-letter string is still correct
            s2 = copy.deepcopy(sheet)
            s2["intended_use"] = s2["intended_use"].lower()
            ok, msgs, val = run(track, good[track][0], v_ok, s2, key)
            n_good += 1
            if not ok:
                wrong += 1
                print("WRONG (lowercase refused):", track, key, msgs)
            # wrong functions
            for conn in bad[track]:
                ok, msgs, val = run(track, conn, v_ok, sheet, key)
                n_bad += 1
                if ok or val is not None:
                    wrong += 1
                    print("WRONG (accepted a wrong function):", track, key, getattr(conn, "__name__", conn))
                elif key == KEYS[0]:
                    print(f"  {track} {getattr(conn, '__name__', [c.__name__ for c in conn] if isinstance(conn, tuple) else conn)}:",
                          next(m for m in msgs if m.startswith("NOT"))[:150])
            for v in BAD_VERDICTS:
                ok, msgs, val = run(track, good[track][0], v, sheet, key)
                n_bad += 1
                if ok or val is not None:
                    wrong += 1
                    print("WRONG (accepted a wrong verdict):", track, key, v.__name__)
                elif key == KEYS[0] and track == "process":
                    print(f"  verdict {v.__name__}:", next(m for m in msgs if m.startswith("NOT"))[:150])
            # wrong data sheets, another learner's KEY, KEY not set
            for name, s in bad_sheets(track, key, other):
                ok, msgs, val = run(track, good[track][0], v_ok, s, key)
                n_bad += 1
                if ok or val is not None:
                    wrong += 1
                    print("WRONG (accepted a wrong sheet):", track, key, name)
                elif key == KEYS[0]:
                    print(f"  {track} sheet {name}:", next(m for m in msgs if m.startswith("NOT"))[:150])
            for k2 in (None, other, 99, 1000, "417"):
                ok, msgs, val = run(track, good[track][0], v_ok, sheet, k2)
                n_bad += 1
                if ok or val is not None:
                    wrong += 1
                    print("WRONG (accepted with KEY", k2, "):", track, key)
    print(f"{n_good} correct and {n_bad} wrong combinations: {wrong} wrong verdicts")
    return wrong


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
