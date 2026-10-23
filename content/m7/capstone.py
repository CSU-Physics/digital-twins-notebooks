"""Shared helpers for the Module 7 capstone (Digital Twins for Semiconductor Manufacturing): validation and the data sheet.

Every data value of the capstone is SIMULATED with the course's models; nothing here comes from a real fab.

mae_ci(errors)          the mean absolute error of a set of errors and the half-width of its 95 % confidence interval
VERDICTS                the three validation verdicts: "meets", "misses" and "insufficient evidence"
blank_datasheet()       an empty twin data sheet (a dict) to fill in
show_options(track)     the fixed choices of the data sheet for one track ("process", "metrology" or "toolset")
show_datasheet(sheet, track, results, tolerance, key, version)   the filled data sheet, as text, with the evidence behind its verdicts
"""

import math

import numpy as np
from scipy import stats

__version__ = "1.1.0"

TRACKS = ("process", "metrology", "toolset")
VERDICTS = ("meets", "misses", "insufficient evidence")


def mae_ci(errors):
    """Mean absolute error of the errors and the half-width of its 95 % confidence interval.

    errors holds one error per independent test unit (a run, or a week); NaN values (units with no measurement) are left
    out. The half-width is t(0.975, n - 1) * s / sqrt(n), where s is the sample standard deviation of the absolute errors.
    Returns (mae, half_width, n)."""
    e = np.abs(np.asarray(errors, dtype=float))
    e = e[~np.isnan(e)]
    n = int(e.size)
    if n < 2:
        raise ValueError("mae_ci() needs at least two errors")
    return float(e.mean()), float(stats.t.ppf(0.975, n - 1) * e.std(ddof=1) / math.sqrt(n)), n


FIELDS = ("intended_use", "data_sources", "heldout_mae", "heldout_verdict", "challenging_mae", "challenging_verdict",
          "baseline_mae", "do_not_use", "risk_mitigation", "low_confidence_action")

TITLES = {
    "intended_use": "1. Intended use and modeled asset (one letter)",
    "data_sources": "2. Data sources and update frequency (one letter)",
    "do_not_use": "4. Conditions where the predictions should not be used (a list of letters)",
    "risk_mitigation": "5. One security or data-integrity risk with its mitigation (one letter)",
    "low_confidence_action": "6. Action when confidence is low (one letter)",
}

OPTIONS = {
    "process": {
        "intended_use": {
            "A": "Run-by-run prediction of the monitor-wafer thickness for any furnace in the fab and any oxidation recipe.",
            "B": "Automatic recipe control of the simulated furnace with no engineer in the loop, validated for production.",
            "C": "Advisory prediction of the monitor-wafer thickness of one simulated dry-oxidation furnace (recipe DRYOX-90 at "
                 "1000 C, 80 to 100 minutes), run by run, to support the engineer who approves each oxidation time.",
            "D": "Prediction of how long a lot waits in front of the furnace.",
        },
        "data_sources": {
            "A": "12 qualification runs (60 to 126 minutes) calibrate the Deal-Grove model once; one monitor-wafer thickness per "
                 "production run updates the offset estimate after every run.",
            "B": "The furnace's temperature trace, once per second, updates the model during each run.",
            "C": "12 qualification runs calibrate the model once; nothing updates it during production.",
            "D": "Every product wafer is measured after every run and the model is refitted to all of them.",
        },
        "do_not_use": {
            "A": "Normal production of DRYOX-90 within 80 to 100 minutes.",
            "B": "Oxidation times outside the qualified 60 to 126 minutes, other temperatures, wet oxidation or other starting oxides.",
            "C": "Runs after the thermocouple starts to fail (the offset estimate moving by more than about 0.1 nm per run).",
            "D": "The run that follows a run with no monitor-wafer measurement.",
        },
        "risk_mitigation": {
            "A": "Risk: an unauthorized recipe change is sent to the furnace through the twin's sync channel. "
                 "Mitigation: retrain the twin more often.",
            "B": "Risk: a monitor-wafer result is attributed to the wrong run (a run ID mix-up). Mitigation: match each result's run "
                 "ID and timestamp with the furnace log, and reject results that do not match or fall outside a plausible range.",
            "C": "Risk: a monitor-wafer result is attributed to the wrong run. Mitigation: widen the tolerance so that such errors "
                 "still pass.",
            "D": "Risk: an unauthorized recipe change is sent to the furnace through the twin's sync channel. Mitigation: keep "
                 "advisory mode, with authenticated and logged engineer approval of every change.",
        },
        "low_confidence_action": {
            "A": "Keep automatic mode and widen the tolerance until the errors pass.",
            "B": "Hold the last approved oxidation time, switch to advisory mode, measure the next runs' monitor wafers, and use the "
                 "twin's recommendation again only when its errors are back within tolerance.",
            "C": "Refit the twin on the challenging runs and report the new error as its validation result.",
            "D": "Use the baseline's prediction instead, without recording the change.",
        },
    },
    "metrology": {
        "intended_use": {
            "A": "Virtual metrology for one simulated furnace: predicting the oxide thickness of the unmeasured product wafers of "
                 "recipe DRYOX-90 (80 to 100 minutes, moisture 2 to 12 ppm), with one sampled wafer measured per run, to decide "
                 "which runs to send to measurement.",
            "B": "Replacing all physical thickness measurement on every furnace in the fab.",
            "C": "Predicting the thickness of any recipe, including new ones, from the furnace traces alone.",
            "D": "Controlling the furnace temperature during the run.",
        },
        "data_sources": {
            "A": "Only the Deal-Grove textbook constants; no measurements.",
            "B": "Five measured wafers of every production run, and the model is refitted after every run.",
            "C": "The 40 training runs only; nothing updates the model in production.",
            "D": "40 training runs with five measured wafers each (trace summaries and thickness) fit the hybrid model once; in "
                 "production, every run's trace summary and one sampled wafer (slot 65) update the offset estimate after every run.",
        },
        "do_not_use": {
            "A": "Wafers in slots other than slot 65, the sampled slot.",
            "B": "Runs after a thermocouple replacement, until the offset estimate has settled.",
            "C": "Recipes outside 80 to 100 minutes, moisture outside 2 to 12 ppm, or other furnaces.",
            "D": "Normal production of DRYOX-90 within the training range.",
        },
        "risk_mitigation": {
            "A": "Risk: trace summaries are tampered with or come from the wrong run. Mitigation: retrain the model every week.",
            "B": "Risk: trace summaries are tampered with or come from the wrong run. Mitigation: check the run ID, timestamp and "
                 "plausible range of every trace summary, and keep measuring one sampled wafer per run to catch a twin that has gone wrong.",
            "C": "Risk: a predicted thickness is recorded as if it were a measurement. Mitigation: label every value as predicted or "
                 "measured, and never release a lot on a predicted value outside tolerance.",
            "D": "Risk: a predicted thickness is recorded as if it were a measurement. Mitigation: round predictions to two decimals, "
                 "like measurements.",
        },
        "low_confidence_action": {
            "A": "Keep reporting predicted thickness and widen the tolerance.",
            "B": "Retrain the model on the challenging runs and report the new error as its validation result.",
            "C": "Measure every wafer of the affected runs, record them as measured, and use the virtual metrology again only after "
                 "its errors on the sampled wafers are back within tolerance.",
            "D": "Use the baseline (the mean of recent sampled wafers) for all wafers, without recording the change.",
        },
    },
    "toolset": {
        "intended_use": {
            "A": "The cycle time of each individual lot, for dispatching decisions.",
            "B": "The weekly mean cycle time of one simulated furnace, lithography and etch line at its usual release rate, with "
                 "predictive furnace maintenance, computed from a given week's lot releases, for capacity planning (validated "
                 "retrospectively, on past weeks with their actual releases).",
            "C": "The cycle time of a whole fab with re-entrant flow and many products.",
            "D": "The oxide thickness of the lots that leave the furnace.",
        },
        "data_sources": {
            "A": "The tools' nameplate process times; no fab data.",
            "B": "Six weeks of the lot log calibrate the twin once; nothing else.",
            "C": "Six weeks of the lot log (release, start and end times at each tool group) calibrate the release rate and the "
                 "process times; each week, that week's actual release times drive the twin (trace-driven).",
            "D": "The weekly mean cycle times of the log, copied into the twin.",
        },
        "do_not_use": {
            "A": "Release rates well above the calibrated rate, other tool counts, or other products.",
            "B": "Weeks in which furnace tubes are calibrated by the predictive maintenance rule.",
            "C": "Normal weeks at the usual release rate.",
            "D": "Weeks with unplanned lithography breakdowns.",
        },
        "risk_mitigation": {
            "A": "Risk: lots missing from the log or logged with wrong times (a gap in the MES records). Mitigation: check the log "
                 "for missing lot numbers and impossible times (an end before its start) before calibrating, and recalibrate after a fix.",
            "B": "Risk: lots missing from the log. Mitigation: run more replications.",
            "C": "Risk: the twin's result is used to promise delivery dates. Mitigation: report a narrower interval.",
            "D": "Risk: the twin's result is changed, or used without its interval. Mitigation: publish each result with its "
                 "interval and validation date, from a controlled copy of the twin whose changes are logged.",
        },
        "low_confidence_action": {
            "A": "Run more replications until the interval is narrow enough to claim the tolerance.",
            "B": "Report the twin's result as usual.",
            "C": "Recalibrate the twin on the breakdown weeks and call the result validated.",
            "D": "Plan with the measured cycle times of recent weeks, mark the twin's results as not valid for these weeks, and add "
                 "a breakdown model, validated on new weeks, before using the twin again.",
        },
    },
}


def blank_datasheet():
    """An empty twin data sheet. Fill every field: letters for the fixed choices, numbers for the errors, and
    one of VERDICTS for each verdict. "note" is optional and not checked."""
    sheet = {f: None for f in FIELDS}
    sheet["note"] = ""
    return sheet


def show_options(track):
    """Print the fixed choices of the data sheet for one track."""
    if track not in OPTIONS:
        raise ValueError(f"track must be one of {TRACKS}")
    for field, title in TITLES.items():
        if field == "do_not_use":
            print("3. Validation results: heldout_mae, challenging_mae, baseline_mae (numbers, from your results), and "
                  "heldout_verdict, challenging_verdict (each one of " + ", ".join(repr(v) for v in VERDICTS) + ")")
            print()
        print(title)
        for letter, text in OPTIONS[track][field].items():
            print(f"  {letter}  {text}")
        print()


UNITS = {"process": ("nm", "runs"), "metrology": ("nm", "runs"), "toolset": ("h", "weeks")}


def show_datasheet(sheet, track, results=None, tolerance=None, key=None, version=None):
    """Print a filled data sheet as text, with the chosen options written out.

    With results (the dict of (mae, half_width, n) for "heldout", "challenging" and "baseline" that Step 9 computes for
    your KEY) and the tolerance, it also prints the evidence behind the verdicts: each MAE with its approximate 95 %
    interval, the number of test units, the tolerance and the units, and the KEY, notebook version and date of the sheet."""
    import datetime

    def opt(field, letter):
        return OPTIONS[track][field].get(str(letter).strip().upper(), "(not one of the options)") if letter not in (None, "") else "(blank)"
    unit, units_of = UNITS.get(track, ("", "units"))
    print(f"TWIN DATA SHEET ({track} twin)")
    print(TITLES["intended_use"][3:].split(" (")[0] + ":", opt("intended_use", sheet.get("intended_use")))
    print(TITLES["data_sources"][3:].split(" (")[0] + ":", opt("data_sources", sheet.get("data_sources")))
    print(f"Validation results ({unit}): held-out MAE", sheet.get("heldout_mae"), "(" + str(sheet.get("heldout_verdict")) + "),",
          "challenging MAE", sheet.get("challenging_mae"), "(" + str(sheet.get("challenging_verdict")) + "),",
          "baseline MAE", sheet.get("baseline_mae"))
    if results:
        tol_text = f"{tolerance:g} {unit}" if tolerance is not None else "(not given)"
        print(f"  Evidence (tolerance {tol_text} on the MAE; the intervals are approximate 95 % intervals that treat the "
              f"test {units_of} as independent):")
        for name in ("heldout", "challenging", "baseline"):
            if name in results:
                m, hw, n = results[name]
                print(f"    {name:12s} MAE {m:.3f} {unit}, interval {m - hw:.3f} to {m + hw:.3f} {unit}, n = {n} {units_of}")
    print("Do not use for:")
    for letter in (sheet.get("do_not_use") or []):
        print("  -", opt("do_not_use", letter))
    print(TITLES["risk_mitigation"][3:].split(" (")[0] + ":", opt("risk_mitigation", sheet.get("risk_mitigation")))
    print(TITLES["low_confidence_action"][3:].split(" (")[0] + ":", opt("low_confidence_action", sheet.get("low_confidence_action")))
    if sheet.get("note"):
        print("Note:", sheet["note"])
    print(f"KEY {key if key is not None else '(not set)'} · notebook version {version or '(not given)'} · "
          f"sheet dated {datetime.date.today().isoformat()} (the day this cell ran)")
