"""Build the three Module 7 capstone notebooks (Digital Twins for Semiconductor Manufacturing) and their Colab copies.

content/m7 holds the prepared components of the three tracks (process_twin.py, metrology_twin.py, toolset_twin.py), the
shared capstone.py, the components they reuse from Modules 3 to 6 (furnace.py, oxide_tools.py, vm_tools.py, r2r_tools.py,
line.py 1.1.0 and the SimPy wheel) and dt_m7_check.py (copied from checks/). The check cell only imports the check file, so
the course's reference solutions are not on screen."""
import shutil
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "m7"
VERSION = "2026-10-24"
EXAMPLE_KEY = 2026
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
RAW = "https://raw.githubusercontent.com/CSU-Physics/digital-twins-notebooks/main/content/m7/"

shutil.copyfile(ROOT / "checks" / "dt_m7_check.py", OUT / "dt_m7_check.py")


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


# ---------------------------------------------------------------- cells shared by the three tracks
def verification_intro():
    return """**Verification and validation are different questions.** Verification asks whether the code computes what the model says. Validation asks whether the model's answers are close enough to the real system for the use you intend. (These are the definitions the National Academies use for digital twins, from the National Research Council's 2012 report.) Verification comes first: a validation result means little if the code does not do what you think it does. The tests below compare the connected twin with answers you can work out exactly."""


def verdict_cells(tol, unit, held_expr):
    return [
        md(f"""## Step 6: your task, `verdict()`

A test gives an error with an interval, not a single number, and the **tolerance** was set before the test, from the twin's intended use: here the mean absolute error must be at most **{tol:g} {unit}**. Compare the whole interval with the tolerance:

- **meets**: the whole interval lies at or below the tolerance, MAE + half-width ≤ tolerance;
- **misses**: the whole interval lies above the tolerance, MAE − half-width > tolerance;
- **insufficient evidence**: the interval contains the tolerance, so this test cannot tell. More test data would narrow the interval.

"Insufficient evidence" is an honest result, not a failure of the method. It is better than a "meets" that the data do not support.

**Your task:** write `verdict(mae, half_width, tolerance)`. Return one of the three strings exactly as written here: `"meets"`, `"misses"` or `"insufficient evidence"`."""),
        code('''def verdict(mae, half_width, tolerance):
    """"meets", "misses" or "insufficient evidence": the 95 % interval of the MAE, mae +/- half_width, against the tolerance."""
    # YOUR CODE HERE
    raise NotImplementedError("Complete verdict() first.")'''),
        code(f'''print(verdict(0.30, 0.10, 0.5), "  (should be meets: the interval runs from 0.20 to 0.40)")
print(verdict(0.45, 0.10, 0.5), "  (should be insufficient evidence: 0.35 to 0.55 contains 0.5)")
print(verdict(0.90, 0.20, 0.5), "  (should be misses: 0.70 to 1.10 lies above 0.5)")
print()
print("held-out test of the example:", verdict({held_expr}, TOLERANCE))'''),
    ]


def datasheet_cells(track, quiz_name):
    return [
        md(f"""## Step 10: the twin data sheet

A twin that others will use needs a label, as a dataset or a model does (datasheets for datasets, Gebru et al. 2021; model cards, Mitchell et al. 2019). The **twin data sheet** has six fields. Fields 1, 2, 5 and 6 are fixed choices, field 3 holds your numbers from Step 9, and field 4 is a list of choices. The cell below prints the choices for this track.

How to fill in field 4, the conditions where the predictions should **not** be used:

- list every condition **outside** the range the twin was calibrated and tested in: nothing supports a claim there;
- list each tested condition whose verdict was **not** "meets" (that is, "misses" or "insufficient evidence");
- do **not** list a condition that your tests included and that met the tolerance: your results support it.

Field 5 needs a risk that this twin's data path really has, and a mitigation that reduces that risk (NIST IR 8356 discusses the security and trust risks of digital twins). Field 6 is what the user should do when the twin's confidence is low.

The check cell in Step 11 reads this data sheet and compares it with your results: the numbers must match Step 9 (3 decimals), the verdicts must follow your `verdict()` rule, and the choices must be consistent with what you found."""),
        code(f'show_options("{track}")'),
        code(f'''datasheet = blank_datasheet()
datasheet["intended_use"] = None           # 1. one letter, for example "A"
datasheet["data_sources"] = None           # 2. one letter
datasheet["heldout_mae"] = None            # 3. your numbers from Step 9, 3 decimals
datasheet["heldout_verdict"] = None        #    "meets", "misses" or "insufficient evidence"
datasheet["challenging_mae"] = None
datasheet["challenging_verdict"] = None
datasheet["baseline_mae"] = None
datasheet["do_not_use"] = None             # 4. a list of letters, for example ["A", "D"]
datasheet["risk_mitigation"] = None        # 5. one letter
datasheet["low_confidence_action"] = None  # 6. one letter
datasheet["note"] = ""                     # optional, not checked

show_datasheet(datasheet, "{track}")'''),
        md(f"""## Step 11: check your work

The cell below tests your functions on cases you cannot see, recomputes the validation results for your KEY with the course's reference functions, and compares your data sheet with them. If everything passes, it prints your **verification value**, a code made from your KEY and the three numbers of your data sheet. Type it into Question 1 of the **{quiz_name}**.

If it says **NOT PASSED**, read the message: it tells you what to fix. Fix it, run the cells you changed, then run this cell again."""),
    ]


def closing_cell(extra):
    return md(f"""## What the data sheet says about the twin

You built a twin from parts that each worked on their own, verified that the connected code does what the model says, and validated it on data it had not seen, against a tolerance set before the test, next to a baseline and under one challenging condition. The data sheet records the result for whoever uses the twin next: what it is for, what it uses, how well it did, where it must not be used, one risk, and what to do when its confidence is low.

{extra}

**Validation is not done once.** The physical system keeps changing: thermocouples age, recipes change, tools are replaced. The National Academies' 2024 report on digital twins concludes that their verification, validation and uncertainty quantification must be a continual process, revisited as the physical system, the data and the use change. A data sheet carries a date for that reason.

**Claim only what the data support.** On simulated data, a result shows how well the twin recovers the course's simulator. It does not validate a twin of a real furnace or a real line.

To keep a copy of this notebook, choose **File > Download**.""")


def colab_copy(cells, name, files, extra_install=""):
    install = (extra_install + "\n") if extra_install else ""
    head = md(f"""## Colab only: get the course files

You are running the **Google Colab** copy of this notebook. Colab needs the course files next to the notebook: the prepared components of this track, the components they reuse from earlier modules, and `dt_m7_check.py`, the check code. Run the cell below once each time you open the notebook; it downloads the files from the course repository{" and installs SimPy" if extra_install else ""}. Everything else is the same as in the Canvas version, and your verification value is the same.""")
    dl = code(f"""{install}import urllib.request

BASE = "{RAW}"
for name in {files!r}:
    urllib.request.urlretrieve(BASE + name, name)
    print("downloaded", name)""")
    colab_cells = [cells[0], head, dl] + cells[1:]
    cnb = nbf.v4.new_notebook(cells=colab_cells, metadata={"colab": {"provenance": []},
                                                           "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                           "language_info": {"name": "python"}})
    return cnb


def write(cells, name, prefix, colab_files, extra_install=""):
    nb = nbf.v4.new_notebook(cells=cells, metadata=META)
    for i, c in enumerate(nb.cells):
        c["id"] = f"{prefix}-{i:02d}"
    text = nbf.writes(nb)
    assert "—" not in text and "–" not in text, "em or en dash in notebook text"
    (OUT / name).write_text(text)
    print("wrote", OUT / name, len(cells), "cells")
    cdir = ROOT / "colab" / "m7"
    cdir.mkdir(parents=True, exist_ok=True)
    cnb = colab_copy(cells, name, colab_files, extra_install)
    for i, c in enumerate(cnb.cells):
        c["id"] = f"{prefix}c-{i:02d}"
    ctext = nbf.writes(cnb)
    assert "—" not in ctext and "–" not in ctext
    cname = name.replace(".ipynb", "-colab.ipynb")
    (cdir / cname).write_text(ctext)
    print("wrote", cdir / cname, len(cnb.cells), "cells")


def process_cells():
    K = EXAMPLE_KEY
    quiz = "Module 7 project quiz for the process twin"
    cells = [
        md(f"""# Module 7 capstone, process twin: a calibrated furnace twin that stays in sync

**Digital Twins for Semiconductor Manufacturing · Module 7 · about 2 hours** · notebook version {VERSION}

This is the **process twin** track of the capstone. You join two prepared components into one twin of the course's furnace and test it the way a twin should be tested before anyone relies on it:

- the **Deal-Grove model** of Module 3, calibrated on qualification runs, predicts the oxide thickness for an oxidation time;
- the **offset estimate** of Module 5, a Kalman filter, keeps that prediction in sync as the thermocouple drifts.

Before each production run the twin predicts the monitor-wafer thickness and recommends the oxidation time (recipe DRYOX-90: dry oxygen, 1000 °C, 80 to 100 minutes). You write two short functions that connect the components, one function that turns a test result into a verdict, and the **twin data sheet**: what the twin is for, what it uses, how it performed, where it must not be used, one risk, and what to do when confidence is low.

The test plan is fixed before any test runs:

| test | data | pass when |
|---|---|---|
| held-out | production runs 1 to 60 of the drift scenario, each predicted before it is measured | mean absolute error (MAE) at most 0.5 nm |
| baseline | the calibrated Deal-Grove model alone, on the same runs | (for comparison) |
| challenging | runs 31 to 60 of the failing-thermocouple scenario | MAE at most 0.5 nm |

Steps 1 to 8 use the **course example furnace (KEY {K})**, so you can compare your numbers with the Module 7 page. Step 9 repeats the tests on your own furnace, with the KEY from Question 1 of the **{quiz}**. All data are **simulated**. Run each code cell with **Shift + Enter**, in order."""),
        md("""## Step 0: start Python

The first time, Python can take up to a minute to start. The cell prints the versions of the libraries and of the course files next to this notebook."""),
        code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import furnace, oxide_tools, r2r_tools, capstone, process_twin
from oxide_tools import predict
from capstone import mae_ci, blank_datasheet, show_options, show_datasheet
from process_twin import (qualification_runs, calibrate, kalman_update, next_time, run_twin, runs_between, plot_log,
                          TOLERANCE_NM, HELDOUT_RUNS, CHALLENGING_RUNS)

EXAMPLE_KEY = {K}
TOLERANCE = TOLERANCE_NM
print("Notebook version {VERSION}")
print("NumPy", np.__version__, "| pandas", pd.__version__, "| furnace.py", furnace.__version__, "| oxide_tools.py", oxide_tools.__version__,
      "| r2r_tools.py", r2r_tools.__version__, "| capstone.py", capstone.__version__, "| process_twin.py", process_twin.__version__)"""),
        md("""## Step 1: the prepared components

A twin in a fab is rarely written from scratch: it is assembled from models and tools that were each built and tested on their own. This track supplies these components:

| component | from | what it does |
|---|---|---|
| `qualification_runs(key)`, `calibrate(key)` | Module 3 | 12 qualification runs and the weighted Deal-Grove fit (SciPy's `curve_fit`) |
| `predict(fit, t)` | Module 3 | the calibrated thickness after `t` minutes, `predict(fit, t)["x"]`, with its bands |
| `kalman_update(d, P, y, Q, R)` | Module 5 | one update of the offset estimate; an observation of NaN keeps the estimate |
| `next_time(d_hat, target, B, BA, xi, limits)` | Module 5 | the oxidation time that aims the twin at the target, within 80 to 100 minutes |
| `run_twin(key, scenario, predict_next, offset_observation)` | new | the 60 production runs with the twin in the loop, using your two connections |
| `mae_ci(errors)` | Module 6 | the mean absolute error and the half-width of its 95 % interval |

What is missing is the **connection**: how the twin turns the calibrated model and the offset estimate into a prediction, and how it turns a measurement into an observation of the offset. That is your part, in Step 3."""),
        md("""## Step 2: calibrate the model on the qualification runs

Before production, the furnace runs 12 **qualification runs** of 60 to 126 minutes on bare wafers, with a freshly calibrated thermocouple, and one monitor wafer of each is measured (standard deviation 0.32 nm). `calibrate()` fits B and B/A to them with the Module 3 helper.

The fitted constants are not the textbook values: on bare wafers the oxide grows faster at first than Deal-Grove says (the thin-oxide regime of Module 3), and the two constants absorb that. The cell also prints χ²/dof, as in Module 3: near 1 the residuals are as large as the measurement noise; well above 1 the model leaves something out, here the thin-oxide regime. The calibrated model is good **inside** the qualified range of 60 to 126 minutes, which covers the recipe's 80 to 100 minutes. Outside that range nothing has tested it."""),
        code("""qual = qualification_runs(EXAMPLE_KEY)
fit = calibrate(EXAMPLE_KEY)
setup = furnace.r2r_setup(EXAMPLE_KEY)
print(qual.to_string(index=False))
print()
print(f"B = {fit['B']:.2f} nm^2/min (+/- {fit['se_B']:.2f}),  B/A = {fit['BA']:.4f} nm/min (+/- {fit['se_BA']:.4f})")
print(f"chi2 / dof = {fit['chi2']:.1f} / {fit['dof']} = {fit['chi2'] / fit['dof']:.2f}:  the residuals are about "
      f"{np.sqrt(fit['chi2'] / fit['dof']):.1f} times the stated measurement noise")
print(f"target thickness of recipe DRYOX-90: {setup['target_nm']} nm")"""),
        md("""## Step 3: your task, connect the components

The twin's prediction for a run of `t_min` minutes is the calibrated Deal-Grove thickness plus the current offset estimate `d_hat` (Module 5: the offset is how far the furnace now sits from the model):

prediction = `predict(fit, t_min)["x"][0]` + `d_hat`

After the run, the monitor wafer gives one **observation of the offset**: the measured thickness minus the calibrated model's thickness for the time that ran (without `d_hat`: the observation is of the offset itself). When a run has no measurement, `measured_nm` is NaN, and so is the observation; the Kalman update then keeps its estimate.

**Your task:** write `predict_next(t_min, fit, d_hat)` and `offset_observation(measured_nm, t_min, fit)`. Each returns one number in nm."""),
        code('''def predict_next(t_min, fit, d_hat):
    """The twin's prediction (nm) of the monitor-wafer thickness for a run of t_min minutes."""
    # YOUR CODE HERE
    raise NotImplementedError("Complete predict_next() first.")


def offset_observation(measured_nm, t_min, fit):
    """The offset observed in one run: measured minus the calibrated model (nm); NaN when there is no measurement."""
    # YOUR CODE HERE
    raise NotImplementedError("Complete offset_observation() first.")'''),
        code('''print(predict_next(90.0, fit, 0.0), "  (the calibrated model at 90 minutes)")
print(predict_next(90.0, fit, -1.0), "  (should be 1 nm less than the line above)")
print(offset_observation(60.0, 90.0, fit), "  (60 nm measured, minus the model at 90 minutes)")
print(offset_observation(np.nan, 90.0, fit), "  (should be nan: no measurement)")'''),
        md("## Step 4: verify the connected twin\n\n" + verification_intro() + """

Four tests: (1) when the twin aims at the target and its time is inside the limits, its prediction is the target exactly; (2) if a run measures exactly what the twin predicted, the observed offset equals the estimate; (3) a run with no measurement leaves the estimate unchanged; (4) the predictions lie in a plausible range for the recipe."""),
        code('''target, lim, xi = setup["target_nm"], setup["t_limits"], furnace.R2R_XI_NM
ok1 = True
for d in (-2.0, -0.5, 0.0, 1.5):
    t = next_time(d, target, fit["B"], fit["BA"], xi, lim)
    if lim[0] < t < lim[1]:
        ok1 &= abs(predict_next(t, fit, d) - target) < 1e-6
ok2 = all(abs(offset_observation(predict_next(t, fit, d), t, fit) - d) < 1e-9 for t in (82.0, 90.0, 98.0) for d in (-1.2, 0.0, 0.8))
d_new, P_new = kalman_update(0.7, 0.05, offset_observation(np.nan, 90.0, fit), 0.01, 0.16)
ok3 = d_new == 0.7 and abs(P_new - 0.06) < 1e-12
ok4 = all(50.0 < predict_next(t, fit, 0.0) < 75.0 for t in (80.0, 90.0, 100.0))
for i, ok in enumerate((ok1, ok2, ok3, ok4), 1):
    print(f"verification test {i}:", "ok" if ok else "FAILED: check Step 3")'''),
        md(f"""## Step 5: validate on held-out runs

Now the twin runs production: 60 runs of the **drift** scenario of Module 5, which it has never seen. Before each run it recommends the oxidation time and predicts the monitor wafer; after the run the measurement updates the offset estimate. Each run is predicted **before** its measurement is used, so every error is an honest out-of-sample error, even though the twin keeps learning. One run has no measurement and adds no error.

The MAE is a mean over 59 runs, and those runs are a sample: `mae_ci()` gives the half-width of its 95 % interval, t(0.975, n − 1) × s / √n, as in Module 6."""),
        code('''log_drift = run_twin(EXAMPLE_KEY, "drift", predict_next, offset_observation)
plot_log(log_drift, title=f"Process twin, example furnace (KEY {EXAMPLE_KEY}), drift scenario")
held = runs_between(log_drift, HELDOUT_RUNS)
h_mae, h_hw, h_n = mae_ci(held["error_nm"])
print(f"held-out: MAE {h_mae:.3f} nm, 95 % interval +/- {h_hw:.3f} nm, over {h_n} measured runs (tolerance {TOLERANCE} nm)")'''),
    ] + verdict_cells(0.5, "nm", "h_mae, h_hw") + [
        md("""## Step 7: compare with the baseline

A twin should beat the simplest thing that could work. The **baseline** here is the calibrated Deal-Grove model alone, with no offset estimate: Module 3 without Module 5. `run_twin()` records its prediction for the same runs and the same oxidation times.

If the twin did not beat its baseline, that would be a valid finding too: the simpler model would be enough."""),
        code('''b_mae, b_hw, b_n = mae_ci(held["baseline_error_nm"])
print(f"baseline (Deal-Grove alone): MAE {b_mae:.3f} nm, +/- {b_hw:.3f} nm")
print(f"twin:                        MAE {h_mae:.3f} nm, +/- {h_hw:.3f} nm")
print(f"the twin's error is {h_mae / b_mae:.0%} of the baseline's")'''),
        md("""## Step 8: the challenging condition

Every validation needs one condition that is harder than normal production. Here it is the **failing thermocouple** of Module 5: from a run between 25 and 35 its bias grows several times faster. The offset estimate was tuned for slow drift (Q = 0.01 nm² per run), so it now lags behind. The test uses runs 31 to 60 of the failing scenario, after the failure has begun. The challenge is built in on purpose, and the notebook says so: a twin is only trusted where it has been tested."""),
        code('''log_fail = run_twin(EXAMPLE_KEY, "failing", predict_next, offset_observation)
plot_log(log_fail, title=f"Process twin, example furnace (KEY {EXAMPLE_KEY}), failing thermocouple")
chal = runs_between(log_fail, CHALLENGING_RUNS)
c_mae, c_hw, c_n = mae_ci(chal["error_nm"])
print(f"challenging: MAE {c_mae:.3f} nm, +/- {c_hw:.3f} nm, over {c_n} runs:", verdict(c_mae, c_hw, TOLERANCE))'''),
        md(f"""## Step 9: your own furnace

Open the **{quiz}** in Canvas. Question 1 gives you a **KEY**, a whole number from 100 to 999. Type it below in place of `None` and run the cell. It repeats Steps 5 to 8 on **your furnace** with **your functions** and prints the numbers for the data sheet."""),
        code('''KEY = None      # <- your KEY from Question 1 of the project quiz, for example KEY = 417

if KEY is None:
    print("Type your KEY in place of None, then run this cell again.")
else:
    my_drift = run_twin(KEY, "drift", predict_next, offset_observation)
    my_fail = run_twin(KEY, "failing", predict_next, offset_observation)
    my_held = runs_between(my_drift, HELDOUT_RUNS)
    results = {"heldout": mae_ci(my_held["error_nm"]),
               "challenging": mae_ci(runs_between(my_fail, CHALLENGING_RUNS)["error_nm"]),
               "baseline": mae_ci(my_held["baseline_error_nm"])}
    print(f"Your furnace, KEY {KEY} (tolerance {TOLERANCE} nm):")
    for name, (m, hw, n) in results.items():
        v = verdict(m, hw, TOLERANCE) if name != "baseline" else "(for comparison)"
        print(f"  {name:12s} MAE {m:.3f} nm   +/- {hw:.3f}   n = {n:2d}   {v}")
    plot_log(my_drift, title=f"Your furnace (KEY {KEY}), drift scenario")'''),
    ] + datasheet_cells("process", quiz) + [
        code(f'''# The course's check code is in dt_m7_check.py, next to this notebook.
# It is kept out of the notebook so that the course's reference solutions are not on screen.
from dt_m7_check import check_process_twin

passed, messages, value = check_process_twin(predict_next, offset_observation, verdict, datasheet, KEY)
for line_text in messages:
    print(line_text)
if passed:
    print()
    print(f"Verification value for KEY = {{KEY}}: {{value:.3f}}")
    print("Type this number into Question 1 of the {quiz}.")'''),
        closing_cell("""**What the process twin showed.** Kept in sync, the calibrated model predicts the monitor wafer to within its tolerance during normal drift, several times better than the model alone. When the thermocouple fails, the same twin lags behind, because its filter was tuned for slow drift: that condition goes on the data sheet as one where its predictions must not be used, and Module 5's alarm and maintenance rule take over."""),
    ]
    return cells


def metrology_cells():
    K = EXAMPLE_KEY
    quiz = "Module 7 project quiz for the metrology twin"
    cells = [
        md(f"""# Module 7 capstone, metrology twin: virtual metrology that stays in sync

**Digital Twins for Semiconductor Manufacturing · Module 7 · about 2 hours** · notebook version {VERSION}

This is the **metrology twin** track of the capstone. You join two prepared components into one twin that predicts the oxide thickness of every product wafer of the course's furnace, and test it the way a twin should be tested before anyone relies on it:

- the **hybrid virtual-metrology model** of Module 4 predicts each wafer from the run's trace summary: the Deal-Grove physics prediction plus a learned ridge correction;
- the **offset estimate** of Module 5, a Kalman filter, keeps those predictions in sync, using one sampled wafer per run (slot 65) that is still measured.

The thermocouples age in production, so a model trained once slowly drifts away from the furnace; the sampled wafer is what brings it back. You write two short functions that connect the components, one function that turns a test result into a verdict, and the **twin data sheet**: what the twin is for, what it uses, how it performed, where it must not be used, one risk, and what to do when confidence is low.

The test plan is fixed before any test runs:

| test | data | pass when |
|---|---|---|
| held-out | production runs 41 to 100, every wafer predicted before its run's sampled wafer is used | mean absolute error (MAE) at most 0.5 nm |
| baseline | the mean of the last 5 sampled wafers, on the same runs | (for comparison) |
| challenging | runs 101 to 115, after the center thermocouple was replaced | MAE at most 0.5 nm |

Steps 1 to 8 use the **course example furnace (KEY {K})**, so you can compare your numbers with the Module 7 page. Step 9 repeats the tests on your own furnace, with the KEY from Question 1 of the **{quiz}**. All data are **simulated**. Run each code cell with **Shift + Enter**, in order."""),
        md("""## Step 0: start Python

The first time, Python can take up to a minute to start, and loading scikit-learn adds a little more. The cell prints the versions of the libraries and of the course files next to this notebook."""),
        code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import furnace, vm_tools, capstone, metrology_twin
from vm_tools import FEATURES, physics_prediction
from capstone import mae_ci, blank_datasheet, show_options, show_datasheet
from metrology_twin import (vm_production, train_correction, kalman_update, run_twin, per_run_mae, plot_log,
                            TOLERANCE_NM, HELDOUT_RUNS, CHALLENGING_RUNS, SAMPLED_SLOT)

EXAMPLE_KEY = {K}
TOLERANCE = TOLERANCE_NM
print("Notebook version {VERSION}")
print("NumPy", np.__version__, "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__, "| furnace.py", furnace.__version__,
      "| vm_tools.py", vm_tools.__version__, "| capstone.py", capstone.__version__, "| metrology_twin.py", metrology_twin.__version__)"""),
        md("""## Step 1: the prepared components

A twin in a fab is rarely written from scratch: it is assembled from models and tools that were each built and tested on their own. This track supplies these components:

| component | from | what it does |
|---|---|---|
| `vm_production(key)` | new data | 115 runs of recipe DRYOX-90: five wafers per run with their trace summary and thickness |
| `physics_prediction(table)` | Module 4 | the Deal-Grove thickness at each wafer's center-zone reading and oxidation time |
| `train_correction(train)` | Module 4 | the scaler-and-ridge pipeline fitted to the residual, measured minus physics, with run-blocked CV |
| `kalman_update(d, P, y, Q, R)` | Module 5 | one update of the offset estimate |
| `run_twin(key, vm_predict, offset_observation)` | new | runs 41 to 115 with the twin in the loop, using your two connections |
| `per_run_mae(log, column, runs)`, `mae_ci(errors)` | Module 6 | the mean absolute error of each run's five wafers, and the MAE with its 95 % interval |

What is missing is the **connection**: how the twin turns the two parts of the hybrid model into one prediction per wafer, and how it turns the sampled wafer's measurement into an observation of the offset. That is your part, in Step 3."""),
        md("""## Step 2: the data and the hybrid model

The first 40 runs, with five measured wafers each, train the learned correction of the hybrid model. In production (runs 41 on), only the sampled wafer of each run (slot 65) goes to the twin. For this validation the other four wafers were measured too, so that the twin's predictions can be checked on every wafer; a fab does this on a sample of runs when it qualifies a virtual-metrology model.

The maintenance log has one entry that matters: the furnace's center thermocouple was **replaced before run 101**."""),
        code("""data = vm_production(EXAMPLE_KEY)
train, prod, after = data["train"], data["production"], data["after_replacement"]
print({name: f"{t['run_index'].min()} to {t['run_index'].max()} ({t['run_index'].nunique()} runs, {len(t)} wafers)" for name, t in data.items()})
model = train_correction(train)
print("ridge penalty chosen by run-blocked cross-validation:", model.named_steps["ridge"].alpha)
print(prod.head(5).to_string(index=False))"""),
        md("""## Step 3: your task, connect the components

The hybrid prediction for the wafers of a run is the physics prediction plus the learned correction (Module 4):

vm = `physics_prediction(table)` + `model.predict(table[FEATURES])`

`run_twin()` adds the current offset estimate `d_hat` to it. After the run, the sampled wafer is measured, and its **observation of the offset** is the measured thickness minus the hybrid prediction `vm` for that wafer (without `d_hat`: the observation is of the offset itself).

**Your task:** write `vm_predict(model, table)`, which returns one prediction (nm) per row of `table`, and `offset_observation(measured_nm, vm_nm)`, which returns one number in nm. Pass only the model's inputs to it: `table[FEATURES]`."""),
        code('''def vm_predict(model, table):
    """The hybrid virtual-metrology prediction (nm) for every wafer (row) of table."""
    # YOUR CODE HERE
    raise NotImplementedError("Complete vm_predict() first.")


def offset_observation(measured_nm, vm_nm):
    """The offset observed on the sampled wafer: measured minus the hybrid prediction (nm)."""
    # YOUR CODE HERE
    raise NotImplementedError("Complete offset_observation() first.")'''),
        code('''run41 = prod[prod["run_index"] == 41]
print(np.round(vm_predict(model, run41), 2), "  (five predictions, one per wafer of run 41)")
print(np.round(run41["thickness_nm"].to_numpy(), 2), "  (what the five wafers measured)")
print(offset_observation(55.40, 55.10), "  (should be 0.3, up to rounding)")'''),
        md("## Step 4: verify the connected twin\n\n" + verification_intro() + """

Four tests: (1) your prediction minus the physics prediction is exactly the learned correction; (2) if the sampled wafer measures the prediction plus 0.37 nm, the observed offset is 0.37 nm; (3) on the training runs the residual, measured minus prediction, averages to zero, as a ridge fit with an intercept must; (4) there is one prediction per wafer."""),
        code('''vm_train = np.asarray(vm_predict(model, train), dtype=float)
ok1 = np.allclose(vm_train - physics_prediction(train), model.predict(train[FEATURES]))
ok2 = abs(offset_observation(float(vm_train[0]) + 0.37, float(vm_train[0])) - 0.37) < 1e-9
ok3 = abs(np.mean(train["thickness_nm"].to_numpy() - vm_train)) < 1e-6
ok4 = vm_train.shape == (len(train),)
for i, ok in enumerate((ok1, ok2, ok3, ok4), 1):
    print(f"verification test {i}:", "ok" if ok else "FAILED: check Step 3")'''),
        md("""## Step 5: validate on held-out runs

Now the twin runs production: runs 41 to 100, which it has never seen. For each run it predicts all five wafers, then the sampled wafer is measured and updates the offset estimate for the next run. Each run is predicted **before** its sampled wafer is used, so every error is an honest out-of-sample error.

The five wafers of a run share the run's effects, so they are not five independent tests. The test unit is the **run**: `per_run_mae()` gives the mean absolute error of each run's five wafers, and `mae_ci()` the MAE over the runs with the half-width of its 95 % interval, t(0.975, n − 1) × s / √n, as in Module 6."""),
        code('''log = run_twin(EXAMPLE_KEY, vm_predict, offset_observation, model=model)
plot_log(log, title=f"Metrology twin, example furnace (KEY {EXAMPLE_KEY})")
h_mae, h_hw, h_n = mae_ci(per_run_mae(log, "error_nm", HELDOUT_RUNS))
print(f"held-out: MAE {h_mae:.3f} nm, 95 % interval +/- {h_hw:.3f} nm, over {h_n} runs (tolerance {TOLERANCE} nm)")'''),
    ] + verdict_cells(0.5, "nm", "h_mae, h_hw") + [
        md("""## Step 7: compare with the baseline

A twin should beat the simplest thing that could work. The **baseline** here predicts every wafer of a run as the mean of the last 5 sampled wafers: what an engineer could do with no model at all. `run_twin()` records it for the same runs.

If the twin did not beat its baseline, that would be a valid finding too: the simpler method would be enough."""),
        code('''b_mae, b_hw, b_n = mae_ci(per_run_mae(log, "baseline_error_nm", HELDOUT_RUNS))
print(f"baseline (mean of the last 5 sampled wafers): MAE {b_mae:.3f} nm, +/- {b_hw:.3f} nm")
print(f"twin:                                         MAE {h_mae:.3f} nm, +/- {h_hw:.3f} nm")
print(f"the twin's error is {h_mae / b_mae:.0%} of the baseline's")'''),
        md("""## Step 8: the challenging condition

Every validation needs one condition that is harder than normal production. Here it is the **thermocouple replacement** before run 101. The new thermocouple reads a few degrees high; the controller holds the reading at its setpoint, so the furnace runs cooler than its traces say, and the wafers come out thinner than the model predicts from those traces. The offset estimate notices it from the sampled wafers, but it was tuned for slow drift, so it takes many runs to catch up. The test uses the 15 runs after the replacement. The challenge is built in on purpose, and the notebook says so: a twin is only trusted where it has been tested."""),
        code('''c_mae, c_hw, c_n = mae_ci(per_run_mae(log, "error_nm", CHALLENGING_RUNS))
print(f"challenging: MAE {c_mae:.3f} nm, +/- {c_hw:.3f} nm, over {c_n} runs:", verdict(c_mae, c_hw, TOLERANCE))
print("per-run MAE after the replacement:", np.round(per_run_mae(log, "error_nm", CHALLENGING_RUNS).to_numpy(), 2))'''),
        md(f"""## Step 9: your own furnace

Open the **{quiz}** in Canvas. Question 1 gives you a **KEY**, a whole number from 100 to 999. Type it below in place of `None` and run the cell. It repeats Steps 5 to 8 on **your furnace** with **your functions** and prints the numbers for the data sheet."""),
        code('''KEY = None      # <- your KEY from Question 1 of the project quiz, for example KEY = 417

if KEY is None:
    print("Type your KEY in place of None, then run this cell again.")
else:
    my_log = run_twin(KEY, vm_predict, offset_observation)
    results = {"heldout": mae_ci(per_run_mae(my_log, "error_nm", HELDOUT_RUNS)),
               "challenging": mae_ci(per_run_mae(my_log, "error_nm", CHALLENGING_RUNS)),
               "baseline": mae_ci(per_run_mae(my_log, "baseline_error_nm", HELDOUT_RUNS))}
    print(f"Your furnace, KEY {KEY} (tolerance {TOLERANCE} nm):")
    for name, (m, hw, n) in results.items():
        v = verdict(m, hw, TOLERANCE) if name != "baseline" else "(for comparison)"
        print(f"  {name:12s} MAE {m:.3f} nm   +/- {hw:.3f}   n = {n:2d}   {v}")
    plot_log(my_log, title=f"Your furnace (KEY {KEY})")'''),
    ] + datasheet_cells("metrology", quiz) + [
        code(f'''# The course's check code is in dt_m7_check.py, next to this notebook.
# It is kept out of the notebook so that the course's reference solutions are not on screen.
from dt_m7_check import check_metrology_twin

passed, messages, value = check_metrology_twin(vm_predict, offset_observation, verdict, datasheet, KEY)
for line_text in messages:
    print(line_text)
if passed:
    print()
    print(f"Verification value for KEY = {{KEY}}: {{value:.3f}}")
    print("Type this number into Question 1 of the {quiz}.")'''),
        closing_cell("""**What the metrology twin showed.** Kept in sync by one sampled wafer per run, the hybrid model predicts the unmeasured wafers to within its tolerance while the thermocouples age, far better than the recent average of sampled wafers. After a thermocouple replacement it is wrong for many runs, because the traces it reads have changed meaning: that condition goes on the data sheet, and a fab would measure every wafer until the twin has caught up again."""),
    ]
    return cells


def toolset_cells():
    K = EXAMPLE_KEY
    quiz = "Module 7 project quiz for the toolset twin"
    cells = [
        md(f"""# Module 7 capstone, toolset twin: a line twin calibrated from the fab's log

**Digital Twins for Semiconductor Manufacturing · Module 7 · about 2 hours** · notebook version {VERSION}

This is the **toolset twin** track of the capstone. You join prepared components into one twin of the furnace, lithography and etch line of Module 6, and test it the way a twin should be tested before anyone relies on it:

- the **SimPy line model** of Module 6, whose furnace tubes drift and are calibrated by the predictive rule of Modules 5 and 6;
- the fab's **lot log**: when each lot was released, and when it started and finished at each tool group.

The twin forecasts the **weekly mean cycle time** of the line, for capacity planning. You estimate its settings from the log (the connection between the log and the model), write one function that turns a test result into a verdict, and fill in the **twin data sheet**: what the twin is for, what it uses, how it performed, where it must not be used, one risk, and what to do when confidence is low.

The "fab" here is the same simulator with settings of its own, which the twin has to estimate from the log, and with one thing the twin does not model. The test plan is fixed before any test runs:

| test | data | pass when |
|---|---|---|
| calibration | weeks 1 to 6 of the log | (estimates only) |
| held-out | weeks 7 to 30: the twin's forecast of each week against the log | mean absolute weekly error (MAE) at most 1.0 h |
| baseline | the mean weekly cycle time of weeks 1 to 6, on the same weeks | (for comparison) |
| challenging | weeks 31 to 40, when the lithography tool breaks down now and then | MAE at most 1.0 h |

Steps 1 to 8 use the **course example line (KEY {K})**, so you can compare your numbers with the Module 7 page. Step 9 repeats the tests on your own line, with the KEY from Question 1 of the **{quiz}**. All data are **simulated**. The twin runs 20 replications for every forecast, so Steps 5, 8 and 9 take a while (up to about a minute each in the browser). Run each code cell with **Shift + Enter**, in order."""),
        md("""## Step 0: start Python

The first time, Python can take up to a minute to start. The cell prints the versions of the libraries and of the course files next to this notebook, including SimPy, which `line.py` imports from the wheel file next to it."""),
        code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import line, capstone, toolset_twin
from capstone import mae_ci, blank_datasheet, show_options, show_datasheet
from toolset_twin import (fab_log, downtime_log, twin_config, twin_run, weekly_cycle_time, replay, plot_weeks, week_window,
                          TOLERANCE_H, CAL_WEEKS, HELDOUT_WEEKS, CHALLENGING_WEEKS, WEEK_H)

EXAMPLE_KEY = {K}
TOLERANCE = TOLERANCE_H
print("Notebook version {VERSION}")
print("NumPy", np.__version__, "| pandas", pd.__version__, "| SimPy", line.simpy.__version__, "| line.py", line.__version__,
      "| capstone.py", capstone.__version__, "| toolset_twin.py", toolset_twin.__version__)"""),
        md("""## Step 1: the prepared components

A twin in a fab is rarely written from scratch: it is assembled from models and tools that were each built and tested on their own. This track supplies these components:

| component | from | what it does |
|---|---|---|
| `fab_log(key)`, `downtime_log(key)` | new data | 40 weeks of the line's lot log, and the lithography breakdowns the fab recorded |
| `twin_config(estimates)` | Module 6 | the twin's line settings from your estimates, with the predictive maintenance rule |
| `twin_run(cfg, key, weeks)`, `replay(cfg, key, weeks)` | Module 6 | the twin driven by the log's own release times: one replication, or the weekly mean over 20 |
| `weekly_cycle_time(log, weeks)` | Module 6 | the mean cycle time of the lots that leave in each week |
| `mae_ci(errors)` | Module 6 | the mean absolute error and the half-width of its 95 % interval |

The twin is **trace-driven**: it does not invent its own lot releases, it releases its lots at the times the fab released them. Its process times, tube drift and calibrations are its own and random, so 20 replications give its forecast for a week. That way a busy week in the fab is a busy week in the twin, and the comparison tests the model, not the luck of the releases (Law 2015 recommends this for validating a simulation against a real system).

What is missing is the **connection** between the log and the model: the twin's release rate and process times have to be estimated from the log. That is your part, in Step 3."""),
        md("""## Step 2: the fab's log

The log starts with the line already running. Each row is one lot: its release `t_in`, the start and end of its furnace run (`f_start`, `f_end`), of its lithography step (`l_start`, `l_end`), the start of its etch step `e_start`, and when it left the line `t_out`, all in hours. A lot still in the line when the log ends has NaN for the steps it has not reached. The downtime log lists the lithography breakdowns the fab recorded."""),
        code("""log = fab_log(EXAMPLE_KEY)
down = downtime_log(EXAMPLE_KEY)
print(log.head(6).round(2).to_string(index=False))
print(f"{len(log)} lots in 40 weeks")
weeks_down = sorted(set((down["t_start"] // WEEK_H + 1).astype(int)))
print("lithography breakdowns recorded:", len(down), "in weeks", weeks_down[0] if weeks_down else "-", "to", weeks_down[-1] if weeks_down else "-")
observed_all = weekly_cycle_time(log, (1, 40))
print(observed_all.round(2).to_string())"""),
        md("""## Step 3: your task, estimate the twin's settings from the log

The twin needs seven numbers, estimated from the lots released in the calibration weeks:

- `arrival_rate`: the lots released in the calibration window divided by its length in hours, `len(calib) / hours` (count every lot released, finished or not);
- `furnace_h`, `litho_h`, `etch_h`: the mean **process time** of each tool group: `f_end - f_start`, `l_end - l_start` and `t_out - e_start`. These start when the tool starts, so they leave out the waiting;
- `furnace_cv`, `litho_cv`, `etch_cv`: the coefficient of variation of each process time, its standard deviation divided by its mean. Use the sample standard deviation, n − 1 in the divisor, which is what pandas' `.std()` gives.

Lots still in the line have NaN for the steps they have not finished: drop them from each process time with `.dropna()` before taking the mean and the standard deviation.

**Your task:** write `twin_estimates(calib, hours)`. `calib` is the part of the log released in the calibration window and `hours` the window's length. Return a dict with the seven keys above."""),
        code('''def twin_estimates(calib, hours):
    """The twin's release rate (lots/h), mean process times (h) and their coefficients of variation, from the lots released
    in the calibration window."""
    # YOUR CODE HERE
    raise NotImplementedError("Complete twin_estimates() first.")'''),
        code('''a, z = week_window(CAL_WEEKS)
calib = log[(log["t_in"] >= a) & (log["t_in"] < z)]
est = twin_estimates(calib, z - a)
print({k: round(float(v), 4) for k, v in est.items()})
print("(the release rate should be near 0.7 to 0.76 lots per hour, the process times near 1.8, 1.0 and 1.2 h)")
cfg = twin_config(est)'''),
        md("## Step 4: verify the connected twin\n\n" + verification_intro() + """

Four tests on one replication of the twin for the held-out weeks: (1) it released exactly the lots the log released in those weeks (the trace-driven input works); (2) every lot's steps happen in order: furnace after release, lithography after the furnace, etch after lithography; (3) its mean lithography process time matches your estimate to within a few percent, as it must with about 4,000 lots drawn from it; (4) its mean cycle time is longer than the sum of the three process times, since lots also wait."""),
        code('''one = twin_run(cfg, EXAMPLE_KEY, HELDOUT_WEEKS, seed=1)
a7, z30 = week_window(HELDOUT_WEEKS)
lots = one.lots
n_log = int(((log["t_in"] >= a7 - 200.0) & (log["t_in"] < z30)).sum())
ok1 = len(lots) == n_log
done = lots.dropna()
ok2 = bool(((done["f_start"] >= done["t_in"]) & (done["l_start"] >= done["f_end"]) & (done["e_start"] >= done["l_end"]) & (done["t_out"] >= done["e_start"])).all())
litho = (lots["l_end"] - lots["l_start"]).dropna()
ok3 = abs(litho.mean() / est["litho_h"] - 1.0) < 0.05
ok4 = (lots["t_out"] - lots["t_in"]).dropna().mean() > est["furnace_h"] + est["litho_h"] + est["etch_h"]
for i, ok in enumerate((ok1, ok2, ok3, ok4), 1):
    print(f"verification test {i}:", "ok" if ok else "FAILED: check Step 3")'''),
        md("""## Step 5: validate on held-out weeks

Now the twin forecasts weeks 7 to 30, which played no part in the calibration. For each week, `replay()` runs 20 replications of the trace-driven twin and averages their weekly mean cycle time; `weekly_cycle_time()` gives what the log measured. The test unit is the **week**: `mae_ci()` gives the mean absolute weekly error with the half-width of its 95 % interval, t(0.975, n − 1) × s / √n, as in Module 6. This cell takes a while."""),
        code('''observed = weekly_cycle_time(log, HELDOUT_WEEKS)
twin = replay(cfg, EXAMPLE_KEY, HELDOUT_WEEKS)
baseline = float(weekly_cycle_time(log, CAL_WEEKS).mean())
plot_weeks(observed, twin, baseline, title=f"Toolset twin, example line (KEY {EXAMPLE_KEY}), held-out weeks")
h_mae, h_hw, h_n = mae_ci(observed - twin)
print(f"held-out: MAE {h_mae:.3f} h, 95 % interval +/- {h_hw:.3f} h, over {h_n} weeks (tolerance {TOLERANCE} h)")'''),
    ] + verdict_cells(1.0, "h", "h_mae, h_hw") + [
        md("""## Step 7: compare with the baseline

A twin should beat the simplest thing that could work. The **baseline** here forecasts every week as the mean weekly cycle time of the six calibration weeks: what a planner could do with the log alone and no model.

If the twin did not beat its baseline, that would be a valid finding too: the simpler method would be enough."""),
        code('''b_mae, b_hw, b_n = mae_ci(observed - baseline)
print(f"baseline (mean of weeks 1 to 6, {baseline:.2f} h): MAE {b_mae:.3f} h, +/- {b_hw:.3f} h")
print(f"twin:                                  MAE {h_mae:.3f} h, +/- {h_hw:.3f} h")
print(f"the twin's error is {h_mae / b_mae:.0%} of the baseline's")'''),
        md("""## Step 8: the challenging condition

Every validation needs one condition that is harder than normal operation. Here it is weeks 31 to 40, when the lithography tool, the bottleneck of Module 6, breaks down now and then and is repaired in a few hours. The twin has no breakdown model: nothing in the calibration weeks showed one. The challenge is built in on purpose, and the notebook says so: a twin is only trusted where it has been tested. This cell takes a while."""),
        code('''obs_c = weekly_cycle_time(log, CHALLENGING_WEEKS)
twin_c = replay(cfg, EXAMPLE_KEY, CHALLENGING_WEEKS)
plot_weeks(obs_c, twin_c, baseline, title=f"Toolset twin, example line (KEY {EXAMPLE_KEY}), weeks with breakdowns")
c_mae, c_hw, c_n = mae_ci(obs_c - twin_c)
print(f"challenging: MAE {c_mae:.3f} h, +/- {c_hw:.3f} h, over {c_n} weeks:", verdict(c_mae, c_hw, TOLERANCE))'''),
        md(f"""## Step 9: your own line

Open the **{quiz}** in Canvas. Question 1 gives you a **KEY**, a whole number from 100 to 999. Type it below in place of `None` and run the cell. It repeats Steps 3 and 5 to 8 on **your line** with **your functions** and prints the numbers for the data sheet. It takes a minute or two."""),
        code('''KEY = None      # <- your KEY from Question 1 of the project quiz, for example KEY = 417

if KEY is None:
    print("Type your KEY in place of None, then run this cell again.")
else:
    my_log = fab_log(KEY)
    a, z = week_window(CAL_WEEKS)
    my_cfg = twin_config(twin_estimates(my_log[(my_log["t_in"] >= a) & (my_log["t_in"] < z)], z - a))
    my_obs = weekly_cycle_time(my_log, HELDOUT_WEEKS)
    my_twin = replay(my_cfg, KEY, HELDOUT_WEEKS)
    my_base = float(weekly_cycle_time(my_log, CAL_WEEKS).mean())
    results = {"heldout": mae_ci(my_obs - my_twin),
               "challenging": mae_ci(weekly_cycle_time(my_log, CHALLENGING_WEEKS) - replay(my_cfg, KEY, CHALLENGING_WEEKS)),
               "baseline": mae_ci(my_obs - my_base)}
    print(f"Your line, KEY {KEY} (tolerance {TOLERANCE} h):")
    for name, (m, hw, n) in results.items():
        v = verdict(m, hw, TOLERANCE) if name != "baseline" else "(for comparison)"
        print(f"  {name:12s} MAE {m:.3f} h   +/- {hw:.3f}   n = {n:2d}   {v}")
    plot_weeks(my_obs, my_twin, my_base, title=f"Your line (KEY {KEY}), held-out weeks")'''),
    ] + datasheet_cells("toolset", quiz) + [
        code(f'''# The course's check code is in dt_m7_check.py, next to this notebook.
# It is kept out of the notebook so that the course's reference solutions are not on screen.
from dt_m7_check import check_toolset_twin

passed, messages, value = check_toolset_twin(twin_estimates, verdict, datasheet, KEY)
for line_text in messages:
    print(line_text)
if passed:
    print()
    print(f"Verification value for KEY = {{KEY}}: {{value:.3f}}")
    print("Type this number into Question 1 of the {quiz}.")'''),
        closing_cell("""**What the toolset twin showed.** Calibrated from six weeks of the log and driven by each week's real releases, the twin forecasts the weekly cycle time to within its tolerance in normal weeks, better than the recent average. When the bottleneck starts to break down, the twin is wrong by hours, because breakdowns are not in it: that condition goes on the data sheet, and the twin needs a breakdown model, validated on new weeks, before it is used for those weeks."""),
    ]
    return cells


write(process_cells(), "DT-M7-capstone-process-twin.ipynb", "m7p",
      ["furnace.py", "oxide_tools.py", "r2r_tools.py", "capstone.py", "process_twin.py", "dt_m7_check.py"])
write(metrology_cells(), "DT-M7-capstone-metrology-twin.ipynb", "m7m",
      ["furnace.py", "vm_tools.py", "capstone.py", "metrology_twin.py", "dt_m7_check.py"])
write(toolset_cells(), "DT-M7-capstone-toolset-twin.ipynb", "m7t",
      ["line.py", "capstone.py", "toolset_twin.py", "dt_m7_check.py"], extra_install="%pip install -q simpy==4.1.2")
