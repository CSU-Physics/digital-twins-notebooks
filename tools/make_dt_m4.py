"""Build the Module 4 browser notebook (Digital Twins for Semiconductor Manufacturing), its Colab copy, and copy its files.

content/m4 holds furnace.py (1.4.0), vm_tools.py (supplied helpers) and dt_m4_check.py (copied from checks/).
The check cell only imports the check file, so the course's reference solutions are not on screen."""
import shutil
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "m4"
NAME = "DT-M4-lab-virtual-metrology.ipynb"
VERSION = "2026-10-22"
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}

shutil.copyfile(ROOT / "checks" / "dt_m4_check.py", OUT / "dt_m4_check.py")


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 4 lab: virtual metrology for oxide thickness

**Digital Twins for Semiconductor Manufacturing · Module 4 · about 80 minutes** · notebook version {VERSION}

A furnace run oxidizes a whole boat of wafers, but the metrology tool measures only a few of them, and some runs not at all. **Virtual metrology** predicts the thickness of the wafers nobody measured, from the furnace's own trace data. In this lab you build and test three virtual-metrology models on 60 simulated production runs:

1. a **baseline**: the mean of the most recently measured wafers;
2. a **data-driven model**: ridge regression on the furnace's trace summaries, in a scikit-learn pipeline;
3. a **hybrid model**: the Module 3 Deal-Grove prediction plus a learned correction.

You tune them with cross-validation that keeps whole runs together, compare them on a **time-ordered test set**, score them again on a new recipe **outside the training range**, and write a check that flags such wafers. Then you predict the wafers of a run that was not measured, for your own furnace.

All the data are **simulated**. The **Module 4 Lab Check** in Canvas asks about your results, so keep this lab open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

The first time, Python can take up to a minute to start, and scikit-learn adds a few seconds more. The cell prints the versions of the libraries and of the two course files next to this notebook: `furnace.py`, the furnace simulator, and `vm_tools.py`, the helpers for this lab."""),
code(f"""import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import furnace, vm_tools
from furnace import vm_runs, VM_PENDING_SLOT
from vm_tools import FEATURES, ALPHAS, time_split, block_folds, scores, physics_prediction, cv_rmse

print("Notebook version {VERSION}")
print("NumPy", np.__version__, "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__,
      "| furnace.py", furnace.__version__, "| vm_tools.py", vm_tools.__version__)"""),
md("""## Step 1: the production runs

`vm_runs(key)` returns the records of one simulated furnace running recipe **DRYOX-90**: dry oxidation at 1000 °C of product wafers that carry only their native oxide. A run-to-run controller sets each run's oxidation time between 80 and 100 minutes. After each run the metrology tool measures five sampled wafers, in slots 5, 35, 65, 95 and 125 of the 125-slot boat. Three tables, one row per wafer:

- `"measured"`: runs 1 to 60, in time order, with the measured thickness;
- `"shifted"`: runs 61 to 66 of a **new recipe, DRYOX-140**, with 140 minutes of oxidation (used in Step 10);
- `"pending"`: run 67, just finished, **not measured**: it has no thickness column.

The seven **features** are what the twin knows about a wafer before anyone measures it:

| feature | meaning |
|---|---|
| `ox_time_min` | oxidation time of the run (min), set by the controller |
| `T_top_C`, `T_center_C`, `T_bottom_C` | mean thermocouple reading of each of the three heater zones during oxidation (°C) |
| `h2o_ppm` | moisture in the oxygen (parts per million) |
| `o2_slm` | oxygen flow (standard litres per minute) |
| `slot` | the wafer's position in the boat, 1 at the top to 125 at the bottom |

The first six are the same for every wafer of a run; only `slot` changes within a run. The cell loads the example furnace (key 2026)."""),
code("""runs = vm_runs(2026)
measured = runs["measured"]
print(measured.head(7).to_string(index=False))
print()
print(f"{measured['run_id'].nunique()} measured runs, {len(measured)} measured wafers;",
      f"oxidation times {measured['ox_time_min'].min():.0f} to {measured['ox_time_min'].max():.0f} min")
print(f"thickness: mean {measured['thickness_nm'].mean():.2f} nm, standard deviation {measured['thickness_nm'].std():.2f} nm")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].scatter(measured["ox_time_min"], measured["thickness_nm"], s=12, c=measured["T_center_C"], cmap="viridis")
axes[0].set_xlabel("oxidation time (min)"); axes[0].set_ylabel("measured thickness (nm)")
axes[0].set_title("color: center-zone reading", fontsize=10)
axes[1].plot(measured["run_index"], measured["thickness_nm"], ".", ms=5)
axes[1].set_xlabel("run (time order)"); axes[1].set_ylabel("measured thickness (nm)")
plt.tight_layout(); plt.show()"""),
md("""**What to notice.** Most of the spread comes from the oxidation time: about half a nanometre more oxide for each extra minute. Within one oxidation time, the zone temperatures and the moisture move the thickness by a nanometre or two. No single feature explains everything, which is why the twin needs a model."""),
md("""## Step 2: split by time, before anything else

A virtual-metrology model is trained on past runs and used on future ones, so the honest test is in **time order**: the first 40 runs (200 wafers) to train and tune the models, and runs 41 to 60 (100 wafers) as the **test set**. The test set is not used for any choice in Steps 3 to 9; it is used once, in Step 9, to compare the finished models.

Two more rules from the reading:

- **All wafers of one run stay on the same side** of every split. Wafers of one run share the run's conditions and also effects no feature records, so a model that has seen four wafers of a run can guess the fifth.
- **Preprocessing is learned inside each fold.** Scaling, like every other step learned from data, is part of the pipeline that cross-validation refits.

For cross-validation inside the training runs, the lab uses **folds of whole runs**: `block_folds()` puts runs 1 to 8 in fold 1, runs 9 to 16 in fold 2, and so on, and returns a scikit-learn `GroupKFold` with those groups."""),
code("""train, test = time_split(measured)
cv, groups = block_folds(train)
print(f"training: runs {train['run_index'].min()} to {train['run_index'].max()}, {len(train)} wafers")
print(f"test:     runs {test['run_index'].min()} to {test['run_index'].max()}, {len(test)} wafers")
print("cross-validation folds (run blocks):", [int(g) for g in sorted(set(groups))])

X_train, y_train = train[FEATURES], train["thickness_nm"]
X_test, y_test = test[FEATURES], test["thickness_nm"]"""),
md("""## Step 3: your task, the baseline `baseline_recent()`

Before any model, a **baseline**: the simplest prediction a fab could use without one. Here it is the mean thickness of the wafers measured in the **3 runs just before** this run. Any model has to beat it to be worth running.

**Your task:** write `baseline_recent(table, n_runs=3)`. For every row (wafer) of `table`, return the mean measured thickness of **all wafers** in the `n_runs` runs just before that wafer's run, by `run_index`. The wafers of the run itself must not be used: when a run finishes, its own wafers are not measured yet. For the first `n_runs` runs, there are not enough earlier runs: return NaN.

Return one value per row, either a pandas Series with the table's index or an array in the table's row order.

*Hint:* per run, the sum and the count of the thicknesses, `table.groupby("run_index")["thickness_nm"].agg(["sum", "count"])`; then `.rolling(n_runs).sum()` and `.shift(1)` on each; divide; and bring the per-run result back to the wafers with `table["run_index"].map(...)`."""),
code("""def baseline_recent(table, n_runs=3):
    \"\"\"For each wafer: the mean measured thickness of all wafers in the n_runs runs before its run (NaN for the first n_runs runs).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete baseline_recent() first.")"""),
code("""base_all = baseline_recent(measured, 3)
base_test = np.asarray(base_all, dtype=float)[measured["run_index"].to_numpy() > 40]
s_base = scores(y_test, base_test)
print(f"baseline on the test runs: RMSE {s_base['rmse']:.2f} nm, mean absolute error {s_base['mae']:.2f} nm, "
      f"largest error {s_base['max_abs']:.2f} nm")

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(test["run_index"], y_test, ".", ms=6, label="measured")
ax.plot(test["run_index"], base_test, "-", lw=1.5, label="baseline: mean of the 3 previous runs")
ax.set_xlabel("run (time order)"); ax.set_ylabel("thickness (nm)"); ax.legend()
plt.show()"""),
md("""Write down the baseline's test RMSE, to 2 decimals: the lab check asks for it.

**What to notice.** The baseline follows slow changes, because it is always made of recent measurements, but it knows nothing about the run it predicts: not its oxidation time, not its temperatures. Its errors are several nanometres."""),
md("""## Step 4: your task, the data-driven model `make_ridge()`

**Ridge regression** is linear regression with a penalty: it chooses the coefficients w that minimize

Σ (yᵢ − ŷᵢ)² + α · Σ w_j²

The penalty α pulls the coefficients toward zero. It matters when features are correlated, like the three zone temperatures, which move together: plain least squares can then give them large coefficients of opposite sign that cancel. Because the penalty compares coefficients with each other, the features must be on the same scale first, so the model is a **pipeline**: `StandardScaler`, then `Ridge`. Inside cross-validation the whole pipeline, scaler included, is refitted on each training fold.

**Your task:** write `make_ridge(alpha)`, which returns `make_pipeline(StandardScaler(), Ridge(alpha=alpha))`."""),
code("""def make_ridge(alpha):
    \"\"\"A scikit-learn pipeline: StandardScaler, then Ridge(alpha=alpha).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete make_ridge() first.")"""),
code("""ridge1 = make_ridge(1.0).fit(X_train, y_train)
coef = pd.Series(ridge1[-1].coef_, index=FEATURES)
print("coefficients on the scaled features (nm per standard deviation of the feature):")
print(coef.round(3).to_string())
print(f"intercept (the mean training thickness): {ridge1[-1].intercept_:.2f} nm")"""),
md("""**What to notice.** The oxidation time has by far the largest coefficient. The three zone temperatures share the temperature effect between them; one of them may even come out negative, although a hotter zone never grows less oxide. With correlated features, single coefficients are not trustworthy; their combined effect is. The oxygen flow has almost no effect, which matches the simulator, where it has none."""),
md("""## Step 5: tune the penalty with folds of whole runs

The penalty α is a modeling choice, so it is tuned on the training runs only, never on the test runs. For each α in `ALPHAS`, `cv_rmse()` runs the cross-validation over the five run blocks: it fits `make_ridge(α)` on four blocks, computes the RMSE on the fifth, and averages over the five folds. (scikit-learn's `GridSearchCV` does the same loop for you; it is written out here so you can see it.)"""),
code("""cv_ridge = {a: cv_rmse(make_ridge(a), X_train, y_train, cv, groups) for a in ALPHAS}
for a, s in cv_ridge.items():
    print(f"alpha = {a:7.2f}: cross-validated RMSE {s:.3f} nm")
alpha_ridge = min(cv_ridge, key=cv_ridge.get)
print(f"chosen alpha: {alpha_ridge}")

ridge = make_ridge(alpha_ridge).fit(X_train, y_train)
print(f"data-driven model, refitted on all 40 training runs; its test RMSE is computed in Step 9")"""),
md("""**What to notice.** With seven features and 40 runs, small penalties all give about the same error, and large ones make it worse: the model is pulled away from the data (underfitting). Here plain least squares would do almost as well; ridge earns its place when there are many features, which real furnace data have, or when the model is flexible, as in the next step."""),
md("""## Step 6: why the folds hold whole runs

A **flexible model** makes the point: all products of the seven features up to degree 3 (119 columns), with a very small penalty, can bend to fit almost anything. The cell scores it two ways for several penalties:

- **random wafer folds** (`KFold` with shuffling): the five wafers of a run are spread over the folds, so the model is validated on wafers of runs it was trained on;
- **run-block folds**, as in Step 5: every validation wafer comes from runs the model has not seen.

It then fits each version on the 40 training runs and scores it on the test runs, which here, for once, only serve to show the effect."""),
code("""random_folds = KFold(n_splits=5, shuffle=True, random_state=0)
rows = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for a in [1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0]:
        flex = make_pipeline(PolynomialFeatures(3), StandardScaler(), Ridge(alpha=a))
        rows.append({"alpha": a,
                     "random wafer folds": cv_rmse(flex, X_train, y_train, random_folds),
                     "run-block folds": cv_rmse(flex, X_train, y_train, cv, groups),
                     "test runs": scores(y_test, flex.fit(X_train, y_train).predict(X_test))["rmse"]})
flex_table = pd.DataFrame(rows)
print("RMSE (nm) of the flexible model:")
print(flex_table.round({"random wafer folds": 3, "run-block folds": 3, "test runs": 3}).to_string(index=False, formatters={"alpha": lambda a: f"{a:g}"}))"""),
md("""**What to notice.** With random wafer folds, the smallest penalty looks best, with an error near the measurement noise: the model has learned each run's thickness from the other wafers of the same run. The run-block folds show what happens on runs it has not seen: the same model is far worse, and a larger penalty is needed. The test runs agree with the run-block folds. A random split would have chosen the wrong model, and reported an error it does not reach."""),
md("""## Step 7: the physics model alone

`physics_prediction(table)` is the Module 3 Deal-Grove model with the **textbook** constants, evaluated at each run's center-zone reading and oxidation time, from the native oxide (1.5 nm). It knows nothing about this furnace: not its own rate constants, not the extra growth of the thin-oxide regime (the wafers start bare), not the moisture or the boat position.

The cell computes its errors on the training wafers and plots them, measured minus physics, against the oxidation time."""),
code("""phys_train = physics_prediction(train)
phys_test = physics_prediction(test)
err_train = y_train.to_numpy() - phys_train
s_phys_train = scores(y_train, phys_train)
print(f"physics alone, training runs: mean error (measured minus physics) {s_phys_train['bias']:+.2f} nm, RMSE {s_phys_train['rmse']:.2f} nm")

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.scatter(train["ox_time_min"], err_train, s=12, c=train["h2o_ppm"], cmap="plasma")
ax.axhline(0, color="gray", lw=1)
ax.set_xlabel("oxidation time (min)"); ax.set_ylabel("measured minus physics (nm)")
ax.set_title("color: moisture (ppm)", fontsize=10)
plt.show()"""),
md("""**What to notice.** The physics model has the right shape but the wrong level: every wafer is several nanometres thicker than it predicts, mostly from the thin-oxide regime of Module 3, and by an amount that changes smoothly with the features (more moisture, more oxide). An error that is smooth and systematic is exactly what a small linear model can learn."""),
md("""## Step 8: your task, the hybrid model `hybrid_predict()`

The **hybrid model** keeps the physics and learns only what the physics gets wrong:

thickness ≈ physics prediction + correction,

where the correction is a ridge model of the physics model's error, measured minus physics, on the same seven features.

**Your task:** write `hybrid_predict(X_train, y_train, phys_train, X_new, phys_new, alpha)`:

1. fit `make_ridge(alpha)` on `X_train` to the errors `y_train - phys_train`;
2. return `phys_new` plus that model's prediction for `X_new`.

The second cell tunes α for the hybrid the same way as in Step 5: cross-validating a ridge model of the errors is the same as cross-validating the hybrid, because the physics part is fixed and has nothing to learn."""),
code("""def hybrid_predict(X_train, y_train, phys_train, X_new, phys_new, alpha):
    \"\"\"Physics prediction for the new wafers plus a ridge correction learned from the physics model's errors on the training wafers.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete hybrid_predict() first.")"""),
code("""cv_hybrid = {a: cv_rmse(make_ridge(a), X_train, err_train, cv, groups) for a in ALPHAS}
for a, s in cv_hybrid.items():
    print(f"alpha = {a:7.2f}: cross-validated RMSE of the hybrid {s:.3f} nm")
alpha_hybrid = min(cv_hybrid, key=cv_hybrid.get)
print(f"chosen alpha for the hybrid: {alpha_hybrid}")"""),
md("""## Step 9: compare the models on the test runs

Now, once, the test runs: the 100 wafers of runs 41 to 60, which no choice so far has used. The table gives each model's RMSE, mean absolute error, largest error and mean error (measured minus predicted)."""),
code("""pred_test = {
    "baseline": base_test,
    "physics alone": phys_test,
    "data-driven (ridge)": ridge.predict(X_test),
    "hybrid": hybrid_predict(X_train, y_train, phys_train, X_test, phys_test, alpha_hybrid),
}
compare = pd.DataFrame({name: scores(y_test, p) for name, p in pred_test.items()}).T
print("test runs 41 to 60, errors in nm:")
print(compare.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(test["run_index"], y_test, ".", color="k", ms=6, label="measured")
for name in ["baseline", "data-driven (ridge)", "hybrid"]:
    ax.plot(test["run_index"], pred_test[name], "o", ms=3, alpha=0.7, label=name)
ax.set_xlabel("run (time order)"); ax.set_ylabel("thickness (nm)"); ax.legend(ncol=2, fontsize=9)
plt.show()"""),
md("""**What to notice.** Both learned models beat the baseline by a wide margin, with errors close to the noise that no trace explains (the run-to-run and wafer-to-wafer effects plus the measurement). Inside the training range the data-driven and hybrid models are about equally good, and the physics model alone is far off. On these test runs, the extra physics buys nothing: a valid finding. The next step is where it matters."""),
md("""## Step 10: a challenging condition, built in on purpose

The engineers try a new recipe, **DRYOX-140**, for thicker oxide: 140 minutes of oxidation. The six runs of `"shifted"` are its first runs. Nothing about them was in the training runs, whose oxidation times stopped at 100 minutes. **This shift is deliberate**: it is the challenging condition of this lab.

The cell scores all four models on the 30 wafers of the new recipe. The baseline uses the last three DRYOX-90 runs: before the first DRYOX-140 wafers are measured, those are the most recent measurements the twin has."""),
code("""shifted = runs["shifted"]
X_shift, y_shift = shifted[FEATURES], shifted["thickness_nm"]
phys_shift = physics_prediction(shifted)
last3 = measured[measured["run_index"] > 57]["thickness_nm"].mean()
pred_shift = {
    "baseline": np.full(len(shifted), last3),
    "physics alone": phys_shift,
    "data-driven (ridge)": ridge.predict(X_shift),
    "hybrid": hybrid_predict(X_train, y_train, phys_train, X_shift, phys_shift, alpha_hybrid),
}
compare_shift = pd.DataFrame({name: scores(y_shift, p) for name, p in pred_shift.items()}).T
print("new recipe DRYOX-140 (140 min), errors in nm:")
print(compare_shift.round(3).to_string())"""),
md("""Write down the hybrid model's RMSE on the new recipe, to 2 decimals: the lab check asks for it.

**What to notice.**

- The baseline is useless: it predicts a 90-minute oxide for a 140-minute run.
- The data-driven model extends a straight line in oxidation time beyond the data. Oxide growth slows as the oxide thickens, so the line overshoots: its errors are several times larger than on the test runs, and all of the same sign.
- The hybrid model takes the time dependence from the physics, which holds at 140 minutes, and its error grows much less. Even so, it is larger than inside the range: the correction was learned for 80 to 100 minutes.

No model's error is as small here as on the test runs, and nothing in the training data could have measured how large it would be. So the twin needs to recognize such wafers."""),
md("""## Step 11: your task, `outside_range()`

A simple, honest check: a wafer is **outside the training range** if any of its features falls outside the range that feature covered in the training data. A small **margin** keeps the check from flagging every wafer that is a hair beyond the extremes the training runs happened to contain: each range is widened at both ends by 10 % of its width. For a feature whose training values run from `min` to `max`, the limits are

min − margin · (max − min)   and   max + margin · (max − min),   with margin = 0.1.

A twin can flag a wafer outside these limits and send it to the metrology tool. (Virtual-metrology systems in industry use refined versions of this idea, such as similarity indexes that measure how far a new wafer's process data are from the training data.)

**Your task:** write `outside_range(X_train, X_new, margin=0.1)`: for every row of `X_new`, True if **any** of its features is below the lower limit or above the upper limit of that feature, computed from `X_train`. A value exactly at a limit is inside. Return one True or False per row.

*Hint:* `X_train.min()` and `X_train.max()` give one value per column; comparing a table with them works column by column; `.any(axis=1)` combines the columns of each row."""),
code("""def outside_range(X_train, X_new, margin=0.1):
    \"\"\"For each row of X_new: True if any feature is outside X_train's range of that feature, widened by margin * (max - min) at both ends.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete outside_range() first.")"""),
code("""flag_test = np.asarray(outside_range(X_train, X_test), dtype=bool)
flag_shift = np.asarray(outside_range(X_train, X_shift), dtype=bool)
print(f"test runs:  {flag_test.sum()} of {len(flag_test)} wafers flagged outside the training range")
print(f"new recipe: {flag_shift.sum()} of {len(flag_shift)} wafers flagged outside the training range")
lo, hi = X_train.min(), X_train.max()
out_cols = [c for c in FEATURES if ((X_test[c] < lo[c] - 0.1 * (hi[c] - lo[c])) | (X_test[c] > hi[c] + 0.1 * (hi[c] - lo[c]))).any()]
print("features outside the limits on some test wafers:", out_cols if out_cols else "none")
print(f"oxidation time limits: {lo['ox_time_min'] - 0.1 * (hi['ox_time_min'] - lo['ox_time_min']):.0f} to "
      f"{hi['ox_time_min'] + 0.1 * (hi['ox_time_min'] - lo['ox_time_min']):.0f} min; the new recipe runs for 140 min")"""),
md("""**What to notice.** Every wafer of the new recipe is flagged, through its oxidation time. In some furnaces a test run is flagged too, when a zone temperature or the moisture falls outside what the 40 training runs contained, even with the margin; such predictions are usually still good. A range check is a warning to look, not proof of an error. And it misses problems that keep every feature in range, such as a slow thermocouple drift, which Module 5 deals with."""),
md("""## Step 12: your own furnace, and the run nobody measured

Open the **Module 4 Lab Check** in Canvas. Question 1 gives you a **KEY**, a whole number from 100 to 999. Type it below in place of `None` and run the cell. It repeats the work for your furnace with **your functions**:

1. tunes the hybrid's penalty on your 40 training runs, with the run-block folds;
2. refits the hybrid on **all 60** measured runs: once the test is done, a twin uses all the data it has;
3. predicts the five wafers of your pending run 67, checks them against the training range, and prints the prediction for the wafer in slot 65."""),
code("""KEY = None      # <- your KEY from Question 1 of the lab check, for example KEY = 417

if KEY is None:
    print("Type your KEY in place of None, then run this cell again.")
else:
    mine = vm_runs(KEY)
    m_all, m_pending = mine["measured"], mine["pending"]
    m_train, m_test = time_split(m_all)
    m_cv, m_groups = block_folds(m_train)
    m_err = m_train["thickness_nm"].to_numpy() - physics_prediction(m_train)
    m_scores = {a: cv_rmse(make_ridge(a), m_train[FEATURES], m_err, m_cv, m_groups) for a in ALPHAS}
    alpha_mine = min(m_scores, key=m_scores.get)

    pred_pending = np.asarray(hybrid_predict(m_all[FEATURES], m_all["thickness_nm"], physics_prediction(m_all),
                                             m_pending[FEATURES], physics_prediction(m_pending), alpha_mine), dtype=float)
    flags = np.asarray(outside_range(m_all[FEATURES], m_pending[FEATURES]), dtype=bool)
    print(f"KEY {KEY}: chosen alpha {alpha_mine}; pending run {m_pending['run_id'].iloc[0]}, oxidation time {m_pending['ox_time_min'].iloc[0]:.0f} min")
    for slot, p, f in zip(m_pending["slot"], pred_pending, flags):
        print(f"  slot {slot:3d}: predicted {p:.3f} nm{'   OUTSIDE the training range' if f else ''}")
    print(f"slot {VM_PENDING_SLOT} prediction: {pred_pending[list(m_pending['slot']).index(VM_PENDING_SLOT)]:.3f} nm")"""),
md("""## Step 13: check your work

The cell below tests your four functions on cases you cannot see. If all pass, it prints your **verification value**: the hybrid model's prediction for the slot-65 wafer of your pending run, in nm, computed with the course's reference functions. It should match the last number Step 12 printed. Type it into Question 1 of the lab check.

If it says **NOT PASSED**, read the message: it tells you what to fix."""),
code("""# The course's check code is in dt_m4_check.py, next to this notebook.
# It is kept out of the notebook so that the course's reference solutions are not on screen.
from dt_m4_check import check_dt_module4

passed, messages, value = check_dt_module4(baseline_recent, make_ridge, hybrid_predict, outside_range, KEY)
for line in messages:
    print(line)
if passed:
    print()
    print(f"Verification value for KEY = {KEY}: {value:.3f}")
    print("Type this number into Question 1 of the Module 4 Lab Check.")"""),
md("""## What this means for the twin

The furnace twin can now estimate every wafer's thickness, including the wafers and runs the metrology tool never measured. Inside the range it was trained on, both learned models came within about half a nanometre and beat the baseline by a factor of several; on a new recipe only the hybrid stayed close, because its time dependence comes from physics. The sampled metrology does not stop: it keeps supplying the wafers that train, test and correct the model, and the wafers the twin flags go to the metrology tool first. Module 5 uses those measurements to keep the twin in step with a furnace that drifts.

**Claim only what the data support.** On simulated data, a good score shows that the models recover the course's simulator. It does not validate a virtual-metrology model for a real furnace.

To keep a copy of this notebook, choose **File > Download**."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"m4-{i:02d}"
text = nbf.writes(nb)
assert "—" not in text, "em dash in notebook text"
(OUT / NAME).write_text(text)
print("wrote", OUT / NAME, len(cells), "cells")

# ---------------------------------------------------------------- optional Google Colab copy
COLAB = ROOT / "colab" / "m4"
COLAB.mkdir(parents=True, exist_ok=True)
CNAME = "DT-M4-lab-virtual-metrology-colab.ipynb"
RAW = "https://raw.githubusercontent.com/CSU-Physics/digital-twins-notebooks/main/content/m4/"
colab_cells = [cells[0],
               md("""## Colab only: get the course files

You are running the **Google Colab** copy of this lab. Colab needs three course files next to the notebook: `furnace.py`, the furnace simulator, `vm_tools.py`, the helpers, and `dt_m4_check.py`, the check code. Run the cell below once each time you open the notebook; it downloads them from the course repository. Everything else is the same as in the Canvas version, and your verification value is the same."""),
               code(f"""import urllib.request

BASE = "{RAW}"
for name in ["furnace.py", "vm_tools.py", "dt_m4_check.py"]:
    urllib.request.urlretrieve(BASE + name, name)
    print("downloaded", name)""")] + cells[1:]
cnb = nbf.v4.new_notebook(cells=colab_cells, metadata={"colab": {"provenance": []},
                                                       "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                       "language_info": {"name": "python"}})
for i, c in enumerate(cnb.cells):
    c["id"] = f"m4c-{i:02d}"
ctext = nbf.writes(cnb)
assert "—" not in ctext
(COLAB / CNAME).write_text(ctext)
print("wrote", COLAB / CNAME, len(colab_cells), "cells")
