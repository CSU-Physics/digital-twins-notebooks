"""Build the Module 2 browser notebook (Digital Twins for Semiconductor Manufacturing), its Colab copy, and copy its files.

content/m2 holds furnace.py (1.2.0), secom_tools.py (supplied helpers), the SECOM data in data/ and dt_m2_check.py
(copied from checks/). The check cell only imports the check file, so the course's reference solutions are not on screen."""
import shutil
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "m2"
NAME = "DT-M2-lab-fab-data.ipynb"
VERSION = "2026-10-21"
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}

shutil.copyfile(ROOT / "checks" / "dt_m2_check.py", OUT / "dt_m2_check.py")


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 2 lab: the fab as a data system

**Digital Twins for Semiconductor Manufacturing · Module 2 · about 80 minutes** · notebook version {VERSION}

A twin is only as good as the data that reach it. This lab has two parts.

**Part A, real fab data (SECOM).** 1,567 production records from a semiconductor fab, each with 590 anonymized signals and the pass or fail result of a line test. You will:

1. see why plain accuracy misleads when only 104 of the 1,567 records are fails;
2. split the records by time **before** doing anything that learns from them;
3. write `missing_fraction()` and `keep_signals()` to drop signals that are mostly missing or nearly constant;
4. write `s2n_scores()` to rank the signals by how well they separate fails from passes, inside the training records only;
5. see how much better the results **look** when the ranking is allowed to see the test records, and why that is a mistake;
6. find out why the top signal is associated with fails without being shown to cause them.

**Part B, simulated furnace records.** The fab's records of 24 runs of the course's furnace, written by three different systems. You will write `align_runs()`, which joins the metrology results to the runs they came from, including one run with no metrology and one whose metrology arrived late, and then work with **your own** records, set by your KEY.

The **Module 2 Lab Check** in Canvas asks about your results, so keep this lab open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

The first time, Python can take up to a minute to start, and scikit-learn, the machine-learning library used in Part A, adds several seconds. The cell prints the versions of the libraries and of the two course files next to this notebook: `furnace.py`, the furnace simulator, and `secom_tools.py`, the helpers for Part A."""),
code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import furnace, secom_tools
from secom_tools import load_secom, time_split, evaluate, screening_pipeline
from furnace import fab_records, ON_TIME_H

print("Notebook version {VERSION}")
print("NumPy", np.__version__, "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__,
      "| furnace.py", furnace.__version__, "| secom_tools.py", secom_tools.__version__)"""),
md("""# Part A: SECOM, real data from a fab

## Step 1: load the records

SECOM was donated to the UCI Machine Learning Repository in 2008 by Michael McCann and Adrian Johnston (https://doi.org/10.24432/C54305, license CC BY 4.0). Each row is one **production record**: the values of 590 signals collected from sensors and process measurement points, and the result of an in-house line test, pass or fail. The signals are anonymized: the data set does not say which tool, sensor or wafer a value comes from.

`load_secom()` returns three things with the same row numbers, 0 to 1566, in time order:

- `X`, a table of the 590 signals, named `s000` to `s589` after their column in the file, with NaN where a value is missing;
- `y`, the label: 1 for a record that **failed** the line test, 0 for a pass;
- `t`, the date and time of the test."""),
code("""X, y, t = load_secom()
print("records:", X.shape[0], "| signals:", X.shape[1], "| fails:", int(y.sum()), "| passes:", int((y == 0).sum()))
print("from", t.min(), "to", t.max())
X.iloc[:5, :8]"""),
md("""## Step 2: the accuracy trap

Suppose a model ignores every signal and predicts **pass** for every record. Its **accuracy**, the share of records it gets right, is the share of passes:

accuracy = 1463 / 1567 = 0.934

That sounds good, and the model finds none of the 104 fails, which are the records a fab cares about. When one class is rare, report numbers that look at each class:

- the **share of fails found** (the recall of fails): 0 of 104 here;
- the **balanced accuracy**, the mean of the share of passes called correctly and the share of fails called correctly: (1 + 0) / 2 = 0.5 here, the same as a coin toss;
- the **AUC**, the area under the ROC curve, which says how well a score puts fails above passes: 0.5 is chance, 1 is perfect.

The cell computes the first two for the always-pass model."""),
code("""always_pass = np.zeros(len(y), dtype=int)          # 0 = pass for every record
accuracy = np.mean(always_pass == y)
found = int(((always_pass == 1) & (y == 1)).sum())
pass_right = np.mean(always_pass[y == 0] == 0)
fail_right = np.mean(always_pass[y == 1] == 1)
print(f"always pass: accuracy {accuracy:.3f}, fails found {found} of {int(y.sum())}, balanced accuracy {(pass_right + fail_right) / 2:.3f}")"""),
md("""## Step 3: split by time, first

A model is useful only if it works on records it has not seen. So the records are split **before** anything learns from them:

- the **training records** are used to choose signals and fit models;
- the **test records** are used once, at the end, to measure the result.

SECOM is in time order, from July to October 2008, and fabs change over time: tools drift, recipes are adjusted, products change. A model in use would be applied to records that come **after** the ones it learned from. So `time_split()` keeps the earliest 70% of the records for training and the latest 30% for testing, instead of picking test records at random. The cell also shows how the share of fails changed from month to month."""),
code("""X_train, X_test, y_train, y_test = time_split(X, y, t)
print(f"training: {len(X_train)} records up to {t.iloc[len(X_train) - 1]}, {int(y_train.sum())} fails ({y_train.mean():.1%})")
print(f"test:     {len(X_test)} records from {t.iloc[len(X_train)]}, {int(y_test.sum())} fails ({y_test.mean():.1%})")
print()
print(pd.DataFrame({"records": y.groupby(t.dt.to_period("M")).size(),
                    "fails": y.groupby(t.dt.to_period("M")).sum(),
                    "share of fails": y.groupby(t.dt.to_period("M")).mean().round(3)}))"""),
md("""**What to notice.** In July 22% of the records failed, in September 3%. Whatever changed in the fab also changed the signals, so a random split would mix July and October records in both sets and hide the change. From here on, everything that **learns** from the data uses `X_train` and `y_train` only."""),
md("""## Step 4: your task, `missing_fraction()`

Real fab data have holes: a sensor was off, a value was not logged, a measurement was skipped. Write `missing_fraction(X)`: for each column of `X`, the number of missing values (NaN) divided by the number of rows. It returns a pandas Series with one number per column.

Hint: `X.isna()` is a table of True and False, and the mean of True and False is the share of True."""),
code("""def missing_fraction(X):
    \"\"\"The fraction of missing values (NaN) in each column of X, as a pandas Series labeled by column name.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete missing_fraction() first.")"""),
code("""miss = missing_fraction(X_train)
print(f"share of all training values that are missing: {X_train.isna().to_numpy().mean():.3f}")
print("signals with no missing values:", int((miss == 0).sum()))
print("signals with more than half missing:", int((miss > 0.5).sum()))

fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(miss, bins=50)
ax.set_xlabel("fraction missing (training records)")
ax.set_ylabel("number of signals")
plt.show()"""),
md("""## Step 5: your task, `keep_signals()`

Two kinds of signal carry little information:

- a signal with **most of its values missing**: whatever is used to fill the holes would be mostly invented;
- a **near-constant** signal, where one value makes up almost all the values that are present: it cannot tell fails from passes.

Write `keep_signals(X_train, max_missing=0.5, max_share=0.95)`. It returns a **list of the names** of the columns to keep. Keep a column when both rules allow it:

1. its missing fraction is **at most** `max_missing`;
2. the most common of its **present** values makes up **less than** `max_share` of the present values.

Hints: `col.dropna()` gives the present values. `present.value_counts(normalize=True)` gives the share of each value, largest first, so `.iloc[0]` is the share of the most common one. A column with no present values at all is dropped by rule 1. Use the arguments `max_missing` and `max_share`, not fixed numbers: the check tries other settings."""),
code("""def keep_signals(X_train, max_missing=0.5, max_share=0.95):
    \"\"\"Names of the columns that are at most max_missing missing and not near-constant (top share below max_share).\"\"\"
    keep = []
    for name in X_train.columns:
        col = X_train[name]
        present = col.dropna()
        # YOUR CODE HERE: append name to keep when both rules allow it
        raise NotImplementedError("Complete keep_signals() first.")
    return keep"""),
code("""kept = keep_signals(X_train)
print(f"signals kept: {len(kept)} of {X_train.shape[1]}")"""),
md("""Write down the number of signals kept: the lab check asks for it.

Both rules were applied to the **training** records only. That matters even here: some signals are constant in the training records and change only later, in the test period. A rule that looked at all 1,567 records would make a different choice, using information from the test period."""),
md("""## Step 6: your task, `s2n_scores()`

Which signals separate fails from passes? One simple score, the one the data set's donors used in their own baseline, is the **signal-to-noise ratio** of each signal:

S2N = |mean of the fails − mean of the passes| / (standard deviation of the fails + standard deviation of the passes)

A large S2N means the fails sit far from the passes compared with the spread of each group. Write `s2n_scores(X, y)`: one S2N per column of `X`, using the labels `y` (1 = fail, 0 = pass). Skip missing values, and use the sample standard deviation, which is what pandas' `.mean()` and `.std()` do by default.

Hint: `X[y == 1]` is the table of the fail rows."""),
code("""def s2n_scores(X, y):
    \"\"\"For each column of X: |mean of fails - mean of passes| / (std of fails + std of passes), skipping NaN.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete s2n_scores() first.")"""),
code("""scores = s2n_scores(X_train[kept], y_train).sort_values(ascending=False)
print("the ten highest scores in the training records:")
print(scores.head(10).round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.arange(1, len(scores) + 1), scores.values, ".", ms=3)
ax.set_xlabel("rank")
ax.set_ylabel("S2N (training records)")
plt.show()"""),
md("""Write down the score of the top signal to 3 decimals: the lab check asks for it.

**What to notice.** No signal separates the classes cleanly: even the best score is well below 1, so the two groups overlap a lot. Most scores form a smooth curve, which is what many signals with no real link to the label would also give."""),
md("""## Step 7: does the ranking predict new records?

`evaluate()` takes the chosen signals, fits a simple classifier (fill in missing values with the training median, scale, then a logistic regression that weighs fails and passes equally) on the **training** records, and scores the **test** records. It returns the AUC, the balanced accuracy, the accuracy and how many test fails it flagged. Every step inside it learns from the training records only.

The cell uses the 20 signals with the highest S2N in the training records, and compares with the always-pass baseline from Step 2."""),
code("""top20 = list(scores.index[:20])
honest = evaluate(X_train, y_train, X_test, y_test, top20)
print(f"always pass:            AUC 0.500, balanced accuracy 0.500, accuracy {np.mean(y_test == 0):.3f}, fails found 0 of {int(y_test.sum())}")
print(f"top 20, training only:  AUC {honest['auc']:.3f}, balanced accuracy {honest['balanced_accuracy']:.3f}, "
      f"accuracy {honest['accuracy']:.3f}, fails found {honest['fails_found']} of {honest['fails']}")"""),
md("""**What to notice.** The classifier finds a few of the test fails and its AUC is a little above 0.5, while its accuracy is **lower** than the always-pass baseline's, because it raises some false alarms. That is a modest result, and a modest result measured honestly is a valid finding: with these signals and this many fails, the later records are hard to predict."""),
md("""## Step 8: what leakage looks like

Now make the mistake on purpose. Rank the signals with **all 1,567 records**, test records included, then fit and score exactly as before. Nothing else changes: the same classifier, the same training records for fitting, the same test records for scoring."""),
code("""scores_all = s2n_scores(X[kept], y).sort_values(ascending=False)
top20_all = list(scores_all.index[:20])
leaky = evaluate(X_train, y_train, X_test, y_test, top20_all)
print(f"signals in both top-20 lists: {len(set(top20) & set(top20_all))}")
print(f"ranked on training records only:  AUC {honest['auc']:.3f}, fails found {honest['fails_found']} of {honest['fails']}")
print(f"ranked on all records (leakage):  AUC {leaky['auc']:.3f}, fails found {leaky['fails_found']} of {leaky['fails']}")"""),
md("""**What to notice.** The leaky version looks much better, and none of that improvement is real. The ranking saw the test labels, so it picked signals that happen to separate the fails **of the test period**. On truly new records, which nobody can rank in advance, it would do no better than the honest version. This is **data leakage**: information that would not be available at prediction time was used to build the model.

To see that leakage can create a result from nothing, the cell below **shuffles** the labels, so that no signal has any real link to them, and repeats both versions five times. With shuffled labels the expected AUC is 0.5."""),
code("""rng = np.random.default_rng(2026)
for i in range(5):
    y_shuf = pd.Series(rng.permutation(y.to_numpy()), index=y.index, name="fail")
    y_shuf_train, y_shuf_test = y_shuf.iloc[:len(X_train)], y_shuf.iloc[len(X_train):]
    pick_honest = s2n_scores(X_train[kept], y_shuf_train).nlargest(20).index
    pick_leaky = s2n_scores(X[kept], y_shuf).nlargest(20).index
    a = evaluate(X_train, y_shuf_train, X_test, y_shuf_test, pick_honest)["auc"]
    b = evaluate(X_train, y_shuf_train, X_test, y_shuf_test, pick_leaky)["auc"]
    print(f"shuffle {i + 1}: AUC ranked on training only {a:.3f}, ranked on all records {b:.3f}")"""),
md("""**What to notice.** With labels that mean nothing, the honest version scores around 0.5, as it should, and the leaky version scores well above 0.5 every time. A score above chance proves nothing unless every choice was made without the test records."""),
md("""## Step 9: the same rule inside cross-validation

Cross-validation splits the training records into folds, fits on all but one fold and scores the one left out, in turn. It is how models are compared and tuned **without touching the test records**. The rule from Step 8 applies inside every fold: the ranking must be learned from the fitting folds only.

`screening_pipeline(s2n_scores, k=20)` builds a scikit-learn **pipeline** that fills in missing values, scales, keeps the 20 signals with the highest score from **your** `s2n_scores()`, and fits the classifier. Because the ranking is a step of the pipeline, scikit-learn repeats it inside every fold, on that fold's fitting records only.

The cell compares two ways of forming five folds: at random, and in time order (each fold is scored with a model fitted on the records before it). It takes several seconds."""),
code("""from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score

pipe = screening_pipeline(s2n_scores, k=20)
random_folds = cross_val_score(pipe, X_train[kept], y_train, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
time_folds = cross_val_score(pipe, X_train[kept], y_train, cv=TimeSeriesSplit(5), scoring="roc_auc")
print("random folds:", np.round(random_folds, 3), f"mean AUC {random_folds.mean():.3f}")
print("time folds:  ", np.round(time_folds, 3), f"mean AUC {time_folds.mean():.3f}")
print(f"test records (Step 7): AUC {honest['auc']:.3f}")"""),
md("""**What to notice.** Random folds give a much more hopeful estimate than the test records do, because each random fold shares months, and the changes of those months, with the records the model was fitted on. Folds in time order are closer to what happened on the later records. With 78 training fails the fold scores also spread widely: that spread is part of the result, and it is why a single number should never be reported alone."""),
md("""## Step 10: associated is not shown to be a cause

The top signal in the training records is `s059`. The cell shows its monthly mean next to the monthly share of fails, and its S2N in the training and the test records."""),
code("""month = t.dt.to_period("M")
print(pd.DataFrame({"mean of s059": X["s059"].groupby(month).mean().round(2), "share of fails": y.groupby(month).mean().round(3)}))
print()
print(f"S2N of s059: training records {s2n_scores(X_train[['s059']], y_train).iloc[0]:.3f}, "
      f"test records {s2n_scores(X_test[['s059']], y_test).iloc[0]:.3f}")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(t[y == 0], X["s059"][y == 0], ".", ms=3, alpha=0.5, label="pass")
ax.plot(t[y == 1], X["s059"][y == 1], "x", ms=5, color="C3", label="fail")
ax.axvline(t.iloc[len(X_train)], color="gray", ls="--", lw=1)
ax.set_ylabel("s059")
ax.legend()
plt.show()"""),
md("""**What to notice.** `s059` and the share of fails both fall from July to October. In the training records fails sit at high `s059` values largely because fails were common in the summer, when `s059` was high. In the test records the separation almost disappears. Something that changed over time moved both, so the association in the training records does not show that `s059` causes fails, or even that it will flag fails next month.

An association found by screening is a lead for an engineer to investigate with process knowledge and, ideally, a designed experiment. It is not a cause. And because SECOM's signals are anonymized, with no link to a tool or a wafer, an engineer could not even look up what `s059` measures."""),
md("""# Part B: the furnace's records, from three systems

## Step 11: four tables, three systems

A fab does not keep one table per run. Each system writes its own records, and a twin has to join them. `fab_records(key)` returns the simulated records of 24 runs of the course's furnace (recipe DRYOX-60: 60 minutes in dry oxygen at 1000 °C), as four tables:

| Table | Written by | One row per | Columns |
|---|---|---|---|
| `runs` | the MES (manufacturing execution system) | run | run_id, tool, recipe, start_time, hold_start, end_time |
| `wafers` | the MES | monitor wafer | run_id, slot, wafer_id |
| `trace` | the furnace controller | reading, every 10 s of the hold | run_id, time_s, zone_temp_C |
| `metrology` | the ellipsometer that measures the oxide | measured wafer | wafer_id, measured_at, thickness_nm |

Each run carries three **monitor wafers**, in slots 5, 25 and 45 of the boat, and the ellipsometer measures their oxide thickness after the run. The ellipsometer knows the **wafer**, not the run: to find the run of a result, look the wafer up in `wafers`. The cell loads the example records (key 2026)."""),
code("""rec = fab_records(2026)
runs, wafers, trace, metrology = rec["runs"], rec["wafers"], rec["trace"], rec["metrology"]
for name, table in rec.items():
    print(f"{name}: {len(table)} rows")
print()
print(runs.head(3).to_string(index=False))
print()
print(wafers.head(3).to_string(index=False))
print()
print(metrology.head(3).to_string(index=False))"""),
md("""## Step 12: summarize each trace, and line up the clocks

The controller logs the zone temperature every 10 s of the hold. A twin rarely needs every reading; it uses **summary indicators** over a step of the recipe, such as the mean, the standard deviation or the slope over the hold. The cell computes the **mean hold temperature** of each run with `groupby`, as in Module 0.

The trace and the MES also count time differently: `time_s` counts seconds from the start of the hold, while the MES writes clock times. To put a reading on the MES clock, add its `time_s` to the run's `hold_start`. The cell does that for the last reading of the first run."""),
code("""hold_mean = trace.groupby("run_id")["zone_temp_C"].mean()
print(hold_mean.head().round(3).to_string())
print()
first = runs.iloc[0]
last_reading = trace[trace["run_id"] == first["run_id"]].iloc[-1]
clock = first["hold_start"] + pd.Timedelta(seconds=int(last_reading["time_s"]))
print(f"run {first['run_id']}: hold starts {first['hold_start']}, last reading at time_s = {int(last_reading['time_s'])} s, "
      f"which is {clock} on the MES clock")"""),
md("""## Step 13: your task, `align_runs()`

Write `align_runs(runs, wafers, metrology)`. It returns a DataFrame with **one row per run of `runs`**, and these columns:

- `run_id`;
- `thickness_nm`: the average thickness of the run's measured wafers;
- `delay_h`: the hours from the run's `end_time` to the time of its **last** metrology record, the moment the run's result is complete.

Steps:

1. Join `metrology` to `wafers` on `wafer_id`, so that each result gets its `run_id`.
2. Group by `run_id`: the mean of `thickness_nm` and the latest `measured_at`.
3. Join that onto `runs` with `how="left"`, so that **every** run stays, including a run with no metrology. Its thickness and delay stay NaN: they are unknown, not zero.
4. `delay_h` = (latest `measured_at` − `end_time`) / `pd.Timedelta(hours=1)`.

Hints: `a.merge(b, on="wafer_id")` joins two tables; `.groupby("run_id").agg(thickness_nm=("thickness_nm", "mean"), last=("measured_at", "max")).reset_index()` does step 2."""),
code("""def align_runs(runs, wafers, metrology):
    \"\"\"One row per run: run_id, thickness_nm (mean of the run's measured wafers) and delay_h (hours from end_time
    to the run's last metrology record). NaN for a run with no metrology.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete align_runs() first.")"""),
code("""table = align_runs(runs, wafers, metrology)
table["hold_mean_C"] = table["run_id"].map(hold_mean)
print(len(table), "rows (there are", len(runs), "runs)")
print(table[["run_id", "hold_mean_C", "thickness_nm", "delay_h"]].round(3).to_string(index=False))"""),
md("""## Step 14: which results can the twin use?

A twin that updates its estimate of the furnace before the next lots are dispatched can use a run's result only if it arrives in time. In this course that limit is `ON_TIME_H` = 12 hours. Each run is then one of three kinds:

- **on time**: the result arrived within 12 hours;
- **late**: the result exists but arrived after 12 hours, too late for the decision it was meant for (it can still be used later, to refit the model);
- **missing**: no result reached the database.

The cell labels the runs, plots the mean hold temperature against the average thickness for the on-time runs, and computes their **correlation**, a number from −1 to 1 that says how closely the two move together along a straight line."""),
code("""def kind(row):
    if np.isnan(row["delay_h"]):
        return "missing"
    return "on time" if row["delay_h"] <= ON_TIME_H else "late"

table["kind"] = table.apply(kind, axis=1)
print(table["kind"].value_counts().to_string())
print(table.loc[table["kind"] != "on time", ["run_id", "thickness_nm", "delay_h", "kind"]].round(2).to_string(index=False))

on_time = table[table["kind"] == "on time"]
r = np.corrcoef(on_time["hold_mean_C"], on_time["thickness_nm"])[0, 1]
print(f"correlation over the {len(on_time)} on-time runs: {r:.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(on_time["hold_mean_C"], on_time["thickness_nm"], "o", label="on time")
late_rows = table[table["kind"] == "late"]
ax.plot(late_rows["hold_mean_C"], late_rows["thickness_nm"], "s", mfc="none", label="late")
ax.set_xlabel("mean hold temperature (°C)")
ax.set_ylabel("average thickness (nm)")
ax.legend()
plt.show()"""),
md("""**What to notice.** Runs that held a little hotter grew a little more oxide: a few tenths of a nanometre more per degree, on oxide about 40 nm thick. Here the link is a cause, because the simulator is built that way and the physics of oxidation (Module 3) says so; the data alone, as in Part A, would only show an association.

The run with no metrology still has its trace, and the twin still knows its hold temperature. Dropping it from the table with an inner join would hide the fact that a result is missing, which is itself something the fab should notice and follow up."""),
md("""## Step 15: your own records

Open the **Module 2 Lab Check** in Canvas. Question 1 gives you a **KEY**, a whole number from 100 to 999. Type it below in place of `None` and run the cell. It repeats Steps 11 to 14 for **your** records."""),
code("""KEY = None      # <- your KEY from Question 1 of the lab check, for example KEY = 417

mine = fab_records(KEY)
my_table = align_runs(mine["runs"], mine["wafers"], mine["metrology"])
my_table["hold_mean_C"] = my_table["run_id"].map(mine["trace"].groupby("run_id")["zone_temp_C"].mean())
my_table["kind"] = my_table.apply(kind, axis=1)
print(my_table.loc[my_table["kind"] != "on time", ["run_id", "thickness_nm", "delay_h", "kind"]].round(2).to_string(index=False))
my_on_time = my_table[my_table["kind"] == "on time"]
print(f"correlation over your {len(my_on_time)} on-time runs: "
      f"{np.corrcoef(my_on_time['hold_mean_C'], my_on_time['thickness_nm'])[0, 1]:.3f}")"""),
md("""## Step 16: check your work

The cell below tests your four functions on cases you cannot see: small made-up tables for Part A, and the records of other KEYs, a wafer that was never measured and records written out of order for Part B. If every test passes, it prints your **verification value**: the correlation between the mean hold temperature and the average thickness over the on-time runs of **your** records, computed with the course's reference functions. It should match the number Step 15 printed. Type it into Question 1 of the lab check.

If it says **NOT PASSED**, read the message: it tells you what to fix."""),
code("""# The course's check code is in dt_m2_check.py, next to this notebook.
# It is kept out of the notebook so that the course's reference solutions are not on screen.
from dt_m2_check import check_dt_module2

passed, messages, value = check_dt_module2(missing_fraction, keep_signals, s2n_scores, align_runs, KEY)
for line in messages:
    print(line)
if passed:
    print()
    print(f"Verification value for KEY = {KEY}: {value:.3f}")
    print("Type this number into Question 1 of the Module 2 Lab Check.")"""),
md("""## What this means for a twin

- **Part A:** most of what a fab records says little about a given outcome, and a ranking of signals is only trustworthy if it was made without the records used to judge it. Report results that look at the rare class, compare with a baseline, and keep time order when the process changes.
- **Part B:** before a twin can learn from a result, the result has to be joined to the right run, and some results arrive late or never. A twin needs rules for both: which results it may use, and what it does when one is missing.

To keep a copy of this notebook, choose **File > Download**."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"m2-{i:02d}"
text = nbf.writes(nb)
assert "—" not in text, "em dash in notebook text"
(OUT / NAME).write_text(text)
print("wrote", OUT / NAME, len(cells), "cells")

# ---------------------------------------------------------------- optional Google Colab copy
COLAB = ROOT / "colab" / "m2"
COLAB.mkdir(parents=True, exist_ok=True)
CNAME = "DT-M2-lab-fab-data-colab.ipynb"
RAW = "https://raw.githubusercontent.com/CSU-Physics/digital-twins-notebooks/main/content/m2/"
colab_cells = [cells[0],
               md("""## Colab only: get the course files

You are running the **Google Colab** copy of this lab. Colab needs the course files next to the notebook: `furnace.py`, the furnace simulator, `secom_tools.py`, the helpers for Part A, `dt_m2_check.py`, the check code, and the two SECOM data files in a folder named `data`. Run the cell below once each time you open the notebook; it downloads them from the course repository (about 2 MB). Everything else is the same as in the Canvas version, and your verification value is the same."""),
               code(f"""import os, urllib.request

BASE = "{RAW}"
os.makedirs("data", exist_ok=True)
for name in ["furnace.py", "secom_tools.py", "dt_m2_check.py", "data/secom.data.gz", "data/secom_labels.data"]:
    urllib.request.urlretrieve(BASE + name, name)
    print("downloaded", name)""")] + cells[1:]
cnb = nbf.v4.new_notebook(cells=colab_cells, metadata={"colab": {"provenance": []},
                                                       "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                       "language_info": {"name": "python"}})
for i, c in enumerate(cnb.cells):
    c["id"] = f"m2c-{i:02d}"
ctext = nbf.writes(cnb)
assert "—" not in ctext
(COLAB / CNAME).write_text(ctext)
print("wrote", COLAB / CNAME, len(colab_cells), "cells")
