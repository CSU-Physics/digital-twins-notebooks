"""Check cell logic for the Module 4 lab (Digital Twins for Semiconductor Manufacturing).

check_dt_module4(baseline_recent, make_ridge, hybrid_predict, outside_range, KEY) returns (passed, messages, verification_value).

The learner writes
  baseline_recent(table, n_runs=3)    for every wafer, the mean measured thickness of all wafers in the n_runs runs just
                                      before its run (by run_index); NaN when there are fewer than n_runs earlier runs;
  make_ridge(alpha)                   a scikit-learn pipeline: StandardScaler, then Ridge(alpha=alpha);
  hybrid_predict(X_train, y_train, phys_train, X_new, phys_new, alpha)
                                      fit make_ridge(alpha) to the physics model's errors on the training wafers,
                                      y_train - phys_train, and return phys_new plus the predicted correction;
  outside_range(X_train, X_new, margin=0.1)
                                      for every row of X_new, True if any of its features is outside the training range
                                      of that feature widened by margin * (max - min) at both ends.

The verification value is the thickness, in nm to 3 decimals, that the hybrid model predicts for the slot-65 wafer of
the learner's pending run: the ridge penalty is tuned on the 40 training runs with folds of whole runs, and the model is
then refitted on all 60 measured runs. It is computed with the course's reference functions and printed only when every
test passes.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from furnace import VM_PENDING_SLOT, vm_runs
from vm_tools import ALPHAS, FEATURES, block_folds, cv_rmse, physics_prediction, time_split


# ---------------------------------------------------------------- reference functions
def reference_baseline(table, n_runs=3):
    per_run = table.groupby("run_index")["thickness_nm"].agg(["sum", "count"]).sort_index()
    total = per_run["sum"].rolling(n_runs).sum().shift(1)
    count = per_run["count"].rolling(n_runs).sum().shift(1)
    return table["run_index"].map(total / count)


def reference_ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def reference_hybrid(X_train, y_train, phys_train, X_new, phys_new, alpha):
    model = reference_ridge(alpha).fit(X_train, np.asarray(y_train, dtype=float) - np.asarray(phys_train, dtype=float))
    return np.asarray(phys_new, dtype=float) + model.predict(X_new)


def reference_outside(X_train, X_new, margin=0.1):
    Xt, Xn = pd.DataFrame(X_train), pd.DataFrame(X_new)
    lo, hi = Xt.min(), Xt.max()
    pad = margin * (hi - lo)
    return ((Xn < lo - pad) | (Xn > hi + pad)).any(axis=1).to_numpy()


def reference_alpha_scores(train):
    cv, groups = block_folds(train)
    resid = train["thickness_nm"].to_numpy() - physics_prediction(train)
    return [cv_rmse(reference_ridge(a), train[FEATURES], resid, cv, groups) for a in ALPHAS]


def reference_alpha(train):
    s = reference_alpha_scores(train)
    return ALPHAS[int(np.argmin(s))]


def personal_raw(key):
    runs = vm_runs(key)
    measured, pending = runs["measured"], runs["pending"]
    train, _ = time_split(measured)
    alpha = reference_alpha(train)
    row = pending[pending["slot"] == VM_PENDING_SLOT]
    pred = reference_hybrid(measured[FEATURES], measured["thickness_nm"], physics_prediction(measured),
                            row[FEATURES], physics_prediction(row), alpha)
    return float(pred[0])


def personal_value(key):
    return round(personal_raw(key), 3)


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


def _values(got, index):
    """Learner output as a float array in the row order of a table with this index (a Series is aligned by index)."""
    if isinstance(got, pd.DataFrame):
        if got.shape[1] != 1:
            raise ValueError("a table with several columns")
        got = got.iloc[:, 0]
    if isinstance(got, pd.Series):
        if not got.index.equals(index):
            if set(got.index) == set(index):
                got = got.reindex(index)
            else:
                return np.asarray(got, dtype=float)
        return got.to_numpy(dtype=float)
    return np.asarray(got, dtype=float).reshape(-1)


def _same(a, b, tol=1e-6):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    both = np.isnan(a) & np.isnan(b)
    ok = np.abs(a - b) <= tol * np.maximum(1.0, np.abs(b))
    return bool(np.all(both | ok))


def _toy_table(rng, n_runs, counts=None, shuffle=False, start=1):
    rows = []
    for r in range(n_runs):
        k = counts[r] if counts is not None else 5
        level = 60.0 + 3.0 * np.sin(r / 2.0) + 0.3 * r
        for j in range(k):
            rows.append({"run_id": f"T-{start + r:02d}", "run_index": start + r, "slot": 5 + 30 * j,
                         "thickness_nm": round(level + float(rng.normal(0.0, 0.5)), 2)})
    t = pd.DataFrame(rows)
    if shuffle:
        t = t.sample(frac=1.0, random_state=int(rng.integers(1, 10 ** 6)))
    return t


# ---------------------------------------------------------------- tests
def _test_baseline(baseline_recent):
    rng = np.random.default_rng(404)
    cases = [(_toy_table(rng, 9), 3), (_toy_table(rng, 10, shuffle=True), 3), (_toy_table(rng, 8, start=41), 2),
             (_toy_table(rng, 9, counts=[5, 3, 5, 4, 5, 2, 5, 5, 1]), 3)]
    for table, n in cases:
        ok, got = _call(baseline_recent, "baseline_recent", table.copy(), n)
        if not ok:
            return False, got
        want = reference_baseline(table, n).to_numpy(dtype=float)
        if isinstance(got, pd.Series) and len(got) == len(table) and not got.index.equals(table.index):
            if _same(got.to_numpy(dtype=float), want) and not _same(got.reindex(table.index).to_numpy(dtype=float), want):
                return False, ("baseline_recent() returns a Series whose index does not match the table's rows (here the rows are "
                               "not in time order). Keep the table's index, for example with table['run_index'].map(...), "
                               "so that each value stays with its wafer.")
        try:
            got = _values(got, table.index)
        except Exception:
            return False, "baseline_recent() should return one number per row of the table: a Series or an array, not a table."
        if got.shape[0] != len(table):
            return False, f"baseline_recent() returned {got.shape[0]} values for a table of {len(table)} wafers: return one value per wafer (row)."
        if _same(got, want):
            continue
        per_run = table.groupby("run_index")["thickness_nm"]
        no_shift = table["run_index"].map(per_run.sum().sort_index().rolling(n).sum() / per_run.count().sort_index().rolling(n).sum()).to_numpy()
        if _same(got, no_shift):
            return False, ("baseline_recent() includes the wafers of the run itself. The baseline may use only runs that finished "
                           "before this one: shift the per-run values by one run, for example with .shift(1).")
        run_means = table["run_index"].map(per_run.mean().sort_index().rolling(n).mean().shift(1)).to_numpy()
        if _same(got, run_means) and not _same(run_means, want):
            return False, ("baseline_recent() averages the run means. Average all wafers of the earlier runs instead: a run with "
                           "more measured wafers counts for more (sum of thicknesses divided by the number of wafers).")
        fixed = reference_baseline(table, 3).to_numpy(dtype=float)
        if n != 3 and _same(got, fixed):
            return False, "baseline_recent() always uses 3 runs. Use the n_runs argument."
        if np.any(np.isnan(got) != np.isnan(want)):
            first = [int(r) for r in sorted(table["run_index"].unique())[:n]]
            return False, (f"baseline_recent() should give NaN for the wafers of the first {n} runs (runs {first}), which do not "
                           f"have {n} earlier runs, and a number for every later wafer.")
        sorted_t = table.sort_values(["run_index", "slot"])
        by_rows = sorted_t["thickness_nm"].rolling(5 * n).mean().shift(1).reindex(table.index).to_numpy()
        if _same(got, by_rows):
            return False, ("baseline_recent() rolls over rows (wafers). Work per run: total and count the wafers of each run, "
                           "then combine the n runs before each run.")
        if not table.index.equals(pd.RangeIndex(len(table))):
            return False, ("baseline_recent() gives wrong values when the rows are not in time order or the index does not start "
                           "at 0. Group by run_index (which sorts the runs) and map the per-run result back with "
                           "table['run_index'].map(...).")
        return False, "baseline_recent() gives wrong values. For each wafer: the mean thickness of all wafers in the n_runs runs before its run."
    return True, "baseline_recent() passed 4 tests, including rows out of time order and runs with different numbers of wafers."


def _test_ridge(make_ridge):
    rng = np.random.default_rng(405)
    X = pd.DataFrame({"a": rng.normal(90, 6, 40), "b": rng.normal(1000, 0.8, 40), "c": rng.uniform(2, 12, 40)})
    y = 0.5 * X["a"] + 0.4 * (X["b"] - 1000) + 0.2 * X["c"] + rng.normal(0, 0.3, 40)
    for alpha in (1.0, 10.0, 0.1):
        ok, model = _call(make_ridge, "make_ridge", alpha)
        if not ok:
            return False, model
        if not isinstance(model, Pipeline):
            return False, ("make_ridge() should return a Pipeline, for example make_pipeline(StandardScaler(), Ridge(alpha=alpha)), "
                           "so that the scaling is learned inside every cross-validation fold.")
        steps = [s for _, s in model.steps]
        if not any(isinstance(s, StandardScaler) for s in steps[:-1]):
            return False, "make_ridge() has no StandardScaler before the Ridge step: ridge penalizes coefficients, so the features must be on the same scale."
        if not isinstance(steps[-1], Ridge):
            return False, "The last step of make_ridge() should be Ridge(alpha=alpha)."
        if abs(float(steps[-1].alpha) - alpha) > 1e-12:
            return False, f"make_ridge({alpha}) made a Ridge with alpha = {steps[-1].alpha}: pass the alpha argument on, Ridge(alpha=alpha)."
        try:
            got = model.fit(X, y).predict(X)
        except Exception as e:
            return False, f"The pipeline from make_ridge() could not be fitted: {type(e).__name__}: {e}"
        if not _same(got, reference_ridge(alpha).fit(X, y).predict(X), 1e-8):
            return False, "The pipeline from make_ridge() predicts differently from StandardScaler followed by Ridge(alpha=alpha)."
    return True, "make_ridge() builds StandardScaler then Ridge with the requested alpha (3 tests)."


def _test_hybrid(hybrid_predict):
    rng = np.random.default_rng(406)
    for case in range(3):
        n, m = 30 + 10 * case, 12
        Xt = pd.DataFrame({"t": rng.uniform(80, 100, n), "T": rng.normal(1000, 0.8, n), "w": rng.uniform(2, 12, n)})
        Xn = pd.DataFrame({"t": rng.uniform(80, 140, m), "T": rng.normal(1000, 0.8, m), "w": rng.uniform(2, 12, m)})
        pt, pn = 0.55 * Xt["t"].to_numpy() + 5.0, 0.55 * Xn["t"].to_numpy() + 5.0
        yt = pt + 3.0 + 0.4 * (Xt["T"].to_numpy() - 1000) + 0.15 * Xt["w"].to_numpy() + rng.normal(0, 0.3, n)
        alpha = (1.0, 10.0, 0.1)[case]
        ok, got = _call(hybrid_predict, "hybrid_predict", Xt.copy(), pd.Series(yt), pt.copy(), Xn.copy(), pn.copy(), alpha)
        if not ok:
            return False, got
        try:
            got = np.asarray(got, dtype=float).reshape(-1)
        except Exception:
            return False, "hybrid_predict() should return one predicted thickness per row of X_new."
        if got.shape[0] != m:
            return False, f"hybrid_predict() returned {got.shape[0]} values for {m} new wafers: predict the rows of X_new."
        want = reference_hybrid(Xt, yt, pt, Xn, pn, alpha)
        if _same(got, want):
            continue
        corr = reference_ridge(alpha).fit(Xt, yt - pt).predict(Xn)
        plain = reference_ridge(alpha).fit(Xt, yt).predict(Xn)
        if _same(got, corr):
            return False, "hybrid_predict() returns only the learned correction: add it to the physics prediction, phys_new + correction."
        if _same(got, plain):
            return False, ("hybrid_predict() fits the measured thickness itself, which is the data-driven model. Fit the physics "
                           "model's error instead: y_train - phys_train, then return phys_new + the predicted error.")
        if _same(got, pn + plain):
            return False, "hybrid_predict() fits y_train instead of y_train - phys_train, then adds the physics again: fit the error."
        if _same(got, pn + reference_ridge(alpha).fit(Xt, pt - yt).predict(Xn)):
            return False, "hybrid_predict() fits phys_train - y_train: the error to learn is measured minus physics, y_train - phys_train."
        if _same(got, pn + reference_ridge(1.0).fit(Xt, yt - pt).predict(Xn)) and alpha != 1.0:
            return False, "hybrid_predict() always uses alpha = 1: pass the alpha argument on, make_ridge(alpha)."
        if _same(got, pn + make_pipeline(Ridge(alpha=alpha)).fit(Xt, yt - pt).predict(Xn)):
            return False, "hybrid_predict() fits Ridge without StandardScaler: use make_ridge(alpha), the pipeline with scaling."
        return False, "hybrid_predict() gives wrong values: fit make_ridge(alpha) to y_train - phys_train on X_train, then return phys_new + model.predict(X_new)."
    return True, "hybrid_predict() passed 3 tests, including new wafers far outside the training times."


def _test_outside(outside_range):
    rng = np.random.default_rng(407)
    Xt = pd.DataFrame({"t": rng.uniform(80, 100, 50), "T": rng.normal(1000, 0.8, 50), "w": rng.uniform(2, 12, 50)})
    lo, hi = Xt.min(), Xt.max()
    pad = 0.1 * (hi - lo)
    rows = [dict(Xt.iloc[k]) for k in (3, 5, 7, 9, 11, 13, 15, 17)]
    rows[1]["t"] = float(hi["t"] + pad["t"]) + 0.5                       # outside, first column only
    rows[2]["w"] = float(hi["w"] + pad["w"]) + 0.3                       # outside, last column only
    rows[3]["T"] = float(lo["T"] - pad["T"]) - 0.05                      # below the range: outside
    rows[4]["t"] = float(hi["t"] + pad["t"])                             # exactly at the upper limit: inside
    rows[5]["t"], rows[5]["w"] = float(lo["t"] - pad["t"]), float(hi["w"] + pad["w"])   # at both limits: inside
    rows[6]["t"] = float(hi["t"] + 0.5 * pad["t"])                       # beyond the largest value, within the margin: inside
    big = Xt["T"][(Xt["T"] - Xt["T"].mean()).abs() > 2 * Xt["T"].std()]
    rows[7]["T"] = float(big.iloc[0]) if len(big) else float(hi["T"])    # far from the mean, but inside the range
    Xn = pd.DataFrame(rows)[Xt.columns]
    want = reference_outside(Xt, Xn)
    ok, got = _call(outside_range, "outside_range", Xt.copy(), Xn.copy())
    if not ok:
        return False, got
    if isinstance(got, pd.DataFrame):
        return False, ("outside_range() returned a table with one column per feature. Combine the columns: a row is "
                       "outside if ANY of its features is outside, for example (...).any(axis=1).")
    try:
        got = np.asarray(got).reshape(-1).astype(bool)
    except Exception:
        return False, "outside_range() should return one True or False per row of X_new."
    if got.shape[0] != len(Xn):
        return False, f"outside_range() returned {got.shape[0]} values for {len(Xn)} rows: return one True or False per row of X_new."
    if not np.array_equal(got, want):
        if not any(got[1:4]):
            return False, ("outside_range() flags no row, but three rows have a feature outside the training range. A row is "
                           "outside if ANY of its features (not all of them) is outside the range of that feature in X_train.")
        by_value = ((Xn < lo * 0.9) | (Xn > hi * 1.1)).any(axis=1).to_numpy()
        if np.array_equal(got, by_value):
            return False, ("outside_range() widens the range by a fraction of the values themselves. The margin is a fraction of the "
                           "training range: the limits are min - margin*(max - min) and max + margin*(max - min).")
        if got[6]:
            return False, ("outside_range() ignores the margin: a value a little above the largest training value is still inside "
                           "when it is within margin * (max - min) of it. The limits are min - margin*(max - min) and max + margin*(max - min).")
        if got[7]:
            return False, ("outside_range() uses a rule other than the training range (for example the mean plus or minus two "
                           "standard deviations). Use the training minimum and maximum of each feature, widened by the margin.")
        if got[4] or got[5]:
            return False, ("outside_range() flags a row whose values lie exactly at a limit, min - margin*(max - min) or "
                           "max + margin*(max - min). Those are inside: flag only values below the lower or above the upper limit (< and >).")
        if got[1] and not got[2]:
            return False, "outside_range() checks only some of the features. A row is outside if any feature, the last one included, is outside its range."
        if got[2] and not got[1]:
            return False, "outside_range() misses a row that is outside in the first feature: check every column."
        if not got[3]:
            return False, "outside_range() misses a value below the lower limit: check both ends of each feature's range."
        return False, "outside_range() gives wrong flags: True when any feature is below min - margin*(max - min) or above max + margin*(max - min)."
    ok, got2 = _call(outside_range, "outside_range", Xt.copy(), Xn.copy(), 0.25)
    if not ok:
        return False, got2
    got2 = np.asarray(got2).reshape(-1).astype(bool)
    if not np.array_equal(got2, reference_outside(Xt, Xn, 0.25)):
        return False, "outside_range() always uses a margin of 0.1: use the margin argument."
    Xn2 = Xt.iloc[:10].copy()
    Xn2.iloc[2, 0] = 150.0
    ok, got3 = _call(outside_range, "outside_range", Xt.copy(), Xn2.copy())
    if not ok:
        return False, got3
    got3 = np.asarray(got3).reshape(-1).astype(bool)
    if not np.array_equal(got3, reference_outside(Xt, Xn2)):
        return False, "outside_range() should use the range of X_train, not of X_new: compare X_new with X_train's minimum and maximum."
    return True, "outside_range() passed 3 tests, including values exactly at the limits and two margins."


def check_dt_module4(baseline_recent, make_ridge, hybrid_predict, outside_range, KEY):
    try:
        key = int(KEY)
    except (TypeError, ValueError):
        return False, ["Set KEY to the number Canvas gives you in Question 1 of the lab check, for example KEY = 417."], None
    if not 100 <= key <= 999:
        return False, [f"KEY should be a whole number from 100 to 999, the number in Question 1 of the lab check; it is {KEY}."], None
    results = [_test_baseline(baseline_recent), _test_ridge(make_ridge), _test_hybrid(hybrid_predict), _test_outside(outside_range)]
    msgs = [("PASSED  " if ok else "NOT PASSED  ") + m for ok, m in results]
    if not all(ok for ok, _ in results):
        return False, msgs, None
    return True, msgs, personal_value(key)
