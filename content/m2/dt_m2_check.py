"""Check cell logic for the Module 2 lab (Digital Twins for Semiconductor Manufacturing).

check_dt_module2(missing_fraction, keep_signals, s2n_scores, align_runs, KEY) returns (passed, messages, verification_value).

Part A (SECOM). The learner writes
  missing_fraction(X)                          the fraction of missing values in each column of X;
  keep_signals(X_train, max_missing, max_share)  the names of the columns to keep: at most max_missing of the values
                                               missing, and no single value making up max_share or more of the
                                               values that are present (a near-constant signal);
  s2n_scores(X, y)                             for each column, |mean of fails - mean of passes| divided by
                                               (sample standard deviation of fails + that of passes), missing values skipped.
Part B (simulated). The learner writes
  align_runs(runs, wafers, metrology)          one row per run of the MES run history, with the average thickness of the
                                               run's monitor wafers (thickness_nm) and the hours from the end of the run to its
                                               last metrology record (delay_h); NaN for a run with no metrology.

The hidden tests use small made-up tables and the simulated records of other KEYs, never the learner's own answers.
The verification value is the correlation between the mean hold temperature and the average thickness over the runs
of the learner's own record set, fab_records(KEY), whose metrology arrived within 12 hours, to 3 decimals. It is
computed with the course's reference functions and printed only when every test passes.
"""
import math

import numpy as np
import pandas as pd

from furnace import ON_TIME_H, fab_records

MAX_MISSING = 0.5
MAX_SHARE = 0.95


# ---------------------------------------------------------------- reference functions
def reference_missing_fraction(X):
    return X.isna().mean()


def _top_share(col):
    present = col.dropna()
    if present.empty:
        return 1.0
    return float(present.value_counts(normalize=True).iloc[0])


def reference_keep_signals(X_train, max_missing=MAX_MISSING, max_share=MAX_SHARE):
    miss = X_train.isna().mean()
    return [c for c in X_train.columns if miss[c] <= max_missing and _top_share(X_train[c]) < max_share]


def reference_s2n(X, y):
    y = pd.Series(np.asarray(y), index=X.index)
    fail, ok = X[y == 1], X[y == 0]
    return (fail.mean() - ok.mean()).abs() / (fail.std() + ok.std())


def reference_align(runs, wafers, metrology):
    m = metrology.merge(wafers[["run_id", "wafer_id"]], on="wafer_id", how="inner")
    g = m.groupby("run_id").agg(thickness_nm=("thickness_nm", "mean"), n_measured=("thickness_nm", "size"),
                                last_measured_at=("measured_at", "max"))
    out = runs[["run_id", "end_time"]].merge(g, left_on="run_id", right_index=True, how="left")
    out["n_measured"] = out["n_measured"].fillna(0).astype(int)
    out["delay_h"] = (out["last_measured_at"] - out["end_time"]) / pd.Timedelta(hours=1)
    return out.reset_index(drop=True)


def hold_means(trace):
    return trace.groupby("run_id")["zone_temp_C"].mean()


def personal_raw(key):
    rec = fab_records(key)
    table = reference_align(rec["runs"], rec["wafers"], rec["metrology"])
    table["hold_mean_C"] = table["run_id"].map(hold_means(rec["trace"]))
    on_time = table[table["delay_h"] <= ON_TIME_H]
    return float(np.corrcoef(on_time["hold_mean_C"], on_time["thickness_nm"])[0, 1])


def personal_value(key):
    return round(personal_raw(key), 3)


# ---------------------------------------------------------------- helpers
def _call(fn, name, *args, **kw):
    try:
        got = fn(*args, **kw)
        if got is None:
            return False, f"{name}() returned nothing (None): end it with a return line that gives back the result."
        return True, got
    except NotImplementedError:
        return False, f"{name}() is not written yet: replace the raise line with your code."
    except Exception as e:
        return False, f"{name}() stopped with {type(e).__name__}: {e}"


def _as_series(got, columns, name):
    if isinstance(got, pd.DataFrame) and got.shape[1] == 1:
        got = got.iloc[:, 0]
    if isinstance(got, pd.Series):
        if set(got.index) == set(columns):
            return got.reindex(columns).astype(float), None
        if len(got) == len(columns) and isinstance(got.index, pd.RangeIndex):
            return pd.Series(np.asarray(got, dtype=float), index=columns), None
        return None, (f"{name}() should return one value per column, labeled by the column names, "
                      f"but it returned {len(got)} values. Work down the columns, for example with X.mean() (axis 0).")
    try:
        arr = np.asarray(got, dtype=float)
    except Exception:
        return None, f"{name}() should return a pandas Series with one number per column."
    if arr.ndim != 1 or arr.size != len(columns):
        return None, f"{name}() should return one number per column ({len(columns)} here), not {arr.size}."
    return pd.Series(arr, index=columns), None


def _same(a, b, tol=1e-9):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore"):
        close = np.abs(a - b) <= tol * np.maximum(1.0, np.abs(b))
    return bool(np.all(both_nan | close))


# ---------------------------------------------------------------- tests, part A
def _frame_missing(seed):
    rng = np.random.default_rng(seed)
    n = 40
    X = pd.DataFrame({c: rng.normal(5.0, 2.0, n) for c in ["a", "b", "c", "d", "e", "f"]})
    for c, k in zip("abcdef", [0, 1, 7, 20, 33, 40]):
        X.loc[rng.choice(n, size=k, replace=False), c] = np.nan
    return X


def _test_missing(missing_fraction):
    for seed in (11, 12):
        X = _frame_missing(seed)
        ok, got = _call(missing_fraction, "missing_fraction", X.copy())
        if not ok:
            return False, got
        got, err = _as_series(got, list(X.columns), "missing_fraction")
        if err:
            return False, err
        want = reference_missing_fraction(X)
        if _same(got, want):
            continue
        if _same(got, X.isna().sum()):
            return False, "missing_fraction() returns the number of missing values: divide by the number of rows to get a fraction."
        if _same(got, 1 - want):
            return False, "missing_fraction() returns the fraction of values that are present: it should be the fraction that are missing (NaN)."
        return False, "missing_fraction() gives the wrong fractions: for each column, the number of NaN values divided by the number of rows."
    return True, "missing_fraction() passed its tests, including a column with no missing values and one with only missing values."


def _frame_keep(seed):
    rng = np.random.default_rng(seed)
    n = 100
    X = pd.DataFrame({
        "ok_10": rng.normal(0, 1, n), "ok_45": rng.normal(10, 3, n), "miss_60": rng.normal(0, 1, n),
        "all_nan": np.full(n, np.nan), "constant": np.full(n, 7.25), "share_97": np.r_[np.full(97, 2.0), rng.normal(2, 1, 3)],
        "share_90": np.r_[np.full(90, 0.0), rng.normal(0, 1, 10)], "nan_const": np.r_[np.full(60, 1.5), np.full(40, np.nan)],
        "tiny_var": 3.0 + rng.normal(0, 1e-7, n)})
    for c, k in [("ok_10", 10), ("ok_45", 45), ("miss_60", 60), ("share_90", 4)]:
        X.loc[rng.choice(n, size=k, replace=False), c] = np.nan
    return X.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _test_keep(keep_signals):
    for seed, kw in [(21, {}), (22, {"max_missing": 0.3, "max_share": 0.85}), (23, {"max_missing": 0.7, "max_share": 0.99})]:
        X = _frame_keep(seed)
        mm, ms = kw.get("max_missing", MAX_MISSING), kw.get("max_share", MAX_SHARE)
        ok, got = _call(keep_signals, "keep_signals", X.copy(), **kw) if kw else _call(keep_signals, "keep_signals", X.copy())
        if not ok:
            return False, got
        if isinstance(got, (pd.Series, np.ndarray)) and np.asarray(got).dtype == bool:
            return False, "keep_signals() returns True and False values: return the names of the columns to keep, for example list(X_train.columns[keep])."
        if isinstance(got, pd.DataFrame):
            return False, "keep_signals() returns a table: return only the names of the columns to keep, for example list(X_train.columns[keep])."
        try:
            got = list(got)
        except Exception:
            return False, "keep_signals() should return a list of column names."
        if len(got) != len(set(got)):
            return False, "keep_signals() lists a column more than once."
        want = reference_keep_signals(X, mm, ms)
        if set(got) == set(want):
            continue
        extra, missing = set(got) - set(want), set(want) - set(got)
        if set(got) == set(X.columns) - set(want):
            return False, "keep_signals() returns the columns it should drop: return the ones that pass both rules."
        if kw and set(got) == set(reference_keep_signals(X)):
            return False, "keep_signals() ignores its max_missing and max_share arguments: use them instead of fixed numbers."
        if "nan_const" in extra:
            return False, ("keep_signals() keeps a signal whose values that are present are all the same. Work out the share of the most "
                           "common value among the values that are present only: value_counts(normalize=True) leaves NaN out by default.")
        if "tiny_var" in missing:
            return False, ("keep_signals() drops a signal whose values are all different but very close together. The rule here is about the "
                           "share of the most common value, not about the size of the standard deviation.")
        if extra & {"miss_60", "all_nan"} or missing & {"ok_45"}:
            return False, f"keep_signals() applies the missing-value rule wrongly: keep a column when its missing fraction is at most max_missing ({mm})."
        if extra & {"constant", "share_97"} or missing & {"share_90"}:
            return False, (f"keep_signals() applies the near-constant rule wrongly: drop a column when one value makes up max_share ({ms}) "
                           "or more of its values that are present.")
        return False, "keep_signals() keeps the wrong columns: check both rules."
    return True, "keep_signals() passed its tests with three settings of max_missing and max_share."


def _frame_s2n(seed):
    rng = np.random.default_rng(seed)
    n = 80
    y = pd.Series((rng.random(n) < 0.2).astype(int), name="fail")
    y.iloc[:3] = 1
    X = pd.DataFrame({f"x{j}": rng.normal(rng.normal(0, 3), rng.uniform(0.5, 4), n) + y * rng.normal(0, 2) for j in range(6)})
    for c in X.columns:
        X.loc[rng.choice(n, size=int(rng.integers(0, 12)), replace=False), c] = np.nan
    idx = np.arange(500, 500 + n)            # like the test records, the index does not start at 0
    X.index, y.index = idx, idx
    return X, y


def _test_s2n(s2n_scores):
    for seed in (31, 32, 33):
        X, y = _frame_s2n(seed)
        ok, got = _call(s2n_scores, "s2n_scores", X.copy(), y.copy())
        if not ok:
            return False, got
        got, err = _as_series(got, list(X.columns), "s2n_scores")
        if err:
            return False, err
        want = reference_s2n(X, y)
        if _same(got, want):
            continue
        f, p = X[y == 1], X[y == 0]
        if _same(got, (f.mean() - p.mean()) / (f.std() + p.std())) or _same(got, (p.mean() - f.mean()) / (f.std() + p.std())):
            return False, "s2n_scores() can be negative: take the absolute value of the difference of the means, with .abs()."
        if _same(got, (f.mean() - p.mean()).abs() / (f.std(ddof=0) + p.std(ddof=0))):
            return False, "s2n_scores() uses the population standard deviation: use the sample standard deviation, pandas' .std() (ddof=1)."
        if _same(got, (f.mean() - p.mean()).abs() / X.std()):
            return False, "s2n_scores() divides by the standard deviation of all the rows: divide by (std of the fails + std of the passes)."
        if _same(got, (f.mean() - p.mean()).abs()):
            return False, "s2n_scores() returns only the difference of the means: divide it by (std of the fails + std of the passes)."
        Xz = X.fillna(0)
        fz, pz = Xz[y == 1], Xz[y == 0]
        if _same(got, (fz.mean() - pz.mean()).abs() / (fz.std() + pz.std())):
            return False, "s2n_scores() counts missing values as 0: leave them out (pandas' .mean() and .std() skip NaN)."
        return False, "s2n_scores() gives wrong scores: |mean of fails - mean of passes| / (std of fails + std of passes), column by column."
    return True, "s2n_scores() passed its tests, including missing values and labels whose index does not start at 0."


# ---------------------------------------------------------------- tests, part B
def _records_cases():
    cases = []
    for k in (318, 777):
        rec = fab_records(k)
        cases.append((f"the records of KEY {k}", rec["runs"], rec["wafers"], rec["metrology"]))
    rec = fab_records(905)
    met = rec["metrology"].copy()
    w = rec["wafers"]
    r5 = w.loc[w["run_id"] == rec["runs"]["run_id"].iloc[4], "wafer_id"].iloc[1]
    met = met[met["wafer_id"] != r5]                         # one wafer of a run not measured
    met = met.sample(frac=1.0, random_state=5).reset_index(drop=True)   # records not in time order
    cases.append(("records with one wafer not measured, written out of order", rec["runs"], w, met))
    return cases


def _test_align(align_runs):
    for label, runs, wafers, met in _records_cases():
        ok, got = _call(align_runs, "align_runs", runs.copy(), wafers.copy(), met.copy())
        if not ok:
            return False, got
        if not isinstance(got, pd.DataFrame):
            return False, "align_runs() should return a DataFrame with one row per run."
        got = got.copy()
        if "run_id" not in got.columns:
            if got.index.name == "run_id":
                got = got.reset_index()
            else:
                return False, "align_runs() should keep the run_id column (or use run_id as the index)."
        for col in ("thickness_nm", "delay_h"):
            if col not in got.columns:
                return False, f"align_runs() should return a column named {col}."
        want = reference_align(runs, wafers, met)
        if len(got) < len(runs):
            return False, (f"align_runs() returned {len(got)} rows for {len(runs)} runs: a run with no metrology was dropped. Keep every run of the "
                           "MES run history: join the results onto runs with how=\"left\".")
        if len(got) > len(runs) or got["run_id"].duplicated().any():
            return False, (f"align_runs() returned {len(got)} rows for {len(runs)} runs. Average the wafer results of each run first "
                           "(groupby run_id), then join one row per run onto runs.")
        if set(got["run_id"]) != set(runs["run_id"]):
            return False, "align_runs() should return the runs of the MES run history, with their run_id values unchanged."
        got = got.set_index("run_id").reindex(want["run_id"])
        try:
            g_th, g_d = got["thickness_nm"].astype(float).values, got["delay_h"].astype(float).values
        except Exception:
            return False, "thickness_nm and delay_h should be numbers (delay_h in hours, for example (time difference) / pd.Timedelta(hours=1))."
        w_th, w_d = want["thickness_nm"].values, want["delay_h"].values
        no_met = np.isnan(w_th)
        if np.any(no_met & ~np.isnan(g_th)) or np.any(no_met & ~np.isnan(g_d)):
            return False, ("For a run with no metrology, thickness_nm and delay_h should be NaN (unknown), not 0 or another number: "
                           "leave the NaN from the left join in place.")
        if not _same(g_th, w_th, 1e-6):
            m = met.merge(wafers, on="wafer_id")
            first = runs["run_id"].map(m.sort_values("measured_at").groupby("run_id")["thickness_nm"].first()).values
            if _same(g_th, first, 1e-6):
                return False, "thickness_nm uses one wafer per run: average the thickness of all the run's measured wafers."
            return False, (f"thickness_nm is wrong for {label}. Find each wafer's run by joining metrology to wafers on wafer_id, "
                           "then average thickness_nm by run_id.")
        if not _same(g_d, w_d, 1e-3):
            if _same(g_d, w_d * 3600, 1e-3) or _same(g_d, w_d * 60, 1e-3):
                return False, "delay_h is not in hours: divide the time difference by pd.Timedelta(hours=1)."
            if _same(g_d, -w_d, 1e-3):
                return False, "delay_h is negative: it is the time of the last metrology record minus the end_time of the run."
            return False, (f"delay_h is wrong for {label}: it is the hours from the run's end_time to the time of its last metrology record "
                           "(the moment the run's result is complete).")
    return True, "align_runs() passed its tests, including a run with no metrology, a late run, a wafer not measured and records out of order."


# ---------------------------------------------------------------- the check
def check_dt_module2(missing_fraction, keep_signals, s2n_scores, align_runs, KEY):
    try:
        key = int(KEY)
    except (TypeError, ValueError):
        return False, ["Set KEY to the number Canvas gives you in Question 1 of the lab check, for example KEY = 417."], None
    if not 100 <= key <= 999:
        return False, [f"KEY should be a whole number from 100 to 999, the number in Question 1 of the lab check; it is {KEY}."], None
    results = [_test_missing(missing_fraction), _test_keep(keep_signals), _test_s2n(s2n_scores), _test_align(align_runs)]
    msgs = [("PASSED  " if ok else "NOT PASSED  ") + m for ok, m in results]
    if not all(ok for ok, _ in results):
        return False, msgs, None
    return True, msgs, personal_value(key)
