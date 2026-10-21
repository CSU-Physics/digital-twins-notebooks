"""Tests of the Module 2 check: good implementations must pass, wrong ones must fail, for several KEYs.
Run from the repository root: python3 checks/test_dt_m2_check.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m2"), str(ROOT / "checks")]
from dt_m2_check import (check_dt_module2, personal_value, reference_align, reference_keep_signals,  # noqa: E402
                         reference_missing_fraction, reference_s2n)
from furnace import fab_records  # noqa: E402


# ---------------------------------------------------------------- missing_fraction
def mf_mean(X): return X.isna().mean()
def mf_sum(X): return X.isna().sum() / len(X)
def mf_loop(X): return pd.Series({c: X[c].isna().sum() / X.shape[0] for c in X.columns})
def mf_numpy(X): return np.isnan(X.to_numpy()).mean(axis=0)
def mf_count(X): return 1 - X.count() / len(X)
MF_GOOD = [mf_mean, mf_sum, mf_loop, mf_numpy, mf_count]

def mf_counts(X): return X.isna().sum()
def mf_present(X): return X.notna().mean()
def mf_rows(X): return X.isna().mean(axis=1)
def mf_raise(X): raise NotImplementedError
def mf_none(X): X.isna().mean()
MF_BAD = [mf_counts, mf_present, mf_rows, mf_raise, mf_none]


# ---------------------------------------------------------------- keep_signals
def ks_template(X_train, max_missing=0.5, max_share=0.95):
    keep = []
    for name in X_train.columns:
        col = X_train[name]
        present = col.dropna()
        if col.isna().mean() > max_missing or len(present) == 0:
            continue
        if present.value_counts(normalize=True).iloc[0] >= max_share:
            continue
        keep.append(name)
    return keep

def ks_vector(X_train, max_missing=0.5, max_share=0.95):
    miss = X_train.isna().mean()
    share = X_train.apply(lambda c: c.value_counts(normalize=True).max() if c.notna().any() else 1.0)
    return list(X_train.columns[(miss <= max_missing) & (share < max_share)])

def ks_mode(X_train, max_missing=0.5, max_share=0.95):
    keep = []
    for name in X_train.columns:
        col = X_train[name]
        if col.isna().mean() <= max_missing:
            present = col.dropna()
            top = (present == present.mode().iloc[0]).mean()
            if top < max_share:
                keep.append(name)
    return keep

def ks_index(X_train, max_missing=0.5, max_share=0.95):
    return pd.Index(ks_vector(X_train, max_missing, max_share))
KS_GOOD = [ks_template, ks_vector, ks_mode, ks_index]

def ks_dropna_false(X_train, max_missing=0.5, max_share=0.95):
    miss = X_train.isna().mean()
    share = X_train.apply(lambda c: c.value_counts(normalize=True, dropna=False).iloc[0])
    return list(X_train.columns[(miss <= max_missing) & (share < max_share)])

def ks_std(X_train, max_missing=0.5, max_share=0.95):
    miss = X_train.isna().mean()
    return list(X_train.columns[(miss <= max_missing) & (X_train.std() > 1e-3)])

def ks_constant_only(X_train, max_missing=0.5, max_share=0.95):
    miss = X_train.isna().mean()
    return list(X_train.columns[(miss <= max_missing) & (X_train.nunique() > 1)])

def ks_fixed(X_train, max_missing=0.5, max_share=0.95):
    return ks_vector(X_train)

def ks_mask(X_train, max_missing=0.5, max_share=0.95):
    miss = X_train.isna().mean()
    share = X_train.apply(lambda c: c.value_counts(normalize=True).max() if c.notna().any() else 1.0)
    return (miss <= max_missing) & (share < max_share)

def ks_missing_flip(X_train, max_missing=0.5, max_share=0.95):
    miss = X_train.isna().mean()
    share = X_train.apply(lambda c: c.value_counts(normalize=True).max() if c.notna().any() else 1.0)
    return list(X_train.columns[(miss > max_missing) | (share >= max_share)])

def ks_no_missing_rule(X_train, max_missing=0.5, max_share=0.95):
    share = X_train.apply(lambda c: c.value_counts(normalize=True).max() if c.notna().any() else 1.0)
    return list(X_train.columns[share < max_share])

def ks_table(X_train, max_missing=0.5, max_share=0.95):
    return X_train[ks_vector(X_train, max_missing, max_share)]
KS_BAD = [ks_dropna_false, ks_std, ks_constant_only, ks_fixed, ks_mask, ks_missing_flip, ks_no_missing_rule, ks_table]


# ---------------------------------------------------------------- s2n_scores
def s2n_ref(X, y):
    fail, ok = X[y == 1], X[y == 0]
    return (fail.mean() - ok.mean()).abs() / (fail.std() + ok.std())

def s2n_values(X, y):
    yy = np.asarray(y)
    fail, ok = X[yy == 1], X[yy == 0]
    return np.abs(fail.mean() - ok.mean()) / (fail.std() + ok.std())

def s2n_loop(X, y):
    out = {}
    for c in X.columns:
        f, p = X.loc[y == 1, c].dropna(), X.loc[y == 0, c].dropna()
        out[c] = abs(f.mean() - p.mean()) / (f.std() + p.std())
    return pd.Series(out)

def s2n_groupby(X, y):
    m, s = X.groupby(y).mean(), X.groupby(y).std()
    return (m.loc[1] - m.loc[0]).abs() / (s.loc[1] + s.loc[0])
S2N_GOOD = [s2n_ref, s2n_values, s2n_loop, s2n_groupby]

def s2n_signed(X, y):
    fail, ok = X[y == 1], X[y == 0]
    return (fail.mean() - ok.mean()) / (fail.std() + ok.std())

def s2n_ddof0(X, y):
    fail, ok = X[y == 1], X[y == 0]
    return (fail.mean() - ok.mean()).abs() / (fail.std(ddof=0) + ok.std(ddof=0))

def s2n_allstd(X, y):
    fail, ok = X[y == 1], X[y == 0]
    return (fail.mean() - ok.mean()).abs() / X.std()

def s2n_fill0(X, y):
    Z = X.fillna(0)
    fail, ok = Z[y == 1], Z[y == 0]
    return (fail.mean() - ok.mean()).abs() / (fail.std() + ok.std())

def s2n_diff(X, y):
    return (X[y == 1].mean() - X[y == 0].mean()).abs()

def s2n_numpy_nan(X, y):
    A, yy = X.to_numpy(), np.asarray(y)
    return np.abs(A[yy == 1].mean(0) - A[yy == 0].mean(0)) / (A[yy == 1].std(0, ddof=1) + A[yy == 0].std(0, ddof=1))

def s2n_pooled(X, y):
    fail, ok = X[y == 1], X[y == 0]
    return (fail.mean() - ok.mean()).abs() / np.sqrt((fail.var() + ok.var()) / 2)
S2N_BAD = [s2n_signed, s2n_ddof0, s2n_allstd, s2n_fill0, s2n_diff, s2n_numpy_nan, s2n_pooled]


# ---------------------------------------------------------------- align_runs
def al_ref(runs, wafers, metrology):
    return reference_align(runs, wafers, metrology)

def al_template(runs, wafers, metrology):
    m = metrology.merge(wafers, on="wafer_id", how="left")
    per_run = m.groupby("run_id").agg(thickness_nm=("thickness_nm", "mean"), last_measured_at=("measured_at", "max")).reset_index()
    table = runs.merge(per_run, on="run_id", how="left")
    table["delay_h"] = (table["last_measured_at"] - table["end_time"]) / pd.Timedelta(hours=1)
    return table

def al_index(runs, wafers, metrology):
    m = wafers.merge(metrology, on="wafer_id")
    g = m.groupby("run_id")
    out = runs.set_index("run_id")[["end_time"]].copy()
    out["thickness_nm"] = g["thickness_nm"].mean()
    out["delay_h"] = (g["measured_at"].max() - out["end_time"]).dt.total_seconds() / 3600
    return out

def al_loop(runs, wafers, metrology):
    rows = []
    for _, r in runs.iterrows():
        ids = wafers.loc[wafers["run_id"] == r["run_id"], "wafer_id"]
        res = metrology[metrology["wafer_id"].isin(ids)]
        if len(res) == 0:
            rows.append({"run_id": r["run_id"], "thickness_nm": np.nan, "delay_h": np.nan})
        else:
            rows.append({"run_id": r["run_id"], "thickness_nm": res["thickness_nm"].mean(),
                         "delay_h": (res["measured_at"].max() - r["end_time"]).total_seconds() / 3600})
    return pd.DataFrame(rows)
AL_GOOD = [al_ref, al_template, al_index, al_loop]

def al_inner(runs, wafers, metrology):
    t = al_template(runs, wafers, metrology)
    return t.dropna(subset=["thickness_nm"])

def al_no_group(runs, wafers, metrology):
    m = metrology.merge(wafers, on="wafer_id")
    t = runs.merge(m, on="run_id", how="left")
    t["delay_h"] = (t["measured_at"] - t["end_time"]) / pd.Timedelta(hours=1)
    return t

def al_seconds(runs, wafers, metrology):
    t = al_template(runs, wafers, metrology)
    t["delay_h"] = (t["last_measured_at"] - t["end_time"]).dt.total_seconds()
    return t

def al_sign(runs, wafers, metrology):
    t = al_template(runs, wafers, metrology)
    t["delay_h"] = -t["delay_h"]
    return t

def al_first(runs, wafers, metrology):
    m = metrology.merge(wafers, on="wafer_id")
    g = m.groupby("run_id").agg(thickness_nm=("thickness_nm", "mean"), first=("measured_at", "min")).reset_index()
    t = runs.merge(g, on="run_id", how="left")
    t["delay_h"] = (t["first"] - t["end_time"]) / pd.Timedelta(hours=1)
    return t

def al_fill0(runs, wafers, metrology):
    return al_template(runs, wafers, metrology).fillna({"thickness_nm": 0.0, "delay_h": 0.0})

def al_one_wafer(runs, wafers, metrology):
    m = metrology.merge(wafers, on="wafer_id").sort_values("measured_at")
    g = m.groupby("run_id").agg(thickness_nm=("thickness_nm", "first"), last=("measured_at", "max")).reset_index()
    t = runs.merge(g, on="run_id", how="left")
    t["delay_h"] = (t["last"] - t["end_time"]) / pd.Timedelta(hours=1)
    return t

def al_slot(runs, wafers, metrology):
    # joins by slot position in the file instead of by wafer ID
    m = metrology.copy()
    m["run_id"] = wafers["run_id"].iloc[: len(m)].values
    g = m.groupby("run_id").agg(thickness_nm=("thickness_nm", "mean"), last=("measured_at", "max")).reset_index()
    t = runs.merge(g, on="run_id", how="left")
    t["delay_h"] = (t["last"] - t["end_time"]) / pd.Timedelta(hours=1)
    return t

def al_start(runs, wafers, metrology):
    t = al_template(runs, wafers, metrology)
    t["delay_h"] = (t["last_measured_at"] - t["start_time"]) / pd.Timedelta(hours=1)
    return t

def al_no_runid(runs, wafers, metrology):
    return al_template(runs, wafers, metrology).drop(columns=["run_id"])

def al_raise(runs, wafers, metrology):
    raise NotImplementedError
AL_BAD = [al_inner, al_no_group, al_seconds, al_sign, al_first, al_fill0, al_one_wafer, al_slot, al_start, al_no_runid, al_raise]


KEYS = [100, 417, 999]
bad = 0
checked_good = checked_bad = 0
G = dict(missing_fraction=mf_mean, keep_signals=ks_template, s2n_scores=s2n_ref, align_runs=al_template)
for key in KEYS:
    want = personal_value(key)
    for i in range(max(len(MF_GOOD), len(KS_GOOD), len(S2N_GOOD), len(AL_GOOD))):
        fns = (MF_GOOD[i % len(MF_GOOD)], KS_GOOD[i % len(KS_GOOD)], S2N_GOOD[i % len(S2N_GOOD)], AL_GOOD[i % len(AL_GOOD)])
        ok, msgs, val = check_dt_module2(*fns, key)
        checked_good += 1
        if not ok or val != want:
            bad += 1
            print("WRONG VERDICT good", key, [f.__name__ for f in fns], msgs)
    for slot, wrongs in [("missing_fraction", MF_BAD), ("keep_signals", KS_BAD), ("s2n_scores", S2N_BAD), ("align_runs", AL_BAD)]:
        for w in wrongs:
            fns = dict(G)
            fns[slot] = w
            ok, msgs, val = check_dt_module2(fns["missing_fraction"], fns["keep_signals"], fns["s2n_scores"], fns["align_runs"], key)
            checked_bad += 1
            if ok or val is not None:
                bad += 1
                print("WRONG VERDICT bad", key, w.__name__, msgs)
            elif key == 417:
                print(f"{w.__name__:>20}: {[m for m in msgs if m.startswith('NOT')][0]}")
for k in [None, "abc", 99, 1000]:
    ok, msgs, val = check_dt_module2(mf_mean, ks_template, s2n_ref, al_template, k)
    checked_bad += 1
    if ok:
        bad += 1
        print("WRONG VERDICT key", k, msgs)
print(check_dt_module2(mf_mean, ks_template, s2n_ref, al_template, 417)[1])
print(f"{checked_good} good and {checked_bad} wrong combinations over KEYs {KEYS}: {bad} wrong verdicts")
assert bad == 0
