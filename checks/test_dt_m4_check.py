"""Tests for dt_m4_check: correct learner functions must pass, each typical mistake must not pass.

Run from the repository root:  python checks/test_dt_m4_check.py
"""
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "content", "m4"), HERE]
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from dt_m4_check import check_dt_module4


# ---------------------------------------------------------------- baseline_recent: good and wrong
def bl_good(table, n_runs=3):
    per_run = table.groupby("run_index")["thickness_nm"].agg(["sum", "count"])
    mean_prev = per_run["sum"].rolling(n_runs).sum().shift(1) / per_run["count"].rolling(n_runs).sum().shift(1)
    return table["run_index"].map(mean_prev)


def bl_good_loop(table, n_runs=3):
    runs = sorted(table["run_index"].unique())
    out = pd.Series(np.nan, index=table.index)
    for i, r in enumerate(runs):
        if i >= n_runs:
            prev = table[table["run_index"].isin(runs[i - n_runs:i])]
            out[table["run_index"] == r] = prev["thickness_nm"].mean()
    return out


def bl_good_array(table, n_runs=3):
    return bl_good(table, n_runs).to_numpy()


def bl_no_shift(table, n_runs=3):
    per_run = table.groupby("run_index")["thickness_nm"].agg(["sum", "count"])
    return table["run_index"].map(per_run["sum"].rolling(n_runs).sum() / per_run["count"].rolling(n_runs).sum())


def bl_run_means(table, n_runs=3):
    m = table.groupby("run_index")["thickness_nm"].mean()
    return table["run_index"].map(m.rolling(n_runs).mean().shift(1))


def bl_fixed3(table, n_runs=3):
    return bl_good(table, 3)


def bl_min_periods(table, n_runs=3):
    per_run = table.groupby("run_index")["thickness_nm"].agg(["sum", "count"])
    mp = per_run["sum"].rolling(n_runs, min_periods=1).sum().shift(1) / per_run["count"].rolling(n_runs, min_periods=1).sum().shift(1)
    return table["run_index"].map(mp)


def bl_rows(table, n_runs=3):
    return table["thickness_nm"].rolling(5 * n_runs).mean().shift(1)


def bl_index_order(table, n_runs=3):
    per_run = table.groupby("run_index")["thickness_nm"].agg(["sum", "count"])
    mean_prev = per_run["sum"].rolling(n_runs).sum().shift(1) / per_run["count"].rolling(n_runs).sum().shift(1)
    return pd.Series(table["run_index"].map(mean_prev).to_numpy())          # loses the table's index


def bl_per_run(table, n_runs=3):
    per_run = table.groupby("run_index")["thickness_nm"].mean()
    return per_run.rolling(n_runs).mean().shift(1)


def bl_none(table, n_runs=3):
    table["run_index"].map(lambda r: r)


def bl_raise(table, n_runs=3):
    raise NotImplementedError("Complete baseline_recent() first.")


# ---------------------------------------------------------------- make_ridge
def mr_good(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def mr_good_pipe(alpha):
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def mr_no_scaler(alpha):
    return make_pipeline(Ridge(alpha=alpha))


def mr_plain_ridge(alpha):
    return Ridge(alpha=alpha)


def mr_fixed_alpha(alpha):
    return make_pipeline(StandardScaler(), Ridge())


def mr_linear(alpha):
    return make_pipeline(StandardScaler(), LinearRegression())


def mr_minmax(alpha):
    return make_pipeline(MinMaxScaler(), Ridge(alpha=alpha))


def mr_raise(alpha):
    raise NotImplementedError("Complete make_ridge() first.")


# ---------------------------------------------------------------- hybrid_predict
def hy_good(X_train, y_train, phys_train, X_new, phys_new, alpha):
    model = mr_good(alpha).fit(X_train, y_train - phys_train)
    return phys_new + model.predict(X_new)


def hy_good_np(X_train, y_train, phys_train, X_new, phys_new, alpha):
    r = np.asarray(y_train) - np.asarray(phys_train)
    return np.asarray(phys_new) + make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X_train, r).predict(X_new)


def hy_correction_only(X_train, y_train, phys_train, X_new, phys_new, alpha):
    return mr_good(alpha).fit(X_train, y_train - phys_train).predict(X_new)


def hy_plain(X_train, y_train, phys_train, X_new, phys_new, alpha):
    return mr_good(alpha).fit(X_train, y_train).predict(X_new)


def hy_double(X_train, y_train, phys_train, X_new, phys_new, alpha):
    return phys_new + mr_good(alpha).fit(X_train, y_train).predict(X_new)


def hy_sign(X_train, y_train, phys_train, X_new, phys_new, alpha):
    return phys_new + mr_good(alpha).fit(X_train, phys_train - y_train).predict(X_new)


def hy_alpha1(X_train, y_train, phys_train, X_new, phys_new, alpha):
    return phys_new + mr_good(1.0).fit(X_train, y_train - phys_train).predict(X_new)


def hy_no_scale(X_train, y_train, phys_train, X_new, phys_new, alpha):
    return phys_new + Ridge(alpha=alpha).fit(X_train, y_train - phys_train).predict(X_new)


def hy_train_rows(X_train, y_train, phys_train, X_new, phys_new, alpha):
    return phys_train + mr_good(alpha).fit(X_train, y_train - phys_train).predict(X_train)


def hy_raise(*a):
    raise NotImplementedError("Complete hybrid_predict() first.")


# ---------------------------------------------------------------- outside_range
def _lim(X_train, margin):
    lo, hi = X_train.min(), X_train.max()
    return lo - margin * (hi - lo), hi + margin * (hi - lo)


def out_good(X_train, X_new, margin=0.1):
    lo, hi = _lim(X_train, margin)
    return ((X_new < lo) | (X_new > hi)).any(axis=1)


def out_good_np(X_train, X_new, margin=0.1):
    a, b = X_train.to_numpy().min(axis=0), X_train.to_numpy().max(axis=0)
    x = X_new.to_numpy()
    return np.any((x < a - margin * (b - a)) | (x > b + margin * (b - a)), axis=1)


def out_no_margin(X_train, X_new, margin=0.1):
    return ((X_new < X_train.min()) | (X_new > X_train.max())).any(axis=1)


def out_fixed_margin(X_train, X_new, margin=0.1):
    lo, hi = _lim(X_train, 0.1)
    return ((X_new < lo) | (X_new > hi)).any(axis=1)


def out_value_margin(X_train, X_new, margin=0.1):
    lo, hi = X_train.min(), X_train.max()
    return ((X_new < lo * (1 - margin)) | (X_new > hi * (1 + margin))).any(axis=1)


def out_inclusive(X_train, X_new, margin=0.1):
    lo, hi = _lim(X_train, margin)
    return ((X_new <= lo) | (X_new >= hi)).any(axis=1)


def out_all(X_train, X_new, margin=0.1):
    lo, hi = _lim(X_train, margin)
    return ((X_new < lo) | (X_new > hi)).all(axis=1)


def out_first_col(X_train, X_new, margin=0.1):
    c = X_train.columns[0]
    lo, hi = _lim(X_train, margin)
    return (X_new[c] < lo[c]) | (X_new[c] > hi[c])


def out_upper_only(X_train, X_new, margin=0.1):
    lo, hi = _lim(X_train, margin)
    return (X_new > hi).any(axis=1)


def out_table(X_train, X_new, margin=0.1):
    lo, hi = _lim(X_train, margin)
    return (X_new < lo) | (X_new > hi)


def out_new_range(X_train, X_new, margin=0.1):
    lo, hi = _lim(X_new, margin)
    return ((X_new < lo) | (X_new > hi)).any(axis=1)


def out_sd(X_train, X_new, margin=0.1):
    m, s = X_train.mean(), X_train.std()
    return ((X_new - m).abs() > 2 * s).any(axis=1)


def out_raise(X_train, X_new, margin=0.1):
    raise NotImplementedError("Complete outside_range() first.")


GOOD = {"baseline": [bl_good, bl_good_loop, bl_good_array], "ridge": [mr_good, mr_good_pipe],
        "hybrid": [hy_good, hy_good_np], "outside": [out_good, out_good_np]}
WRONG = {"baseline": [bl_no_shift, bl_run_means, bl_fixed3, bl_min_periods, bl_rows, bl_index_order, bl_per_run, bl_none, bl_raise],
         "ridge": [mr_no_scaler, mr_plain_ridge, mr_fixed_alpha, mr_linear, mr_minmax, mr_raise],
         "hybrid": [hy_correction_only, hy_plain, hy_double, hy_sign, hy_alpha1, hy_no_scale, hy_train_rows, hy_raise],
         "outside": [out_no_margin, out_fixed_margin, out_value_margin, out_inclusive, out_all, out_first_col, out_upper_only,
                     out_table, out_new_range, out_sd, out_raise]}


def main():
    bad, n_good, n_wrong = 0, 0, 0
    keys = [100, 417, 999]
    g = {k: v[0] for k, v in GOOD.items()}
    for key in keys:
        for b in GOOD["baseline"]:
            for r in GOOD["ridge"]:
                for h in GOOD["hybrid"]:
                    for o in GOOD["outside"]:
                        ok, msgs, val = check_dt_module4(b, r, h, o, key)
                        n_good += 1
                        if not ok or val is None or abs(val - __import__('dt_m4_check').verification_code(key)) > 1e-9:
                            bad += 1
                            print("GOOD FAILED", key, b.__name__, r.__name__, h.__name__, o.__name__, msgs)
        for part, fns in WRONG.items():
            for f in fns:
                args = dict(g)
                args[part] = f
                ok, msgs, val = check_dt_module4(args["baseline"], args["ridge"], args["hybrid"], args["outside"], key)
                n_wrong += 1
                if ok or val is not None:
                    bad += 1
                    print("WRONG PASSED", key, f.__name__)
                elif key == keys[0]:
                    print(f"{f.__name__:>20}: " + [m for m in msgs if m.startswith("NOT")][0])
    print(check_dt_module4(g["baseline"], g["ridge"], g["hybrid"], g["outside"], 417))
    print(check_dt_module4(g["baseline"], g["ridge"], g["hybrid"], g["outside"], None)[1])
    print(check_dt_module4(g["baseline"], g["ridge"], g["hybrid"], g["outside"], 5000)[1])
    print(f"{n_good} good and {n_wrong} wrong combinations over KEYs {keys}: {bad} wrong verdicts")
    return bad


def bypass_test():
    """The verification value is not the result the notebook prints earlier."""
    from dt_m4_check import personal_value as _result, verification_code as _code  # noqa: E402
    for _k in (417, 100):
        assert abs(_code(_k) - _result(_k)) > 0.0005, "the verification value must differ from the result printed before the check"
    print("bypass test passed: the verification value differs from the printed result for KEYs", (417, 100))


if __name__ == "__main__":
    bypass_test()
    sys.exit(1 if main() else 0)

