"""Helpers for the Module 4 lab (Digital Twins for Semiconductor Manufacturing): virtual metrology for oxide thickness.

Every data value the lab uses is SIMULATED with the course's furnace model (furnace.py 1.4.0, vm_runs).

time_split(table)           the first 40 runs for training and tuning, runs 41 to 60 for the final, time-ordered test
run_blocks(table)           fold labels for cross-validation: blocks of 8 consecutive runs, so a run is never split
scores(y, y_pred)           RMSE, mean absolute error, largest error and mean error (measured minus predicted), in nm
physics_prediction(table)   the Module 3 Deal-Grove thickness with textbook constants at each run's center-zone
                            reading and oxidation time, from the native oxide
cv_rmse(model, X, y, folds) mean cross-validated RMSE of a scikit-learn model over the folds
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score

from furnace import VM_FEATURES, VM_TRAIN_RUNS, VM_XI_NM, _rates_array

__version__ = "1.0.1"

FEATURES = list(VM_FEATURES)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)   # ridge penalties the lab tries
BLOCK_RUNS = 8                           # runs per cross-validation block: 40 training runs make 5 blocks


def time_split(table, n_train_runs=VM_TRAIN_RUNS):
    """Split a wafer table by time: rows of runs 1..n_train_runs, and the rows of later runs."""
    train = table[table["run_index"] <= n_train_runs].reset_index(drop=True)
    test = table[table["run_index"] > n_train_runs].reset_index(drop=True)
    return train, test


def run_blocks(table, block_runs=BLOCK_RUNS):
    """One label per row: the block of consecutive runs the row's run belongs to (runs 1-8 are block 0, 9-16 block 1, ...).
    Used as groups with GroupKFold, every fold holds out whole runs that are next to each other in time."""
    return ((table["run_index"].to_numpy() - 1) // block_runs).astype(int)


def block_folds(table, block_runs=BLOCK_RUNS):
    """GroupKFold over the run blocks: (cv, groups) to pass to cross_val_score or GridSearchCV.
    The folds keep whole runs (and blocks of neighbouring runs) together, which prevents leakage within runs. They are not
    walk-forward validation: GroupKFold does not put the training runs before the validation runs, so some folds train on
    later runs than they validate. For a tool that drifts, tune with expanding-window splits over whole runs instead."""
    groups = run_blocks(table, block_runs)
    return GroupKFold(n_splits=len(np.unique(groups))), groups


def scores(y, y_pred):
    """Errors of predictions against measurements, in nm: rmse, mae, max_abs and bias (mean of measured minus predicted)."""
    y = np.asarray(y, dtype=float)
    e = y - np.asarray(y_pred, dtype=float)
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "max_abs": float(np.max(np.abs(e))), "bias": float(np.mean(e))}


def physics_prediction(table, xi_nm=VM_XI_NM):
    """Deal-Grove thickness (nm) with the textbook dry-O2 constants for (100) silicon (Module 3), one value per row:
    evaluated at the row's center-zone reading T_center_C and oxidation time ox_time_min, starting from xi_nm.
    It knows nothing about this furnace: not its own rate constants, the thin-oxide regime, moisture or the boat position."""
    B, BA = _rates_array(table["T_center_C"].to_numpy())
    A = B / BA
    tau = (xi_nm ** 2 + A * xi_nm) / B
    t = table["ox_time_min"].to_numpy(dtype=float)
    return A / 2.0 * (np.sqrt(1.0 + 4.0 * B * (t + tau) / A ** 2) - 1.0)


def cv_rmse(model, X, y, cv, groups=None):
    """Mean RMSE over the cross-validation folds (the model is refitted, preprocessing included, inside every fold)."""
    s = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="neg_root_mean_squared_error")
    return float(-np.mean(s))
