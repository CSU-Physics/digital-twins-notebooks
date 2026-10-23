"""Prepared components of the Module 7 metrology-twin track (Digital Twins for Semiconductor Manufacturing).

The metrology twin predicts the oxide thickness of every product wafer of recipe DRYOX-90 from the furnace's trace
summary (virtual metrology, Module 4) and keeps itself in sync with one sampled wafer per run (the offset estimate of
Module 5). Every data value is SIMULATED with the course's furnace model (furnace.py 1.5.0).

vm_production(key)        training runs 1 to 40, production runs 41 to 100, and runs 101 to 115 after a thermocouple replacement
train_correction(train)   the Module 4 hybrid model's learned part: a scaler-and-ridge pipeline fitted to the residual
                          (measured minus the Deal-Grove physics prediction), with the penalty chosen by run-blocked CV
kalman_update(...)        the Module 5 Kalman update of the offset estimate (the course's version)
run_twin(key, vm_predict, offset_observation)
                          runs the twin over runs 41 to 115; one row per wafer
per_run_mae(log, column)  the mean absolute error of the five wafers of each run
TOLERANCE_NM, HELDOUT_RUNS, CHALLENGING_RUNS, SAMPLED_SLOT, Q_NM2, R_NM2   the validation plan and the filter settings
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import furnace as F
from vm_tools import FEATURES, physics_prediction

__version__ = "1.0.1"

TRAIN_RUNS = 40                # runs 1 to 40 train the hybrid model (five measured wafers each)
LAST_RUN = 115                 # production runs 41 to 100, then 15 runs after the thermocouple replacement
REPLACED_FROM_RUN = 101        # the center thermocouple was replaced before run 101 (from the maintenance log)
SAMPLED_SLOT = 65              # in production, one wafer per run (slot 65) is measured and sent to the twin
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
Q_NM2 = 0.004                  # variance of the offset's random step per run (nm^2)
R_NM2 = 0.20                   # variance of one sampled-wafer offset observation (nm^2)
TOLERANCE_NM = 0.5             # the twin is fit for its use if its mean absolute error is at most 0.5 nm
HELDOUT_RUNS = (41, 100)       # held-out test: production runs 41 to 100, every wafer predicted before its run is sampled
CHALLENGING_RUNS = (101, 115)  # challenging test: the 15 runs after the thermocouple replacement
BASELINE_RUNS = 5              # baseline: the mean of the last 5 sampled wafers
_DRIFT_C_PER_RUN = 0.025       # the true temperature falls below the readings by this much per run (thermocouple aging)
_REPLACEMENT_C = 3.0           # the new thermocouple reads 3.0 C high, so the furnace runs 3.0 C cooler than its traces say


def _draw(key):
    rng = np.random.default_rng([2085, int(key)])
    truth = {"fB": float(rng.uniform(0.92, 1.08)), "fBA": float(rng.uniform(0.88, 1.12)), "thin_C": float(rng.uniform(1.2, 1.8)),
             "thin_L": F.THIN_L_NM, "tc_offset_C": float(rng.uniform(-1.0, 1.0)), "run_sd_nm": 0.30, "wafer_sd_nm": 0.15,
             "metrology_sd_nm": 0.25}
    return truth, rng


_DATA = {}


def vm_production(key):
    """Runs of recipe DRYOX-90 (dry O2, 1000 C, native oxide, 80 to 100 minutes) on the simulated furnace of key.

    Returns a dict of three DataFrames, one row per wafer (slots 5, 35, 65, 95, 125), in time order:
      "train"              runs 1 to 40, used to fit the hybrid model;
      "production"         runs 41 to 100;
      "after_replacement"  runs 101 to 115, after the center thermocouple was replaced.
    Columns: run_index, slot, sampled (True for the wafer measured in production, slot 65), ox_time_min, T_top_C,
    T_center_C, T_bottom_C, h2o_ppm, o2_slm, and thickness_nm. In production only the sampled wafer goes to the twin; the
    other wafers were measured too, for this validation only, so the twin's predictions can be checked on every wafer.
    The same key always gives the same runs."""
    key = int(key)
    if key in _DATA:
        return {k: v.copy() for k, v in _DATA[key].items()}
    truth, rng = _draw(key)
    n = LAST_RUN
    t_ox = rng.integers(80, 101, n).astype(float)
    h2o = np.round(rng.uniform(2.0, 12.0, n), 1)
    run_off = rng.normal(0.0, 0.8, n)
    zones = run_off[:, None] + rng.normal(0.0, 0.15, (n, 3)) + np.array(F._VM_ZONE_OFFSET)
    o2 = np.round(rng.uniform(9.5, 10.5, n), 2)
    run_eff = rng.normal(0.0, truth["run_sd_nm"], n)
    rows = []
    for i in range(n):
        readings = np.round(F.SETPOINT_C + zones[i], 2)
        cooler = _REPLACEMENT_C if i + 1 >= REPLACED_FROM_RUN else 0.0
        for s in F.VM_SLOTS:
            T_read = float(np.interp(s, F._VM_ZONE_SLOTS, readings))
            rows.append({"run_index": i + 1, "slot": s, "sampled": s == SAMPLED_SLOT, "ox_time_min": t_ox[i],
                         "T_top_C": readings[0], "T_center_C": readings[1], "T_bottom_C": readings[2], "h2o_ppm": h2o[i],
                         "o2_slm": o2[i], "_T": T_read + truth["tc_offset_C"] - _DRIFT_C_PER_RUN * i - cooler, "_re": run_eff[i]})
    df = pd.DataFrame(rows)
    B, BA = F._rates_array(df["_T"].to_numpy())
    speed = 1.0 + F._VM_H2O_PER_PPM * df["h2o_ppm"].to_numpy()
    x = F._grow_many(df["ox_time_min"].to_numpy(), F.VM_XI_NM, B * truth["fB"] * speed, BA * truth["fBA"] * speed,
                     truth["thin_C"], truth["thin_L"])
    x = x + df["_re"].to_numpy() + rng.normal(0.0, truth["wafer_sd_nm"], len(df))
    df["thickness_nm"] = np.round(x + rng.normal(0.0, truth["metrology_sd_nm"], len(df)), 2)
    df = df.drop(columns=["_T", "_re"])
    r = df["run_index"]
    out = {"train": df[r <= TRAIN_RUNS].reset_index(drop=True),
           "production": df[(r > TRAIN_RUNS) & (r < REPLACED_FROM_RUN)].reset_index(drop=True),
           "after_replacement": df[r >= REPLACED_FROM_RUN].reset_index(drop=True)}
    _DATA[key] = out
    return {k: v.copy() for k, v in out.items()}


def train_correction(train):
    """Module 4: the learned correction of the hybrid model. A StandardScaler and Ridge pipeline, fitted to the residual
    (measured thickness minus physics_prediction()) with the FEATURES as inputs. The ridge penalty is chosen from ALPHAS by
    cross-validation over blocks of 8 consecutive runs (GroupKFold), so a run is never split. Returns the fitted pipeline."""
    groups = ((train["run_index"].to_numpy() - 1) // 8).astype(int)
    search = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": list(ALPHAS)},
                          cv=GroupKFold(n_splits=len(np.unique(groups))), scoring="neg_root_mean_squared_error")
    search.fit(train[FEATURES], train["thickness_nm"] - physics_prediction(train), groups=groups)
    return search.best_estimator_


def kalman_update(d_prev, P_prev, y, Q, R):
    """Module 5: one Kalman update of the offset estimate for a random-walk offset. A missing observation (NaN) keeps the
    estimate and lets its variance grow. Returns (d, P)."""
    P_minus = P_prev + Q
    if np.isnan(y):
        return d_prev, P_minus
    K = P_minus / (P_minus + R)
    return d_prev + K * (y - d_prev), (1.0 - K) * P_minus


def run_twin(key, vm_predict, offset_observation, model=None):
    """Run the metrology twin over production runs 41 to 115 of the furnace of key.

    For each run, in time order: vm_predict(model, wafers) predicts the five wafers from the run's trace summary (the
    hybrid model of Module 4), and the twin's prediction is that plus the current offset estimate d_hat. Then the sampled
    wafer (slot 65) is measured: offset_observation(measured_nm, vm_nm) turns it into an observation of the offset, and
    kalman_update() updates d_hat (starting at 0 nm with P = R) for the next run. The baseline predicts every wafer of a run
    as the mean of the last 5 sampled wafers. model is train_correction(vm_production(key)["train"]) unless given.

    Returns a DataFrame, one row per wafer: run_index, slot, sampled, predicted_nm, baseline_nm, measured_nm,
    error_nm and baseline_error_nm (measured minus prediction), d_hat_nm (the estimate used for the run)."""
    data = vm_production(key)
    if model is None:
        model = train_correction(data["train"])
    runs = pd.concat([data["production"], data["after_replacement"]], ignore_index=True)
    train = data["train"]
    recent = list(train[train["sampled"]].sort_values("run_index")["thickness_nm"].to_numpy()[-BASELINE_RUNS:])
    d_hat, P = 0.0, R_NM2
    parts = []
    for k, wafers in runs.groupby("run_index", sort=True):
        wafers = wafers.reset_index(drop=True)
        vm = np.asarray(vm_predict(model, wafers), dtype=float).reshape(-1)
        if vm.size != len(wafers):
            raise ValueError(f"vm_predict() must return one prediction per wafer ({len(wafers)}), not {vm.size}")
        base = float(np.mean(recent[-BASELINE_RUNS:]))
        part = wafers[["run_index", "slot", "sampled"]].copy()
        part["predicted_nm"] = vm + d_hat
        part["baseline_nm"] = base
        part["measured_nm"] = wafers["thickness_nm"].to_numpy()
        part["error_nm"] = part["measured_nm"] - part["predicted_nm"]
        part["baseline_error_nm"] = part["measured_nm"] - base
        part["d_hat_nm"] = d_hat
        parts.append(part)
        s = int(np.flatnonzero(wafers["sampled"].to_numpy())[0])
        y = float(wafers["thickness_nm"].iloc[s])
        obs = np.asarray(offset_observation(y, vm[s]), dtype=float).reshape(-1)
        if obs.size != 1:
            raise ValueError("offset_observation() must return one number")
        d_hat, P = kalman_update(d_hat, P, float(obs[0]), Q_NM2, R_NM2)
        recent.append(y)
    return pd.concat(parts, ignore_index=True)


def per_run_mae(log, column="error_nm", runs=None):
    """The mean absolute error of the wafers of each run (one value per run, as a Series indexed by run_index), for
    runs first to last when runs = (first, last). The wafers of one run share the run's effects, so the run, not the
    wafer, is the independent test unit."""
    if runs is not None:
        log = log[(log["run_index"] >= runs[0]) & (log["run_index"] <= runs[1])]
    return log.groupby("run_index")[column].apply(lambda e: float(np.mean(np.abs(e))))


def plot_log(log, title=None, tolerance=TOLERANCE_NM, windows=None):
    """The sampled wafers, and the twin's and the baseline's per-run mean absolute error against the run, with the tolerance.

    The verdict is about the MAE over all the runs of a test window, not about single runs: a run above the tolerance line
    does not by itself fail the twin, and an MAE within the tolerance does not make every run acceptable. For each test
    window (by default the held-out and the challenging window) the lower panel draws the twin's MAE over the window as a
    horizontal line with its approximate 95 % interval (which treats the runs as independent) as a shaded band."""
    import matplotlib.pyplot as plt
    from capstone import mae_ci
    if windows is None:
        windows = {"held-out": HELDOUT_RUNS, "challenging": CHALLENGING_RUNS}
    tw = per_run_mae(log, "error_nm")
    bl = per_run_mae(log, "baseline_error_nm")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True)
    s = log[log["sampled"]]
    axes[0].plot(s["run_index"], s["measured_nm"], "o", ms=4, color="0.35", label="sampled wafer, measured")
    axes[0].plot(s["run_index"], s["predicted_nm"], ".-", ms=4, lw=1.1, color="C0", label="sampled wafer, twin (before measuring)")
    axes[0].set_ylabel("thickness (nm)")
    axes[1].axhline(tolerance, color="C2", lw=1.2, ls="--", label=f"tolerance on the window MAE, {tolerance:g} nm")
    axes[1].plot(tw.index, tw.values, ".-", ms=4, lw=1, color="C0", alpha=0.8, label="twin: mean |error| of the run's 5 wafers")
    axes[1].plot(bl.index, bl.values, ".-", ms=3, lw=0.8, color="C3", label="baseline")
    lines = []
    for k, (name, (a, b)) in enumerate(windows.items()):
        runs = tw.loc[(tw.index >= a) & (tw.index <= b)]
        if runs.size < 2:
            continue
        m, hw, n = mae_ci(runs.to_numpy())
        axes[1].fill_between([a - 0.5, b + 0.5], m - hw, m + hw, color="C0", alpha=0.18, lw=0,
                             label="twin MAE of each test window, with its approx. 95 % interval" if k == 0 else None)
        axes[1].hlines(m, a - 0.5, b + 0.5, color="C0", lw=2.4)
        lines.append(f"{name}, runs {a} to {b}: twin MAE {m:.2f} nm, interval {m - hw:.2f} to {m + hw:.2f} nm (n = {n})")
    if lines:
        axes[1].set_title("; ".join(lines) + f"; tolerance on the MAE {tolerance:g} nm", fontsize=8, loc="left", wrap=True)
    axes[1].axvline(REPLACED_FROM_RUN - 0.5, color="k", lw=0.8, ls=":")
    axes[1].set_ylabel("mean |error| (nm)")
    axes[1].set_xlabel("run")
    axes[1].set_ylim(0, None)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.9)
    if title:
        axes[0].set_title(title, fontsize=10)
    fig.text(0.01, 0.005, "The verdict compares the MAE of a whole test window, with its interval, with the tolerance. "
             "A single run above the line does not fail the twin,\nand an MAE within the tolerance does not make every run "
             "acceptable.", fontsize=7.5, color="0.3", ha="left", va="bottom")
    plt.tight_layout(rect=(0, 0.045, 1, 1))
    plt.show()
