"""Helpers for the Module 5 lab (Digital Twins for Semiconductor Manufacturing): keeping the twin in sync.

Every data value the lab uses is SIMULATED with the course's furnace model (furnace.py 1.5.0, r2r_setup and r2r_furnace).

dg_thickness(t_min, B, BA, xi_nm)   the twin's Deal-Grove thickness after t_min minutes from xi_nm (Module 3)
qualification_offset(setup)         the offset d of the qualification run: measured minus the twin's prediction
closed_loop(key, method, ...)       runs the 60 production runs under one controller and returns a log, one row per run
summary(log, target_nm)             mean absolute error, largest deviation and runs held at a time limit
compare(logs, target_nm)            summary() for several logs, as a table
plot_runs(logs, setup)              thickness, oxidation time and offset estimate against the run number
"""

import math

import numpy as np
import pandas as pd

from furnace import R2R_QUAL_MIN, R2R_RUNS, r2r_furnace, r2r_setup

__version__ = "1.0.0"

Q_NM2 = 0.01          # how much the true offset may change from one run to the next (variance, nm^2), stated for the lab
R_NM2 = 0.16          # variance of one run's offset observation around the furnace's offset (nm^2), from the qualification study
LAMBDA = 0.2          # the EWMA weight the lab starts with
ALARM_LIMIT_NM = 2.0  # recalibration alarm: the estimate has moved more than this from the qualification offset ...
ALARM_RUNS = 3        # ... for this many runs in a row


def dg_thickness(t_min, B, BA, xi_nm):
    """Deal-Grove thickness (nm) after t_min minutes, from an initial oxide xi_nm: x = (-A + sqrt(A^2 + 4 B (t + tau))) / 2,
    with A = B / BA and tau = (xi^2 + A xi) / B. B in nm^2/min, BA (that is, B/A) in nm/min."""
    A = B / BA
    tau = (xi_nm ** 2 + A * xi_nm) / B
    return (-A + np.sqrt(A * A + 4.0 * B * (np.asarray(t_min, dtype=float) + tau))) / 2.0


def qualification_offset(setup):
    """The offset d of the qualification run 0: its measured thickness minus the twin's prediction for its time."""
    return float(setup["qual_thickness_nm"] - dg_thickness(setup["qual_time_min"], setup["B"], setup["BA"], setup["xi_nm"]))


def _missing(y):
    return y is None or (isinstance(y, (float, np.floating)) and math.isnan(y))


def closed_loop(key, method, next_time=None, ewma_update=None, kalman_update=None, scenario="drift", lam=LAMBDA,
                Q=Q_NM2, R=R_NM2, mode="automatic", review_every=3):
    """Run the 60 production runs of furnace key under one controller, on that scenario's disturbances.

    method "fixed": every run at the qualification time (90 min), no estimate;
           "ewma":  offset estimate d_hat = ewma_update(d_hat, offset, lam);
           "kalman": (d_hat, P) = kalman_update(d_hat, P, offset, Q, R), starting from P = R.
    Both estimators start from the qualification offset. Before each run, next_time(d_hat, target_nm, B, BA, xi_nm, t_limits)
    recommends the oxidation time. mode "automatic" runs every recommendation; mode "advisory" runs a new recommendation
    only on the runs an engineer reviews (runs 1, 1 + review_every, ...) and otherwise keeps the last approved time.
    After the run, the offset observation is the measured thickness minus the twin's prediction for the time that ran
    (NaN when the run was not measured).

    Returns a DataFrame, one row per run: run, recommended_min, ox_time_min, held (time at a limit), measured_nm,
    offset_nm, d_hat_nm (estimate after the run), P_nm2 (Kalman variance after the run), achieved_nm (the run's true mean
    thickness: only a simulation knows it) and error_nm (achieved minus target)."""
    if method not in ("fixed", "ewma", "kalman"):
        raise ValueError('method must be "fixed", "ewma" or "kalman"')
    if method != "fixed" and next_time is None:
        raise ValueError("pass your next_time function: closed_loop(KEY, method, next_time=next_time, ...)")
    if method == "ewma" and ewma_update is None:
        raise ValueError("pass your EWMA function: closed_loop(KEY, 'ewma', next_time=next_time, ewma_update=ewma_update)")
    if method == "kalman" and kalman_update is None:
        raise ValueError("pass your Kalman function: closed_loop(KEY, 'kalman', next_time=next_time, kalman_update=kalman_update)")
    if mode not in ("automatic", "advisory"):
        raise ValueError('mode must be "automatic" or "advisory"')
    s = r2r_setup(key)
    lo, hi = s["t_limits"]
    furn = r2r_furnace(key, scenario)
    d_hat = qualification_offset(s) if method != "fixed" else float("nan")
    P = float(R) if method == "kalman" else float("nan")
    applied = R2R_QUAL_MIN
    rows = []
    for run in range(1, R2R_RUNS + 1):
        if method == "fixed":
            rec = R2R_QUAL_MIN
        else:
            rec = float(next_time(d_hat, s["target_nm"], s["B"], s["BA"], s["xi_nm"], s["t_limits"]))
        if mode == "automatic" or (run - 1) % int(review_every) == 0:
            applied = rec
        t = applied
        y = furn.run(t)
        offset = float("nan") if _missing(y) else float(y - dg_thickness(t, s["B"], s["BA"], s["xi_nm"]))
        if method == "ewma":
            d_hat = float(ewma_update(d_hat, offset, lam))
        elif method == "kalman":
            d_hat, P = kalman_update(d_hat, P, offset, Q, R)
            d_hat, P = float(d_hat), float(P)
        rows.append({"run": run, "recommended_min": rec, "ox_time_min": t, "held": bool(t <= lo or t >= hi),
                     "measured_nm": y, "offset_nm": offset, "d_hat_nm": d_hat, "P_nm2": P,
                     "achieved_nm": furn.achieved[-1], "error_nm": furn.achieved[-1] - s["target_nm"]})
    return pd.DataFrame(rows)


def summary(log, target_nm=None):
    """mae_nm (mean absolute error of the achieved thickness), max_dev_nm (largest absolute error) and held_runs
    (runs whose oxidation time sat at a time limit). target_nm is not needed: the log already holds the errors."""
    err = log["error_nm"].to_numpy(dtype=float)
    return {"mae_nm": float(np.mean(np.abs(err))), "max_dev_nm": float(np.max(np.abs(err))), "held_runs": int(log["held"].sum())}


def compare(logs, target_nm=None):
    """summary() of each log in a dict {name: log}, as a table with one row per controller."""
    table = pd.DataFrame([summary(log) for log in logs.values()], index=list(logs.keys()))
    return table[["mae_nm", "max_dev_nm", "held_runs"]]


def plot_runs(logs, setup, title=None):
    """Three panels against the run number: achieved thickness with the target, oxidation time with its limits, and the
    offset estimate with the offset observations of the last log. logs is a dict {name: log}."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.6), sharex=True)
    for i, (name, log) in enumerate(logs.items()):
        color = f"C{i}"
        axes[0].plot(log["run"], log["achieved_nm"], ".-", ms=4, lw=1, color=color, label=name)
        axes[1].plot(log["run"], log["ox_time_min"], "-", lw=1.4, color=color, label=name)
        if log["d_hat_nm"].notna().any():
            axes[2].plot(log["run"], log["d_hat_nm"], "-", lw=1.4, color=color, label=f"{name} estimate")
    last = list(logs.values())[-1]
    axes[2].plot(last["run"], last["offset_nm"], "o", ms=3, color="0.6", label="offset observed (measured minus twin)")
    axes[0].axhline(setup["target_nm"], color="k", lw=1, ls="--", label="target")
    for lim in setup["t_limits"]:
        axes[1].axhline(lim, color="0.4", lw=1, ls=":")
    axes[0].set_ylabel("achieved (nm)")
    axes[1].set_ylabel("oxidation time (min)")
    axes[2].set_ylabel("offset d (nm)")
    axes[2].set_xlabel("run")
    for ax in axes:
        ax.legend(fontsize=8, loc="best")
    if title:
        axes[0].set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()
