"""Prepared components of the Module 7 process-twin track (Digital Twins for Semiconductor Manufacturing).

The process twin predicts the monitor-wafer oxide thickness of each production run of the course's simulated furnace
(recipe DRYOX-90: dry O2, 1000 C, native oxide, 80 to 100 minutes) and recommends the next oxidation time. It joins the
Deal-Grove model of Module 3, calibrated on qualification runs, with the offset estimate of Module 5.
Every data value is SIMULATED with furnace.py (1.5.0); the drift and the failing thermocouple are those of Module 5.

qualification_runs(key)    12 qualification runs (60 to 126 minutes) of furnace key, before production starts
calibrate(key)             fit_deal_grove() of Module 3 on those runs: the twin's B and B/A, with their uncertainty
kalman_update(...)         the Module 5 Kalman update of the offset estimate (the course's version)
next_time(...)             the Module 5 recipe rule: the oxidation time that aims the twin at the target, within limits
run_twin(key, scenario, predict_next, offset_observation)
                           runs the 60 production runs with the twin in the loop; one row per run
TOLERANCE_NM, HELDOUT_RUNS, CHALLENGING_RUNS, Q_NM2, R_NM2   the validation plan and the filter settings
"""

import numpy as np
import pandas as pd

import furnace as F
from oxide_tools import fit_deal_grove, predict
from r2r_tools import dg_thickness

__version__ = "1.0.0"

QUAL_TIMES_MIN = (60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126)   # qualification runs, minutes
MONITOR_SD_NM = 0.32           # standard deviation of one monitor-wafer measurement, stated by the metrology group
Q_NM2 = 0.01                   # Module 5: variance of the offset's random step per run (nm^2)
R_NM2 = 0.16                   # Module 5: variance of one offset observation (nm^2)
TOLERANCE_NM = 0.5             # the twin is fit for its use if its mean absolute error is at most 0.5 nm
HELDOUT_RUNS = (1, 60)         # held-out test: runs 1 to 60 of the "drift" scenario, each predicted before it is measured
CHALLENGING_RUNS = (31, 60)    # challenging test: runs 31 to 60 of the "failing" scenario (the thermocouple fails by run 35)


def qualification_runs(key):
    """Qualification of furnace key before production: 12 runs of 60 to 126 minutes on bare wafers (1.5 nm native oxide),
    with the thermocouple freshly calibrated. One monitor-wafer measurement per run (0.01 nm resolution, standard
    deviation MONITOR_SD_NM). Columns: run_id, time_min, xi_nm, thickness_nm. The same key always gives the same runs."""
    key = int(key)
    truth, twin, bias, run_T, monitor = F._r2r_draw(key)
    rng = np.random.default_rng([2077, key])
    t = np.array(QUAL_TIMES_MIN, dtype=float)
    T = F.SETPOINT_C + truth["tc_offset_C"] + rng.normal(0.0, truth["run_T_sd_C"], t.size)
    B, BA = F._rates_array(T)
    x = F._grow_many(t, F.R2R_XI_NM, B * truth["fB"], BA * truth["fBA"], truth["thin_C"], truth["thin_L"])
    y = np.round(x + rng.normal(0.0, truth["monitor_sd_nm"], t.size), 2)
    return pd.DataFrame({"run_id": [f"QL{key}-{i + 1:02d}" for i in range(t.size)], "time_min": t,
                         "xi_nm": F.R2R_XI_NM, "thickness_nm": y})


def calibrate(key):
    """The Module 3 fit on the qualification runs of furnace key: a dict with B, BA, A, their standard errors, chi2, dof,
    residuals and what predict() needs. The twin's Deal-Grove prediction for a time t is predict(fit, t)["x"]."""
    q = qualification_runs(key)
    return fit_deal_grove(q["time_min"], q["thickness_nm"], MONITOR_SD_NM, F.R2R_XI_NM)


def kalman_update(d_prev, P_prev, y, Q, R):
    """Module 5: one Kalman update of the offset estimate for a random-walk offset. A missing observation (NaN) keeps the
    estimate and lets its variance grow. Returns (d, P)."""
    P_minus = P_prev + Q
    if np.isnan(y):
        return d_prev, P_minus
    K = P_minus / (P_minus + R)
    return d_prev + K * (y - d_prev), (1.0 - K) * P_minus


def next_time(d_hat, target_nm, B, BA, xi_nm, t_limits):
    """Module 5: the oxidation time (minutes) for which the Deal-Grove twin plus the offset estimate meets the target,
    clipped to the recipe's limits."""
    x = target_nm - d_hat
    A = B / BA
    t = (x ** 2 - xi_nm ** 2 + A * (x - xi_nm)) / B
    return min(max(t, t_limits[0]), t_limits[1])


def _float(v, name):
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size != 1:
        raise ValueError(f"{name}() must return one number, not {a.size}")
    return float(a[0])


def run_twin(key, scenario, predict_next, offset_observation):
    """Run the 60 production runs of furnace key ("drift" or "failing" scenario) with the twin in the loop.

    Before each run the twin recommends the oxidation time with next_time(), and predict_next(t_min, fit, d_hat) gives its
    prediction of that run's monitor-wafer thickness. After the run, offset_observation(measured_nm, t_min, fit) turns the
    measurement into an observation of the offset, and kalman_update() updates the estimate (starting at 0 nm with P = R:
    the calibration was made on this furnace, so it starts with no offset). The baseline is the calibrated Deal-Grove model
    alone, for the same times, with no offset estimate.

    Returns a DataFrame, one row per run: run, ox_time_min, predicted_nm (the twin, before the run), baseline_nm,
    measured_nm (NaN when not measured), error_nm and baseline_error_nm (measured minus prediction), d_hat_nm (after the run)."""
    setup = F.r2r_setup(key)
    fit = calibrate(key)
    furn = F.r2r_furnace(key, scenario)
    d_hat, P = 0.0, R_NM2
    rows = []
    for run in range(1, F.R2R_RUNS + 1):
        t = next_time(d_hat, setup["target_nm"], fit["B"], fit["BA"], F.R2R_XI_NM, setup["t_limits"])
        pred = _float(predict_next(t, fit, d_hat), "predict_next")
        base = float(predict(fit, t)["x"][0])
        y = furn.run(t)
        obs = _float(offset_observation(y, t, fit), "offset_observation")
        d_hat, P = kalman_update(d_hat, P, obs, Q_NM2, R_NM2)
        rows.append({"run": run, "ox_time_min": t, "predicted_nm": pred, "baseline_nm": base, "measured_nm": y,
                     "error_nm": y - pred, "baseline_error_nm": y - base, "d_hat_nm": d_hat})
    return pd.DataFrame(rows)


def runs_between(log, first_last):
    """The rows of a run_twin() log for runs first to last (inclusive)."""
    a, b = first_last
    return log[(log["run"] >= a) & (log["run"] <= b)]


def plot_log(log, title=None, tolerance=TOLERANCE_NM):
    """Measured and predicted monitor-wafer thickness against the run, and the twin's errors with the tolerance band."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(log["run"], log["measured_nm"], "o", ms=4, color="0.35", label="measured monitor wafer")
    axes[0].plot(log["run"], log["predicted_nm"], ".-", ms=4, lw=1.2, color="C0", label="twin (predicted before the run)")
    axes[0].plot(log["run"], log["baseline_nm"], "--", lw=1.2, color="C3", label="baseline: Deal-Grove alone")
    axes[0].set_ylabel("thickness (nm)")
    axes[1].axhspan(-tolerance, tolerance, color="C2", alpha=0.15, label=f"tolerance, +/- {tolerance:g} nm")
    axes[1].plot(log["run"], log["error_nm"], ".-", ms=4, lw=1, color="C0", label="twin error")
    axes[1].plot(log["run"], log["baseline_error_nm"], ".-", ms=3, lw=0.8, color="C3", label="baseline error")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("measured - predicted (nm)")
    axes[1].set_xlabel("production run")
    for ax in axes:
        ax.legend(fontsize=8, loc="best")
    if title:
        axes[0].set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()
