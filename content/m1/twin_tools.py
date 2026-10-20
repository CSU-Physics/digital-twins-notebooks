"""Helpers for the Module 1 lab (Digital Twins for Semiconductor Manufacturing): a first-order model of the furnace
zone, a fitting function for its time constant, and a loop that runs a twin against a stream of readings.
The learner writes update_prediction(); everything here is supplied."""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from furnace import LOAD_C, SETPOINT_C, STEP_S

__version__ = "1.0.0"


def first_order(t, tau, T0=LOAD_C, T_set=SETPOINT_C):
    """Zone temperature of a first-order system that starts at T0 and approaches T_set with time constant tau (s)."""
    return T_set - (T_set - T0) * np.exp(-np.asarray(t, dtype=float) / tau)


def fit_tau(t, y, T0=LOAD_C, T_set=SETPOINT_C, tau_guess=150.0):
    """Least-squares fit of the time constant tau to readings y at times t (missing readings are left out).
    Returns (tau, standard error of tau), both in seconds. Uses scipy.optimize.curve_fit."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(y)
    popt, pcov = curve_fit(lambda tt, tau: first_order(tt, tau, T0, T_set), t[ok], y[ok], p0=[tau_guess])
    return float(popt[0]), float(np.sqrt(pcov[0, 0]))


def propagate(T, dt, tau, T_set=SETPOINT_C):
    """Move a predicted temperature T forward by dt seconds with the first-order model."""
    return T_set + (T - T_set) * np.exp(-dt / tau)


def run_twin(stream, tau, gain, update, every=1, T0=LOAD_C, T_set=SETPOINT_C):
    """Run a twin against a stream of readings.

    Before each reading arrives, the twin predicts it with the first-order model (column "predicted").
    Then, for every `every`-th reading, it corrects its state with update(predicted, reading, gain).
    With update=None the twin never looks at the readings: that is the model alone.
    Returns a table with time_s, reading, predicted, error (reading minus predicted) and state (after the update)."""
    t = stream["time_s"].to_numpy(dtype=float)
    y = stream["zone_temp_C"].to_numpy(dtype=float)
    T, rows = float(T0), []
    for k in range(t.size):
        if k > 0:
            T = float(propagate(T, t[k] - t[k - 1], tau, T_set))
        predicted = T
        if update is not None and k % every == 0:
            T = float(update(T, y[k], gain))
        rows.append((t[k], y[k], predicted, y[k] - predicted, T))
    return pd.DataFrame(rows, columns=["time_s", "reading", "predicted", "error", "state"])


def rms(x):
    """Root-mean-square of the values in x, leaving out missing values."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.sqrt(np.mean(x ** 2)))
