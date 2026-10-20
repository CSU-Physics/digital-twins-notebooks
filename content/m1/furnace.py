"""Simulated oxidation furnace for the SemiAcademy course Digital Twins for Semiconductor Manufacturing.

Every value this module produces is SIMULATED. It stands in for a real furnace so that every learner
gets a personal data set (set by their KEY) and every result can be checked.

Version 1.0.0 (Module 0): furnace_log(key), a zone-temperature log of one furnace run.
Version 1.1.0 (Module 1): temperature_stream(key), a reading stream with a disturbance the model does not know about.
furnace_log is unchanged, so Module 0 results stay the same. Later modules add oxidation, metrology, drift and a toolset.
"""
import numpy as np
import pandas as pd

__version__ = "1.1.0"

SETPOINT_C = 1000.0     # zone temperature the controller holds, in degrees C
LOAD_C = 800.0          # temperature of the zone when the wafers are pushed in
TAU_S = 150.0           # time constant of the warm-up, in seconds (Module 1 is about this number)
STEP_S = 10             # one reading every 10 seconds
DURATION_S = 1800       # a 30-minute log
HOLD_START_S = 1200     # from 20 minutes on, the zone is at its setpoint (the hold period)


def furnace_log(key):
    """A simulated log of one furnace run: one row every 10 s for 30 minutes.

    Columns: time_s (seconds from the push), run_id, setpoint_C, zone_temp_C (the thermocouple
    reading, rounded to 0.01 degrees C; NaN where the logger missed a reading).
    The same key always gives the same log.
    """
    key = int(key)
    rng = np.random.default_rng([2026, key])
    t = np.arange(0, DURATION_S + STEP_S, STEP_S)
    offset = rng.uniform(-0.5, 0.5)                 # this run settles a little above or below the setpoint
    sigma = rng.uniform(0.30, 0.60)                 # thermocouple noise, degrees C
    true = SETPOINT_C + offset - (SETPOINT_C - LOAD_C) * np.exp(-t / TAU_S)
    reading = np.round(true + rng.normal(0.0, sigma, t.size), 2)
    n_missing = int(rng.integers(3, 7))
    missing = rng.choice(np.arange(5, t.size), size=n_missing, replace=False)
    reading[missing] = np.nan
    return pd.DataFrame({"time_s": t, "run_id": f"R{key}", "setpoint_C": SETPOINT_C, "zone_temp_C": reading})


# ---------------------------------------------------------------- Module 1: a reading stream for the first twin
STREAM_FIT_END_S = 600      # Module 1 fits tau to the readings before 600 s (10 minutes)


def _dip_shape(s):
    """Shape of the disturbance, 0 before it starts, rising over about a minute and fading over several: peak value 1."""
    s = np.asarray(s, dtype=float)
    a, b = 40.0, 240.0
    s_peak = a * np.log((a + b) / a)
    peak = (1 - np.exp(-s_peak / a)) * np.exp(-s_peak / b)
    out = np.where(s > 0, (1 - np.exp(-np.clip(s, 0, None) / a)) * np.exp(-np.clip(s, 0, None) / b), 0.0)
    return out / peak


def _stream_params(key):
    rng = np.random.default_rng([2038, int(key)])
    params = {"tau_s": float(rng.uniform(120.0, 190.0)), "sigma_C": float(rng.uniform(0.30, 0.60)),
              "disturbance_start_s": int(rng.integers(84, 108)) * STEP_S, "disturbance_depth_C": float(rng.uniform(8.0, 16.0))}
    return params, rng


def stream_truth(key):
    """The hidden settings of temperature_stream(key): only a simulation can tell you these.
    Returns tau_s, sigma_C (thermocouple noise), disturbance_start_s and disturbance_depth_C (the peak cooling)."""
    return _stream_params(key)[0]


def temperature_stream(key):
    """A simulated stream of zone-temperature readings, one every 10 s for 30 minutes after the push.

    The zone warms from 800 C toward the 1000 C setpoint like a first-order system with time constant tau
    (between 120 and 190 s, set by the key). Some time after 14 minutes a disturbance the model does not know
    about (more purge gas, for example) cools the zone by several degrees, and the controller brings it back.
    A few readings are missing (NaN). Same columns as furnace_log; the same key always gives the same stream.
    """
    truth, rng = _stream_params(key)
    t = np.arange(0, DURATION_S + STEP_S, STEP_S)
    true = SETPOINT_C - (SETPOINT_C - LOAD_C) * np.exp(-t / truth["tau_s"])
    true = true - truth["disturbance_depth_C"] * _dip_shape(t - truth["disturbance_start_s"])
    reading = np.round(true + rng.normal(0.0, truth["sigma_C"], t.size), 2)
    n_missing = int(rng.integers(3, 7))
    missing = rng.choice(np.arange(5, t.size), size=n_missing, replace=False)
    reading[missing] = np.nan
    return pd.DataFrame({"time_s": t, "run_id": f"S{int(key)}", "setpoint_C": SETPOINT_C, "zone_temp_C": reading})
