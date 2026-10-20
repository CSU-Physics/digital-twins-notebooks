"""Simulated oxidation furnace for the SemiAcademy course Digital Twins for Semiconductor Manufacturing.

Every value this module produces is SIMULATED. It stands in for a real furnace so that every learner
gets a personal data set (set by their KEY) and every result can be checked.

Version 1.0.0 (Module 0): furnace_log(key), a zone-temperature log of one furnace run.
Later modules add the oxidation, metrology, drift and toolset models to this file.
"""
import numpy as np
import pandas as pd

__version__ = "1.0.0"

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
