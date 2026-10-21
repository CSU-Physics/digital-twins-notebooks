"""Simulated oxidation furnace for the SemiAcademy course Digital Twins for Semiconductor Manufacturing.

Every value this module produces is SIMULATED. It stands in for a real furnace so that every learner
gets a personal data set (set by their KEY) and every result can be checked.

Version 1.0.0 (Module 0): furnace_log(key), a zone-temperature log of one furnace run.
Version 1.1.0 (Module 1): temperature_stream(key), a reading stream with a disturbance the model does not know about.
Version 1.2.0 (Module 2): fab_records(key), the records a fab keeps about 24 runs of the furnace: the MES run history,
the furnace's hold traces and the metrology results of the monitor wafers.
Earlier functions are unchanged, so Module 0 and Module 1 results stay the same. Later modules add drift and a toolset.
"""
import numpy as np
import pandas as pd

__version__ = "1.2.0"

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


# ---------------------------------------------------------------- Module 2: the fab's records of 24 runs
N_RUNS = 24                 # runs in the record set
RUN_CYCLE_H = 4             # a new run starts about every 4 hours
RAMP_UP_MIN = 30            # push and ramp to the setpoint in nitrogen (no oxide grows)
HOLD_MIN = 60               # oxidation hold in dry oxygen
RAMP_DOWN_MIN = 40          # ramp down and pull
TRACE_STEP_S = 10           # the furnace controller logs the zone temperature every 10 s during the hold
MONITOR_SLOTS = (5, 25, 45) # three monitor wafers per run: near the top, the middle and the bottom of the boat
ON_TIME_H = 12.0            # a twin that updates before the next lot is dispatched can use a result only if it is this fresh
START = pd.Timestamp("2026-03-02 06:00")
_K_EV = 8.617333e-5         # Boltzmann constant, eV/K


def _dry_oxide_nm(T_C, hold_h, xi_nm=1.5):
    """Deal-Grove thickness for dry O2 on (100) silicon (Deal and Grove 1965 constants, B/A divided by 1.68 for (100)).

    Module 2 uses it only to make the simulated thickness respond to the hold temperature; Module 3 studies the model.
    fab_records keeps using this function, so Module 2 results stay the same in later versions."""
    T = np.asarray(T_C, dtype=float) + 273.15
    B = 772.0 * np.exp(-1.23 / (_K_EV * T))                 # parabolic rate constant, um^2/h
    BA = 6.23e6 / 1.68 * np.exp(-2.0 / (_K_EV * T))         # linear rate constant, um/h
    A = B / BA
    xi = xi_nm / 1000.0
    tau = (xi ** 2 + A * xi) / B
    x_um = A / 2.0 * (np.sqrt(1.0 + 4.0 * B * (hold_h + tau) / A ** 2) - 1.0)
    return 1000.0 * x_um


def _fab_draw(key):
    rng = np.random.default_rng([2046, int(key)])
    n = N_RUNS
    start = START + pd.Timedelta(minutes=int(rng.integers(0, 60)))
    jitter = rng.integers(0, 40, n)                                   # minutes of waiting before each run
    starts = [start + pd.Timedelta(hours=RUN_CYCLE_H * i, minutes=int(jitter[i])) for i in range(n)]
    offset = rng.normal(0.0, 0.7, n)                                  # each run holds a little above or below 1000 C
    sigma = 0.35                                                      # thermocouple noise, C
    special = rng.choice(np.arange(3, n - 2), size=2, replace=False)  # one run with no metrology, one measured late
    codes = rng.choice(np.arange(100000, 1000000), size=n * len(MONITOR_SLOTS), replace=False)
    return rng, starts, offset, sigma, int(special[0]), int(special[1]), codes


def fab_truth(key):
    """The hidden settings of fab_records(key): only a simulation can tell you these.
    Returns hold_offset_C (one per run), missing_run and late_run (run IDs)."""
    rng, starts, offset, sigma, i_missing, i_late, codes = _fab_draw(key)
    return {"hold_offset_C": [round(float(v), 3) for v in offset],
            "missing_run": f"R{int(key)}-{i_missing + 1:02d}", "late_run": f"R{int(key)}-{i_late + 1:02d}"}


def fab_records(key):
    """The records a fab keeps about 24 simulated runs of the oxidation furnace (recipe DRYOX-60: 60 minutes in dry O2 at 1000 C).

    Returns a dict of four DataFrames, each written by a different system:
      "runs"      MES run history, one row per run: run_id, tool, recipe, start_time, hold_start, end_time.
      "wafers"    MES wafer history, one row per monitor wafer: run_id, slot, wafer_id (three per run).
      "trace"     furnace controller trace of each hold, one row every 10 s: run_id, time_s (seconds from the
                  start of the hold), zone_temp_C (NaN where the controller missed a reading).
      "metrology" ellipsometer results, one row per measured wafer, in the order they were written:
                  wafer_id, measured_at, thickness_nm. It knows the wafer, not the run.
    In each record set the metrology of one run is missing and the metrology of another run arrived late.
    The same key always gives the same records.
    """
    key = int(key)
    rng, starts, offset, sigma, i_missing, i_late, codes = _fab_draw(key)
    run_ids = [f"R{key}-{i + 1:02d}" for i in range(N_RUNS)]
    hold_start = [s + pd.Timedelta(minutes=RAMP_UP_MIN) for s in starts]
    end = [s + pd.Timedelta(minutes=RAMP_UP_MIN + HOLD_MIN + RAMP_DOWN_MIN) for s in starts]
    runs = pd.DataFrame({"run_id": run_ids, "tool": "FURN-03", "recipe": "DRYOX-60",
                         "start_time": starts, "hold_start": hold_start, "end_time": end})

    t = np.arange(0, HOLD_MIN * 60, TRACE_STEP_S)
    traces = []
    for i, rid in enumerate(run_ids):
        z = np.round(SETPOINT_C + offset[i] + rng.normal(0.0, sigma, t.size), 2)
        z[rng.choice(t.size, size=int(rng.integers(2, 6)), replace=False)] = np.nan
        traces.append(pd.DataFrame({"run_id": rid, "time_s": t, "zone_temp_C": z}))
    trace = pd.concat(traces, ignore_index=True)

    wafer_rows, met_rows = [], []
    slot_temp = {5: -0.3, 25: 0.0, 45: 0.3}            # the boat is a little cooler at the top, warmer at the bottom
    for i, rid in enumerate(run_ids):
        run_extra = rng.normal(0.0, 0.15)              # run-to-run variation the hold temperature does not explain
        if i == i_late:
            delay = rng.uniform(26.0, 40.0)            # the ellipsometer was down: measured more than a day later
        else:
            delay = rng.uniform(1.5, 6.0)
        for j, slot in enumerate(MONITOR_SLOTS):
            wid = f"W{codes[i * len(MONITOR_SLOTS) + j]}"
            wafer_rows.append({"run_id": rid, "slot": slot, "wafer_id": wid})
            if i == i_missing:
                continue                               # these results never reached the database
            x = _dry_oxide_nm(SETPOINT_C + offset[i] + slot_temp[slot], HOLD_MIN / 60.0) + run_extra + rng.normal(0.0, 0.20)
            when = end[i] + pd.Timedelta(hours=float(delay)) + pd.Timedelta(minutes=4 * j)
            met_rows.append({"wafer_id": wid, "measured_at": when.floor("s"), "thickness_nm": round(float(x + rng.normal(0.0, 0.15)), 2)})
    wafers = pd.DataFrame(wafer_rows)
    metrology = pd.DataFrame(met_rows).sort_values("measured_at", kind="stable").reset_index(drop=True)
    return {"runs": runs, "wafers": wafers, "trace": trace, "metrology": metrology}
