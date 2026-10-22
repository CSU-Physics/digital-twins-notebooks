"""Simulated oxidation furnace for the SemiAcademy course Digital Twins for Semiconductor Manufacturing.

Every value this module produces is SIMULATED. It stands in for a real furnace so that every learner
gets a personal data set (set by their KEY) and every result can be checked.

Version 1.0.0 (Module 0): furnace_log(key), a zone-temperature log of one furnace run.
Version 1.1.0 (Module 1): temperature_stream(key), a reading stream with a disturbance the model does not know about.
Version 1.2.0 (Module 2): fab_records(key), the records a fab keeps about 24 runs of the furnace: the MES run history,
the furnace's hold traces and the metrology results of the monitor wafers.
Version 1.3.0 (Module 3): textbook_rates() and oxidation_runs(key), oxide thickness measured after dry oxidation for
different times at 1000 C, including a thin-oxide run that grows faster than the Deal-Grove model predicts.
Version 1.4.0 (Module 4): vm_runs(key), 67 production runs of a dry-oxidation recipe with the furnace's trace summaries
for each wafer and the measured thickness, for virtual metrology.
Version 1.5.0 (Module 5): r2r_setup(key) and r2r_furnace(key, scenario), the DRYOX-90 recipe run by run while the
thermocouple bias slowly grows, for run-to-run control.
Earlier functions are unchanged, so earlier results stay the same. Later modules add a toolset.
"""
import numpy as np
import pandas as pd

__version__ = "1.5.0"

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


# ---------------------------------------------------------------- Module 3: oxidation runs for calibrating Deal-Grove
OX_T_C = 1000.0             # every calibration run is at 1000 C in dry O2
OX_XI_NM = 25.0             # the calibration wafers carry an oxide of 25.0 nm, grown and measured before these runs
OX_SIGMA_NM = 0.5           # standard deviation of one ellipsometer measurement, stated by the metrology group
OX_TIMES_MIN = (30, 60, 90, 120, 150, 180, 210, 240, 300, 360, 420, 480)
OX_NEW_MIN = 270            # a new run, inside the calibrated range, to be predicted
THIN_XI_NM = 1.5            # bare wafers: only the native oxide
THIN_TIMES_MIN = (10, 20, 30, 40)
THIN_L_NM = 7.0             # decay length of the extra early growth (the thin-oxide regime)
_STEP_MIN = 0.05


def textbook_rates(T_C, ambient="dry", orientation="100"):
    """Deal and Grove (1965) rate constants at temperature T_C, converted to nm and minutes.

    Returns (B, BA): the parabolic rate constant B in nm^2/min and the linear rate constant B/A in nm/min.
    ambient "dry" (O2) or "wet" (H2O at 640 torr); orientation "111" as in the 1965 table, or "100", for which B/A
    is divided by 1.68. These are textbook values; a real furnace, and the course's simulated one, differ from them."""
    T = float(T_C) + 273.15
    if ambient == "dry":
        c1, e1, c2, e2 = 7.72e2, 1.23, 6.23e6, 2.00
    elif ambient == "wet":
        c1, e1, c2, e2 = 3.86e2, 0.78, 1.63e8, 2.05
    else:
        raise ValueError('ambient must be "dry" or "wet"')
    if orientation == "100":
        c2 = c2 / 1.68
    elif orientation != "111":
        raise ValueError('orientation must be "100" or "111"')
    B = c1 * np.exp(-e1 / (_K_EV * T)) * 1e6 / 60.0          # um^2/h to nm^2/min
    BA = c2 * np.exp(-e2 / (_K_EV * T)) * 1e3 / 60.0         # um/h to nm/min
    return float(B), float(BA)


def _ox_draw(key):
    rng = np.random.default_rng([2054, int(key)])
    B0, BA0 = textbook_rates(OX_T_C)
    truth = {"B": B0 * float(rng.uniform(0.92, 1.08)), "BA": BA0 * float(rng.uniform(0.88, 1.12)),
             "thin_C": float(rng.uniform(1.2, 1.8)), "thin_L": THIN_L_NM}
    return truth, rng


def oxidation_truth(key):
    """The hidden settings of oxidation_runs(key): only a simulation can tell you these.
    B (nm^2/min) and BA (nm/min) of this furnace's Deal-Grove growth, and thin_C (nm/min) and thin_L (nm) of the extra
    early growth C * exp(-x / L) that the simulator adds to the Deal-Grove rate, which matters only below about 20 nm."""
    return _ox_draw(key)[0]


def _grow(times, xi, B, BA, C, L):
    """Integrate dx/dt = B / (2x + A) + C exp(-x/L) from x = xi at t = 0 (fourth-order Runge-Kutta, step 0.05 min)."""
    A = B / BA
    rate = lambda x: B / (2.0 * x + A) + C * np.exp(-x / L)
    want = sorted(set(float(t) for t in times))
    n = int(round(want[-1] / _STEP_MIN))
    marks = {int(round(t / _STEP_MIN)): t for t in want}
    out, x, h = {}, float(xi), _STEP_MIN
    for i in range(n + 1):
        if i in marks:
            out[marks[i]] = x
        k1 = rate(x); k2 = rate(x + h / 2 * k1); k3 = rate(x + h / 2 * k2); k4 = rate(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array([out[float(t)] for t in times])


def oxidation_runs(key):
    """Oxide thickness measured after dry oxidation at 1000 C in the course's simulated furnace.

    Returns a dict of three DataFrames with the columns run_id, time_min, xi_nm (oxide before the run), thickness_nm
    (one ellipsometer measurement, rounded to 0.01 nm, standard deviation OX_SIGMA_NM):
      "calibration"  12 runs of 30 to 480 minutes on wafers with a 25.0 nm starting oxide;
      "new"          3 wafers of one new run of 270 minutes, with the same starting oxide;
      "thin"         4 runs of 10 to 40 minutes on bare wafers (1.5 nm of native oxide): the thin-oxide regime, where
                     the simulator, like real dry oxidation, grows oxide faster than the Deal-Grove model.
    The same key always gives the same runs.
    """
    key = int(key)
    truth, rng = _ox_draw(key)
    B, BA, C, L = truth["B"], truth["BA"], truth["thin_C"], truth["thin_L"]
    sets = {}
    cal_t = np.array(OX_TIMES_MIN, dtype=float)
    cal_x = _grow(cal_t, OX_XI_NM, B, BA, C, L)
    sets["calibration"] = pd.DataFrame({"run_id": [f"OX{key}-{i + 1:02d}" for i in range(cal_t.size)], "time_min": cal_t,
                                        "xi_nm": OX_XI_NM, "thickness_nm": np.round(cal_x + rng.normal(0.0, OX_SIGMA_NM, cal_t.size), 2)})
    new_x = _grow([OX_NEW_MIN], OX_XI_NM, B, BA, C, L)[0]
    sets["new"] = pd.DataFrame({"run_id": f"OX{key}-NEW", "time_min": float(OX_NEW_MIN), "xi_nm": OX_XI_NM,
                                "thickness_nm": np.round(new_x + rng.normal(0.0, OX_SIGMA_NM, 3), 2)})
    thin_t = np.array(THIN_TIMES_MIN, dtype=float)
    thin_x = _grow(thin_t, THIN_XI_NM, B, BA, C, L)
    sets["thin"] = pd.DataFrame({"run_id": [f"OX{key}-T{i + 1}" for i in range(thin_t.size)], "time_min": thin_t,
                                 "xi_nm": THIN_XI_NM, "thickness_nm": np.round(thin_x + rng.normal(0.0, OX_SIGMA_NM, thin_t.size), 2)})
    return sets


# ---------------------------------------------------------------- Module 4: production runs for virtual metrology
VM_RUNS = 60                # measured production runs of recipe DRYOX-90, in time order
VM_TRAIN_RUNS = 40          # the first 40 runs train and tune the models; runs 41 to 60 are the time-ordered test set
VM_SHIFT_RUNS = 6           # six later runs of a new recipe, DRYOX-140: a condition outside the training range
VM_SHIFT_MIN = 140.0        # oxidation time of the new recipe, minutes
VM_TIME_RANGE = (80, 100)   # the run-to-run controller sets each DRYOX-90 run's oxidation time in this range, minutes
VM_SLOTS = (5, 35, 65, 95, 125)  # five sampled wafers per run, from the top (slot 1) to the bottom (slot 125) of the boat
VM_PENDING_SLOT = 65        # the wafer of the pending run whose thickness the lab predicts
VM_XI_NM = 1.5              # product wafers enter with their native oxide only
VM_FEATURES = ["ox_time_min", "T_top_C", "T_center_C", "T_bottom_C", "h2o_ppm", "o2_slm", "slot"]
_VM_ZONE_SLOTS = (1, 63, 125)    # the three heater zones and their thermocouples: top, center, bottom of the boat
_VM_ZONE_OFFSET = (-0.4, 0.0, 0.3)
_VM_H2O_PER_PPM = 0.006     # each ppm of moisture in the oxygen speeds both rate constants by 0.6 % (simulated)
_VM_DRIFT_C_PER_RUN = 0.012 # the thermocouples slowly read high: the true temperature falls by this much per run


def _rates_array(T_C):
    """textbook_rates() for dry O2 on (100) silicon, for an array of temperatures: (B in nm^2/min, B/A in nm/min)."""
    T = np.asarray(T_C, dtype=float) + 273.15
    B = 7.72e2 * np.exp(-1.23 / (_K_EV * T)) * 1e6 / 60.0
    BA = 6.23e6 / 1.68 * np.exp(-2.00 / (_K_EV * T)) * 1e3 / 60.0
    return B, BA


def _grow_many(t_min, xi, B, BA, C, L):
    """_grow() for many wafers at once: each wafer i grows from xi for t_min[i] minutes with its own B[i] and BA[i]."""
    t_min = np.asarray(t_min, dtype=float)
    A = B / BA
    rate = lambda x: B / (2.0 * x + A) + C * np.exp(-x / L)
    stop = np.round(t_min / _STEP_MIN).astype(int)
    x, h = np.full(t_min.shape, float(xi)), _STEP_MIN
    out = np.full(t_min.shape, np.nan)
    for i in range(int(stop.max()) + 1):
        done = stop == i
        if done.any():
            out[done] = x[done]
        k1 = rate(x); k2 = rate(x + h / 2 * k1); k3 = rate(x + h / 2 * k2); k4 = rate(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def _vm_draw(key):
    rng = np.random.default_rng([2062, int(key)])
    truth = {"fB": float(rng.uniform(0.92, 1.08)), "fBA": float(rng.uniform(0.88, 1.12)), "thin_C": float(rng.uniform(1.2, 1.8)),
             "thin_L": THIN_L_NM, "tc_offset_C": float(rng.uniform(-1.0, 1.0)), "h2o_per_ppm": _VM_H2O_PER_PPM,
             "drift_C_per_run": _VM_DRIFT_C_PER_RUN, "run_sd_nm": 0.35, "wafer_sd_nm": 0.20, "metrology_sd_nm": 0.30}
    return truth, rng


def vm_truth(key):
    """The hidden settings of vm_runs(key): only a simulation can tell you these.
    fB and fBA scale the textbook B and B/A of this furnace; thin_C and thin_L set its extra early growth (Module 3);
    tc_offset_C is how far the true temperature sits from the thermocouple readings (true minus reading) at the first run;
    h2o_per_ppm is the speed-up of both rate constants per ppm of moisture; drift_C_per_run is how fast the true temperature
    falls below the readings as the thermocouples age; run_sd_nm, wafer_sd_nm and metrology_sd_nm are the standard
    deviations of the run-to-run effects no trace explains, of wafer-to-wafer effects and of one measurement."""
    return _vm_draw(key)[0]


def vm_runs(key):
    """Production runs of the course's simulated furnace, for virtual metrology.

    Recipe DRYOX-90 oxidizes product wafers (native oxide only) in dry O2 at 1000 C. The run-to-run controller sets
    each run's oxidation time between 80 and 100 minutes. For each run the furnace logs a summary of its traces, and the
    metrology tool measures five sampled wafers (slots 5, 35, 65, 95 and 125 of the boat).

    Returns a dict of three DataFrames, one row per wafer, in time order:
      "measured"  runs 1 to 60 of DRYOX-90 with their measured thickness;
      "shifted"   runs 61 to 66 of a new recipe, DRYOX-140, with 140 minutes of oxidation: outside the training range;
      "pending"   run 67 of DRYOX-90, just finished: its wafers are not measured, so it has no thickness column.
    Columns: run_id, run_index, recipe, slot, ox_time_min (oxidation time), T_top_C, T_center_C, T_bottom_C (mean
    thermocouple reading of each heater zone during the oxidation step), h2o_ppm (moisture in the oxygen), o2_slm (oxygen
    flow, standard litres per minute) and thickness_nm (one ellipsometer measurement, rounded to 0.01 nm).
    The same key always gives the same runs.
    """
    key = int(key)
    truth, rng = _vm_draw(key)
    n = VM_RUNS + VM_SHIFT_RUNS + 1
    t_ox = np.concatenate([rng.integers(VM_TIME_RANGE[0], VM_TIME_RANGE[1] + 1, VM_RUNS).astype(float),
                           np.full(VM_SHIFT_RUNS, VM_SHIFT_MIN), rng.integers(85, 96, 1).astype(float)])
    run_off = rng.normal(0.0, 0.8, n)                                   # each run holds a little above or below 1000 C
    zones = run_off[:, None] + rng.normal(0.0, 0.15, (n, 3)) + np.array(_VM_ZONE_OFFSET)
    h2o = np.round(rng.uniform(2.0, 12.0, n), 1)
    o2 = np.round(rng.uniform(9.5, 10.5, n), 2)
    run_eff = rng.normal(0.0, truth["run_sd_nm"], n)
    rows = []
    for i in range(n):
        readings = np.round(SETPOINT_C + zones[i], 2)
        recipe = "DRYOX-90" if (i < VM_RUNS or i == n - 1) else "DRYOX-140"
        for s in VM_SLOTS:
            T_read = float(np.interp(s, _VM_ZONE_SLOTS, readings))
            rows.append({"run_id": f"VM{key}-{i + 1:02d}", "run_index": i + 1, "recipe": recipe, "slot": s,
                         "ox_time_min": t_ox[i], "T_top_C": readings[0], "T_center_C": readings[1], "T_bottom_C": readings[2],
                         "h2o_ppm": h2o[i], "o2_slm": o2[i],
                         "_T_true": T_read + truth["tc_offset_C"] - truth["drift_C_per_run"] * i, "_run_eff": run_eff[i]})
    df = pd.DataFrame(rows)
    B, BA = _rates_array(df["_T_true"].to_numpy())
    speed = 1.0 + truth["h2o_per_ppm"] * df["h2o_ppm"].to_numpy()
    x = _grow_many(df["ox_time_min"].to_numpy(), VM_XI_NM, B * truth["fB"] * speed, BA * truth["fBA"] * speed,
                   truth["thin_C"], truth["thin_L"])
    x = x + df["_run_eff"].to_numpy() + rng.normal(0.0, truth["wafer_sd_nm"], len(df))
    df["thickness_nm"] = np.round(x + rng.normal(0.0, truth["metrology_sd_nm"], len(df)), 2)
    df = df.drop(columns=["_T_true", "_run_eff"])
    out = {"measured": df[df["run_index"] <= VM_RUNS].reset_index(drop=True),
           "shifted": df[(df["run_index"] > VM_RUNS) & (df["run_index"] <= VM_RUNS + VM_SHIFT_RUNS)].reset_index(drop=True),
           "pending": df[df["run_index"] == n].drop(columns=["thickness_nm"]).reset_index(drop=True)}
    return out


# ---------------------------------------------------------------- Module 5: run-to-run control of a drifting furnace
R2R_RUNS = 60               # production runs after the qualification run (run 0)
R2R_QUAL_MIN = 90.0         # oxidation time of the qualification run, minutes
R2R_TIME_LIMITS = (80.0, 100.0)   # the recipe allows oxidation times from 80 to 100 minutes (the range Module 4 saw)
R2R_XI_NM = 1.5             # product and monitor wafers enter with their native oxide only
R2R_SCENARIOS = ("drift", "failing")
_R2R_GRID = (75.0, 105.0)   # the simulator precomputes each run's growth curve over this time window, minutes


def _r2r_draw(key):
    rng = np.random.default_rng([2070, int(key)])
    B0, BA0 = textbook_rates(SETPOINT_C)
    truth = {"fB": float(rng.uniform(0.92, 1.08)), "fBA": float(rng.uniform(0.88, 1.12)),
             "thin_C": float(rng.uniform(1.2, 1.8)), "thin_L": THIN_L_NM, "tc_offset_C": float(rng.uniform(-0.5, 0.5)),
             "bias_rate_C_per_run": float(rng.uniform(0.06, 0.09)), "bias_walk_sd_C": 0.10, "run_T_sd_C": 0.45,
             "monitor_sd_nm": 0.32, "fail_start_run": int(rng.integers(25, 36)), "fail_rate_C_per_run": 0.35,
             "missing_run": int(rng.integers(20, 46))}
    twin = {"B": round(B0 * truth["fB"] * float(rng.uniform(0.99, 1.01)), 2),
            "BA": round(BA0 * truth["fBA"] * float(rng.uniform(0.99, 1.01)), 4)}
    n = R2R_RUNS + 1
    walk = np.concatenate([[0.0], np.cumsum(rng.normal(0.0, truth["bias_walk_sd_C"], n - 1))])
    bias = truth["bias_rate_C_per_run"] * np.arange(n) + walk
    run_T = rng.normal(0.0, truth["run_T_sd_C"], n)
    monitor = rng.normal(0.0, truth["monitor_sd_nm"], n)
    return truth, twin, bias, run_T, monitor


def r2r_truth(key):
    """The hidden settings of r2r_furnace(key): only a simulation can tell you these.
    fB, fBA, thin_C and thin_L describe this furnace's growth (as in Module 3); tc_offset_C is how far the true temperature
    sits from the reading at the qualification run; the thermocouple bias b (reading minus true temperature) grows by
    bias_rate_C_per_run plus a random walk with steps of bias_walk_sd_C; run_T_sd_C is the run-to-run temperature noise;
    monitor_sd_nm is the standard deviation of one monitor-wafer measurement around the run's mean thickness. In the
    "failing" scenario the bias grows faster by fail_rate_C_per_run from run fail_start_run on. missing_run has no
    measurement. bias_C lists b for runs 0 to 60 of each scenario."""
    truth, twin, bias, run_T, monitor = _r2r_draw(key)
    out = dict(truth)
    out["bias_C"] = {s: _r2r_bias(truth, bias, s).round(4).tolist() for s in R2R_SCENARIOS}
    return out


def _r2r_bias(truth, bias, scenario):
    if scenario == "drift":
        return bias.copy()
    if scenario == "failing":
        k = np.arange(bias.size)
        return bias + truth["fail_rate_C_per_run"] * np.clip(k - truth["fail_start_run"], 0, None)
    raise ValueError('scenario must be "drift" or "failing"')


_R2R_CACHE = {}


def _r2r_curves(key, scenario):
    """Each run's mean thickness over the time window, from its true temperature (cached per key and scenario)."""
    ck = (int(key), scenario)
    if ck not in _R2R_CACHE:
        truth, twin, bias, run_T, monitor = _r2r_draw(key)
        b = _r2r_bias(truth, bias, scenario)
        T_true = SETPOINT_C + truth["tc_offset_C"] - b + run_T
        B, BA = _rates_array(T_true)
        B, BA = B * truth["fB"], BA * truth["fBA"]
        A = B / BA
        C, L = truth["thin_C"], truth["thin_L"]
        rate = lambda x: B / (2.0 * x + A) + C * np.exp(-x / L)
        h = _STEP_MIN
        i0, i1 = int(round(_R2R_GRID[0] / h)), int(round(_R2R_GRID[1] / h))
        x = np.full(B.shape, R2R_XI_NM)
        curves = np.empty((B.size, i1 - i0 + 1))
        for i in range(i1 + 1):
            if i >= i0:
                curves[:, i - i0] = x
            k1 = rate(x); k2 = rate(x + h / 2 * k1); k3 = rate(x + h / 2 * k2); k4 = rate(x + h * k3)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        grid = np.arange(i0, i1 + 1) * h
        _R2R_CACHE[ck] = (grid, curves, monitor, truth["missing_run"])
    return _R2R_CACHE[ck]


def r2r_setup(key):
    """What the process engineer knows about the furnace before production starts (recipe DRYOX-90, dry O2, 1000 C).

    Returns a dict: target_nm (the thickness the recipe must reach, nm), B and BA (the twin's Deal-Grove constants for
    this furnace, nm^2/min and nm/min, calibrated as in Module 3), xi_nm (native oxide, nm), t_limits (allowed oxidation
    times, minutes), qual_time_min and qual_thickness_nm (the qualification run 0 and its monitor-wafer measurement)."""
    truth, twin, bias, run_T, monitor = _r2r_draw(key)
    grid, curves, monitor, missing = _r2r_curves(key, "drift")
    # the target is the mean thickness this furnace grows in 90 minutes with no bias and no run noise
    T0 = SETPOINT_C + truth["tc_offset_C"]
    B, BA = _rates_array(np.array([T0]))
    target = float(_grow_many([R2R_QUAL_MIN], R2R_XI_NM, B * truth["fB"], BA * truth["fBA"], truth["thin_C"], truth["thin_L"])[0])
    qual = float(np.interp(R2R_QUAL_MIN, grid, curves[0])) + monitor[0]
    return {"target_nm": round(target, 1), "B": twin["B"], "BA": twin["BA"], "xi_nm": R2R_XI_NM,
            "t_limits": R2R_TIME_LIMITS, "qual_time_min": R2R_QUAL_MIN, "qual_thickness_nm": round(qual, 2)}


class r2r_furnace:
    """The simulated furnace in production, one run at a time: f = r2r_furnace(key, "drift"); y = f.run(92.5).

    scenario "drift": the thermocouple bias grows slowly; "failing": from a run between 25 and 35 the bias grows much
    faster. Every r2r_furnace with the same key and scenario has the same disturbances (bias, temperature noise and
    measurement noise of each run), so different controllers can be compared on the same sequence.
    run(ox_time_min) processes the next run (1, 2, ... 60) with that oxidation time and returns the monitor wafer's
    measured thickness in nm (rounded to 0.01 nm), or NaN when the run has no measurement.
    achieved holds the true mean thickness of each run so far: only a simulation can tell you this."""

    def __init__(self, key, scenario="drift"):
        if scenario not in R2R_SCENARIOS:
            raise ValueError('scenario must be "drift" or "failing"')
        self.key, self.scenario = int(key), scenario
        self._grid, self._curves, self._monitor, self._missing = _r2r_curves(key, scenario)
        self.run_index = 0
        self.achieved = []

    def run(self, ox_time_min):
        t = float(ox_time_min)
        if not (self._grid[0] <= t <= self._grid[-1]):
            raise ValueError(f"oxidation time {t} min is outside the simulator's window of {_R2R_GRID[0]:g} to {_R2R_GRID[1]:g} min")
        if self.run_index >= R2R_RUNS:
            raise RuntimeError(f"this furnace has run all {R2R_RUNS} production runs")
        self.run_index += 1
        k = self.run_index
        x = float(np.interp(t, self._grid, self._curves[k]))
        self.achieved.append(x)
        if k == self._missing:
            return float("nan")
        return round(x + float(self._monitor[k]), 2)
