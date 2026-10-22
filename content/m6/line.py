"""The Module 6 line model (Digital Twins for Semiconductor Manufacturing): a furnace, lithography and etch line in SimPy.

Everything here is SIMULATED. Lots of 25 wafers are released at random times and visit three tool groups in order:
furnace (thermal oxidation), lithography, etch. A lot waits for a free tool, is processed, and moves on. There is one
product and no re-entrant flow: a real fab sends the same lot through lithography and etch many times.

run_line(cfg, seed, hours)        one replication of the line: a LineRun with the lot log and the calibration log
BASE_LINE                         the line's settings (a dict): arrival rate, tool counts, mean process times, maintenance
line_config(**changes)            a copy of BASE_LINE with some settings changed
LineRun                           .lots (one row per lot), .calibrations (one row per furnace calibration), .hours, .cfg, .seed

Times are in hours. The furnace tubes carry the drift of Module 5: a thermocouple bias that grows with every run, an estimate
of the resulting thickness offset from one monitor wafer per run (the Module 5 Kalman filter), and an optional calibration.
SimPy (MIT license) is imported from the wheel next to this file, or from the installed package.
"""

import glob
import math
import os
import sys

import numpy as np
import pandas as pd

__version__ = "1.0.0"


def _import_simpy():
    """Import SimPy from the wheel in this folder when there is one, else from the installed packages."""
    here = os.path.dirname(os.path.abspath(__file__)) or "."
    for wheel in sorted(glob.glob(os.path.join(here, "simpy-*.whl")) + glob.glob("simpy-*.whl")):
        if wheel not in sys.path:
            sys.path.insert(0, os.path.abspath(wheel))
    try:
        import simpy
        return simpy
    except ImportError as e:  # pragma: no cover
        raise ImportError("SimPy is not available: keep the simpy-4.1.2 wheel next to this notebook, or install it "
                          "with pip install simpy") from e


simpy = _import_simpy()

STAGES = ("furnace", "litho", "etch")

# --- the line (all times in hours) -------------------------------------------------------------------------------------
BASE_LINE = {
    "arrival_rate": 0.85,        # lots released per hour (random, exponential gaps)
    "furnace_tools": 2, "litho_tools": 1, "etch_tools": 2,
    "furnace_h": 1.8,            # mean time a lot spends in a furnace tube: load, 90-minute oxidation, unload
    "litho_h": 1.0,              # mean time on a lithography tool
    "etch_h": 1.2,               # mean time in an etch chamber
    "furnace_cv": 0.30, "litho_cv": 0.50, "etch_cv": 0.40,   # coefficient of variation of the process time (gamma distributed)
    "maintenance": "none",       # "none", "preventive" (calibrate every interval_runs runs) or "predictive" (calibrate when the drift estimate says so)
    "interval_runs": 50,         # preventive: furnace runs between calibrations of one tube
    "limit_nm": 2.0, "k_runs": 3,  # predictive: the Module 5 alarm, the estimate beyond limit_nm for k_runs runs in a row
    "calibration_h": 8.0,        # a calibration takes one tube out of service this long
}

# --- the furnace drift of Module 5, per tube ------------------------------------------------------------------------------
NM_PER_C = 0.52          # thickness change per degree of thermocouple bias, 90-minute dry oxidation at 1000 C (Module 5)
RATE_C = (0.06, 0.09)    # bias growth per run, drawn once per tube (Module 5, drift scenario)
WALK_C = 0.10            # random step of the bias per run (degrees)
FAULT_RATE_C = 0.35      # extra bias growth per run once a thermocouple starts to fail (Module 5, failing scenario)
FAULT_PROB = 1.0 / 150   # chance per run that a healthy thermocouple starts to fail
COMP_LIMIT_NM = 5.0      # drift the run-to-run controller can still correct: the oxidation time reaches its 100-minute limit
Q_NM2 = 0.01             # Module 5 Kalman filter: variance of the offset's random step per run (nm^2)
R_NM2 = 0.16             # variance of one monitor-wafer observation around the offset (nm^2)


def line_config(**changes):
    """BASE_LINE with the given settings changed, for example line_config(litho_tools=2, litho_h=0.9)."""
    unknown = sorted(set(changes) - set(BASE_LINE))
    if unknown:
        raise KeyError(f"unknown setting(s) {unknown}; the settings are {sorted(BASE_LINE)}")
    cfg = dict(BASE_LINE)
    cfg.update(changes)
    return cfg


def _check_config(cfg):
    if cfg["arrival_rate"] <= 0:
        raise ValueError("arrival_rate must be above 0 lots per hour")
    for s in STAGES:
        if int(cfg[s + "_tools"]) < 1 or cfg[s + "_h"] <= 0 or cfg[s + "_cv"] <= 0:
            raise ValueError(f"{s}: tools must be at least 1, and the mean time and its variation above 0")
    if cfg["maintenance"] not in ("none", "preventive", "predictive"):
        raise ValueError('maintenance must be "none", "preventive" or "predictive"')


class LineRun:
    """One replication. lots: one row per lot that entered the line; NaN where the lot had not got that far when the run ended.
    calibrations: one row per furnace calibration (tube, t_start, t_end, reason, runs, drift_nm)."""

    def __init__(self, lots, calibrations, hours, cfg, seed):
        self.lots, self.calibrations, self.hours, self.cfg, self.seed = lots, calibrations, hours, cfg, seed

    def __repr__(self):
        done = int(self.lots["t_out"].notna().sum())
        return f"LineRun(seed={self.seed}, hours={self.hours}, lots in {len(self.lots)}, lots out {done}, calibrations {len(self.calibrations)})"


class _Tube:
    """One furnace tube: its thermocouple bias, and the Module 5 estimate of the thickness offset that bias causes."""

    def __init__(self, tid, rng, hours):
        self.id = tid
        m = int(hours / 0.5) + 80                     # more runs than the tube can make
        self.rate = rng.uniform(*RATE_C)
        self.walk = rng.normal(0.0, WALK_C, m)
        self.noise = rng.normal(0.0, math.sqrt(R_NM2), m)
        self.fault_u = rng.random(m)
        self.i = 0
        self.reset()

    def reset(self):
        self.bias = 0.0
        self.failing = False
        self.fault_at = None                          # the run (since the last calibration) in which the thermocouple started to fail
        self.runs = 0                                 # runs since the last calibration
        self.d_hat = 0.0                              # estimate of the drift (nm), from the Kalman filter
        self.P = R_NM2
        self.recent = []                              # the estimates since the last calibration

    def run(self):
        """One furnace run: the bias grows, the monitor wafer is read, the estimate is updated. Returns the true drift (nm)."""
        i = self.i
        self.i += 1
        if not self.failing and self.fault_u[i] < FAULT_PROB:
            self.failing = True
            self.fault_at = self.runs + 1
        self.bias += self.rate + (FAULT_RATE_C if self.failing else 0.0) + self.walk[i]
        drift = NM_PER_C * self.bias
        y = drift + self.noise[i]
        P_minus = self.P + Q_NM2                      # Module 5: Kalman filter for a random-walk offset
        K = P_minus / (P_minus + R_NM2)
        self.d_hat += K * (y - self.d_hat)
        self.P = (1.0 - K) * P_minus
        self.runs += 1
        self.recent.append(self.d_hat)
        return drift


def run_line(cfg, seed, hours, trigger=None):
    """Simulate the line for `hours` hours, starting empty, and return a LineRun.

    Replications with the same seed share their random numbers whatever the settings: the same lots arrive at the same
    relative times and draw the same unit process times, so two settings are compared on the same stream of lots.
    trigger(recent, limit_nm, k_runs): used when cfg["maintenance"] is "predictive"; it receives one tube's drift estimates
    since its last calibration (a list, the latest last) and returns True when that tube should be calibrated now."""
    cfg = {**BASE_LINE, **cfg}
    _check_config(cfg)
    if cfg["maintenance"] == "predictive" and trigger is None:
        raise ValueError("maintenance 'predictive' needs your rule: run_line(cfg, seed, hours, trigger=calibrate_now)")
    n_draw = int(2.0 * hours) + 200                  # more lots than any setting of the lab can release
    lam = cfg["arrival_rate"]
    t_in = np.cumsum(np.random.default_rng([6060, int(seed), 1]).exponential(1.0, n_draw)) / lam
    if t_in[-1] < hours:
        raise ValueError("arrival_rate is too high for this run length")
    t_in = t_in[t_in < hours]
    n = len(t_in)
    unit = {}
    for j, s in enumerate(STAGES):
        shape = 1.0 / cfg[s + "_cv"] ** 2
        unit[s] = np.random.default_rng([6060, int(seed), 2 + j]).gamma(shape, 1.0 / shape, n_draw)[:n] * cfg[s + "_h"]

    env = simpy.Environment()
    litho = simpy.Resource(env, capacity=int(cfg["litho_tools"]))
    etch = simpy.Resource(env, capacity=int(cfg["etch_tools"]))
    tubes = simpy.Store(env)
    all_tubes = [_Tube(j, np.random.default_rng([6060, int(seed), 10 + j]), hours) for j in range(int(cfg["furnace_tools"]))]
    for tube in all_tubes:
        tubes.put(tube)
    cols = ["t_in", "f_start", "f_end", "l_start", "l_end", "e_start", "t_out"]
    log = np.full((n, len(cols)), np.nan)
    tube_of = np.full(n, -1)
    off = np.zeros(n, dtype=bool)
    cal_rows = []

    def calibrate(tube, reason, drift):
        t0 = env.now
        yield env.timeout(cfg["calibration_h"])
        fault_runs = float(tube.runs - tube.fault_at + 1) if tube.failing else float("nan")
        cal_rows.append((tube.id, t0, env.now, reason, tube.runs, drift, fault_runs))
        tube.reset()
        yield tubes.put(tube)

    def lot(i):
        log[i, 0] = env.now
        tube = yield tubes.get()                      # furnace: wait for a free tube
        log[i, 1] = env.now
        tube_of[i] = tube.id
        yield env.timeout(unit["furnace"][i])
        log[i, 2] = env.now
        drift = tube.run()
        off[i] = drift > COMP_LIMIT_NM                # the controller can no longer hold the target thickness
        reason = None
        if cfg["maintenance"] == "preventive" and tube.runs >= cfg["interval_runs"]:
            reason = "preventive"
        elif cfg["maintenance"] == "predictive" and trigger(list(tube.recent[-int(cfg["k_runs"]):]), cfg["limit_nm"], int(cfg["k_runs"])):
            reason = "predictive"
        if reason:
            env.process(calibrate(tube, reason, drift))   # the tube is out of service; the lot moves on
        else:
            yield tubes.put(tube)
        with litho.request() as req:                  # lithography
            yield req
            log[i, 3] = env.now
            yield env.timeout(unit["litho"][i])
        log[i, 4] = env.now
        with etch.request() as req:                   # etch
            yield req
            log[i, 5] = env.now
            yield env.timeout(unit["etch"][i])
        log[i, 6] = env.now

    def source():
        last = 0.0
        for i in range(n):
            yield env.timeout(t_in[i] - last)
            last = t_in[i]
            env.process(lot(i))

    env.process(source())
    env.run(until=hours)
    lots = pd.DataFrame(log, columns=cols)
    lots.insert(0, "lot", np.arange(1, n + 1))
    lots["tube"] = tube_of
    lots["off_target"] = off
    cal = pd.DataFrame(cal_rows, columns=["tube", "t_start", "t_end", "reason", "runs", "drift_nm", "fault_runs"]).sort_values("t_start", ignore_index=True)
    return LineRun(lots, cal, float(hours), cfg, int(seed))
