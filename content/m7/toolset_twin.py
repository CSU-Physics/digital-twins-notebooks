"""Prepared components of the Module 7 toolset-twin track (Digital Twins for Semiconductor Manufacturing).

The toolset twin computes the weekly mean cycle time of the furnace, lithography and etch line of Module 6, whose furnace
tubes drift as in Module 5 and are calibrated by the predictive rule. It is calibrated from the fab's lot log and driven by
each week's actual lot releases (trace-driven). The capstone validates it retrospectively: replay() re-runs past weeks
conditional on the releases that actually happened, so the test shows how well the line model reproduces the cycle time
when the releases are known, not how well it forecasts future weeks whose releases are not yet known. The calibration
uses the lots released in weeks 1 to 6, including the steps of the few that finish early in week 7, which a forecast made
at the end of week 6 could not use. Every data value is SIMULATED with line.py (1.1.0); the "fab" is the same
model with settings of its own, which the twin has to estimate, and with lithography breakdowns that the twin does not model.

fab_log(key)              40 weeks of the line's lot log: release time and the start and end of each tool group, in hours
downtime_log(key)         the lithography breakdowns recorded by the fab (start and end, in hours)
calibrate_now(...)        the Module 5 drift alarm as the predictive maintenance rule (the course's version)
twin_config(estimates)    the twin's line settings from your estimates of the release rate and the process times
weekly_cycle_time(log, weeks)   the mean cycle time of the lots that leave in each week
twin_run(cfg, key, weeks, seed)   one trace-driven replication of the twin, on the log's clock (for verification)
replay(cfg, key, weeks)   the twin's weekly mean cycle time for past weeks, driven by the fab's release times (20 replications)
WEEK_H, CAL_WEEKS, HELDOUT_WEEKS, CHALLENGING_WEEKS, TOLERANCE_H   the validation plan
"""

import numpy as np
import pandas as pd

import line
from line import line_config, run_line

__version__ = "1.0.1"

WEEK_H = 168.0                 # one week, hours
CAL_WEEKS = (1, 6)             # weeks 1 to 6 of the log calibrate the twin
HELDOUT_WEEKS = (7, 30)        # held-out test: weeks 7 to 30
CHALLENGING_WEEKS = (31, 40)   # challenging test: weeks 31 to 40, when the lithography tool breaks down now and then
TOLERANCE_H = 1.0              # the twin is fit for its use if its mean absolute weekly error is at most 1.0 h
REPLICATIONS = 20              # replications of the twin per week (seeds 1 to 20)
LEAD_H = 200.0                 # the twin starts empty this long before the first week it replays (its warm-up)
_WARM_H = 300.0                # the simulated fab ran this long before the log starts
_LAST_WEEK = 40


def _truth(key):
    rng = np.random.default_rng([2093, int(key)])
    return {"arrival_rate": float(rng.uniform(0.70, 0.76)), "furnace_h": float(rng.uniform(1.75, 1.85)),
            "litho_h": float(rng.uniform(0.95, 1.03)), "etch_h": float(rng.uniform(1.15, 1.25)),
            "litho_mtbf_h": float(rng.uniform(30.0, 40.0)), "litho_mttr_h": 5.0}


def calibrate_now(recent, limit, k):
    """Module 5 and 6: True when the last k drift estimates are all beyond the limit in size."""
    recent = list(recent)
    return len(recent) >= k and all(abs(v) > limit for v in recent[-k:])


_FAB = {}


def _fab(key):
    key = int(key)
    if key not in _FAB:
        t = _truth(key)
        cfg = line_config(arrival_rate=t["arrival_rate"], furnace_h=t["furnace_h"], litho_h=t["litho_h"], etch_h=t["etch_h"],
                          maintenance="predictive", litho_mtbf_h=t["litho_mtbf_h"], litho_mttr_h=t["litho_mttr_h"],
                          litho_down_from_h=_WARM_H + (CHALLENGING_WEEKS[0] - 1) * WEEK_H)
        r = run_line(cfg, 900000 + key, _WARM_H + _LAST_WEEK * WEEK_H, trigger=calibrate_now)
        lots = r.lots.copy()
        cols = ["t_in", "f_start", "f_end", "l_start", "l_end", "e_start", "t_out"]
        lots[cols] = lots[cols] - _WARM_H
        lots = lots[lots["t_in"] >= 0].reset_index(drop=True)
        lots["lot"] = np.arange(1, len(lots) + 1)
        lots = lots[["lot"] + cols]
        down = r.downtime.copy()
        down[["t_start", "t_end"]] = down[["t_start", "t_end"]] - _WARM_H
        _FAB[key] = (lots, down.reset_index(drop=True))
    lots, down = _FAB[key]
    return lots.copy(), down.copy()


def fab_log(key):
    """The fab's lot log for the 40 weeks of the line of key, one row per lot released since the log started (hour 0).
    Columns: lot, t_in (release), f_start, f_end (furnace), l_start, l_end (lithography), e_start (etch start), t_out
    (leaves the line), all in hours; NaN where a lot had not got that far when the log ends. The line was already
    running when the log starts. The same key always gives the same log."""
    return _fab(key)[0]


def downtime_log(key):
    """The lithography breakdowns that the fab recorded: tool, t_start and t_end in hours since the log started."""
    return _fab(key)[1]


def week_window(weeks):
    """The hours (start, end) of weeks first to last: week 1 is hours 0 to 168."""
    return (weeks[0] - 1) * WEEK_H, weeks[1] * WEEK_H


def twin_config(estimates):
    """The twin's line settings: line_config() with your estimates of arrival_rate, furnace_h, litho_h, etch_h,
    furnace_cv, litho_cv and etch_cv, the line's tool counts, and the predictive maintenance rule of Modules 5 and 6.
    The twin has no breakdowns: nothing in the calibration weeks showed any."""
    need = ("arrival_rate", "furnace_h", "litho_h", "etch_h", "furnace_cv", "litho_cv", "etch_cv")
    missing = [k for k in need if k not in estimates]
    if missing:
        raise KeyError(f"the estimates need {missing}")
    return line_config(maintenance="predictive", **{k: float(np.asarray(estimates[k], dtype=float).reshape(-1)[0]) for k in need})


def weekly_cycle_time(log, weeks):
    """The mean cycle time (hours) of the lots that leave the line in each week from weeks[0] to weeks[1]: a Series indexed
    by week. The window of week w is [168 (w - 1), 168 w)."""
    t_in, t_out = log["t_in"].to_numpy(dtype=float), log["t_out"].to_numpy(dtype=float)
    out = {}
    for w in range(weeks[0], weeks[1] + 1):
        a, b = (w - 1) * WEEK_H, w * WEEK_H
        with np.errstate(invalid="ignore"):
            m = (t_out >= a) & (t_out < b)
        out[w] = float(np.mean(t_out[m] - t_in[m])) if m.any() else float("nan")
    return pd.Series(out, name="cycle_time_h")


_REPLAY = {}


def twin_run(cfg, key, weeks, seed=1):
    """One replication of the trace-driven twin for weeks[0] to weeks[1]: it starts empty LEAD_H hours before the first
    week and releases its lots at the fab's own release times. Returns the LineRun, with every time on the log's clock
    (hours since the log started)."""
    log = fab_log(key)
    start = (weeks[0] - 1) * WEEK_H - LEAD_H
    end = weeks[1] * WEEK_H
    t_in = log["t_in"].to_numpy(dtype=float)
    t_in = t_in[(t_in >= start) & (t_in < end)] - start
    r = run_line({**line.BASE_LINE, **cfg}, seed, end - start, trigger=calibrate_now, t_in=t_in)
    cols = ["t_in", "f_start", "f_end", "l_start", "l_end", "e_start", "t_out"]
    r.lots[cols] = r.lots[cols] + start
    r.calibrations[["t_start", "t_end"]] = r.calibrations[["t_start", "t_end"]] + start
    return r


def replay(cfg, key, weeks, replications=REPLICATIONS):
    """The twin's replay of the weekly mean cycle time for weeks[0] to weeks[1], trace-driven: the twin starts empty
    LEAD_H hours before the first week, its lots are released at the fab's own release times (from fab_log(key)), and
    its process times, tube drift and calibrations are its own, from seeds 1 to 20. Returns a Series indexed by week: the
    mean over the replications. Results are kept, so asking again with the same settings is instant."""
    cfg = {**line.BASE_LINE, **cfg}
    ck = (tuple(sorted(cfg.items())), int(key), tuple(weeks), int(replications))
    if ck not in _REPLAY:
        per = [weekly_cycle_time(twin_run(cfg, key, weeks, s).lots, weeks).to_numpy() for s in range(1, int(replications) + 1)]
        _REPLAY[ck] = pd.Series(np.mean(per, axis=0), index=range(weeks[0], weeks[1] + 1), name="twin_h")
    return _REPLAY[ck].copy()


def plot_weeks(observed, twin, baseline, title=None, tolerance=TOLERANCE_H, window_name="test"):
    """Observed and replayed weekly mean cycle time against the week, and the twin's error, with the tolerance for scale.

    The verdict is about the mean absolute error (MAE) over all the weeks shown, not about single weeks: a week outside the
    shaded band does not by itself fail the twin, and an MAE within the tolerance does not make every week acceptable.
    The lower panel's title gives that MAE with its approximate 95 % interval (which treats the weeks as independent) and
    the tolerance."""
    import matplotlib.pyplot as plt
    from capstone import mae_ci
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True)
    axes[0].plot(observed.index, observed.values, "o-", ms=4, lw=1, color="0.35", label="fab log")
    axes[0].plot(twin.index, twin.values, ".-", ms=4, lw=1.2, color="C0", label="twin (trace-driven replay, 20 replications)")
    axes[0].plot(observed.index, np.full(len(observed), baseline), "--", lw=1.2, color="C3", label="baseline: mean of weeks 1 to 6")
    axes[0].set_ylabel("weekly mean cycle time (h)")
    axes[1].axhspan(-tolerance, tolerance, color="C2", alpha=0.15, label=f"+/- {tolerance:g} h: the MAE tolerance, for scale")
    axes[1].plot(observed.index, (observed - twin).values, ".-", ms=4, lw=1, color="C0", label="twin error (log - twin)")
    axes[1].axhline(0, color="k", lw=0.8)
    err = (observed - twin).to_numpy(dtype=float)
    if np.sum(~np.isnan(err)) >= 2:
        m, hw, n = mae_ci(err)
        axes[1].set_title(f"{window_name} weeks {int(observed.index[0])} to {int(observed.index[-1])}: twin MAE {m:.2f} h, "
                          f"approx. 95 % interval {m - hw:.2f} to {m + hw:.2f} h (n = {n}), tolerance on the MAE {tolerance:g} h",
                          fontsize=8.5, loc="left")
    axes[1].set_ylabel("error (h)")
    axes[1].set_xlabel("week")
    for ax in axes:
        ax.legend(fontsize=8, loc="upper left")
    if title:
        axes[0].set_title(title, fontsize=10)
    fig.text(0.01, 0.005, "The verdict compares the MAE of all the weeks shown, with its interval, with the tolerance. "
             "A single week outside the band does not fail the twin,\nand an MAE within the tolerance does not make every week "
             "acceptable.", fontsize=7.5, color="0.3", ha="left", va="bottom")
    plt.tight_layout(rect=(0, 0.045, 1, 1))
    plt.show()
