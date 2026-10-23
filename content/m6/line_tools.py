"""Helpers for the Module 6 lab (Digital Twins for Semiconductor Manufacturing): a toolset twin of furnace, lithography and etch.

Every lot in the lab is SIMULATED with line.py (the SimPy line model).

HOURS, WARMUP_H, SEEDS             the lab's run length (h), warm-up (h) and the seeds of its 20 replications
run_replications(cfg, seeds, ...)  runs one LineRun per seed (results are kept, so asking again is instant)
stage_table(run, t_start, t_end)   per tool group: tools, utilization, lots waiting, mean wait, lots in the group, mean time in the group
stage_summary(runs, ...)           stage_table() averaged over replications
welch(runs, bin_h, w)              Welch's moving average of the number of lots in the line, for choosing a warm-up
plot_run(run, ...)                 the number of lots in the line and in each queue over time, one replication
plot_welch(w, warmup)              the Welch plot
replication_table(runs, ...)       one row per replication: throughput, cycle time, lots in the line, Little's ratio
maintenance_table(runs, ...)       calibrations, off-target lots and cycle time per replication
"""

import numpy as np
import pandas as pd

import line
from line import STAGES, line_config, run_line

__version__ = "1.0.1"

HOURS = 1000.0                       # one replication simulates this long (hours), starting with an empty line
WARMUP_H = 200.0                     # the first hours are discarded: the line starts empty
SEEDS = tuple(range(1, 21))          # 20 replications: seeds 1 to 20

_CACHE = {}


def _key(cfg, seed, hours, trigger):
    return (tuple(sorted({**line.BASE_LINE, **cfg}.items())), int(seed), float(hours), getattr(trigger, "__name__", None) if trigger else None, id(trigger) if trigger else None)


def run_replications(cfg=None, seeds=SEEDS, hours=HOURS, trigger=None):
    """Run the line once per seed and return the list of LineRun objects. A run that was done before with the same
    settings, seed, length and trigger function is reused."""
    cfg = {**line.BASE_LINE, **(cfg or {})}
    out = []
    for s in seeds:
        k = _key(cfg, s, hours, trigger)
        if k not in _CACHE:
            _CACHE[k] = run_line(cfg, s, hours, trigger=trigger)
        out.append(_CACHE[k])
    return out


# ---------------------------------------------------------------- reference measurements (used by the helpers below)
def _overlap(a, b, t0, t1):
    """Hours that the intervals [a, b] spend inside [t0, t1]; a NaN end means the interval is still open at the end of the run."""
    a = np.asarray(a, dtype=float)
    b = np.where(np.isnan(np.asarray(b, dtype=float)), np.inf, np.asarray(b, dtype=float))
    ok = ~np.isnan(a)
    return np.clip(np.minimum(b[ok], t1) - np.maximum(a[ok], t0), 0.0, None)


def _in_window(t, t0, t1):
    t = np.asarray(t, dtype=float)
    with np.errstate(invalid="ignore"):
        return (t >= t0) & (t < t1)


def reference_throughput(t_out, t_start, t_end):
    return float(_in_window(t_out, t_start, t_end).sum()) / (t_end - t_start)


def reference_cycle_time(t_in, t_out, t_start, t_end):
    t_in, t_out = np.asarray(t_in, dtype=float), np.asarray(t_out, dtype=float)
    m = _in_window(t_out, t_start, t_end)
    return float(np.mean(t_out[m] - t_in[m])) if m.any() else float("nan")


def reference_wip(t_in, t_out, t_start, t_end):
    return float(_overlap(t_in, t_out, t_start, t_end).sum()) / (t_end - t_start)


# ---------------------------------------------------------------- tool groups
_GROUPS = {"furnace": ("t_in", "f_start", "f_end"),     # (arrives, starts processing, leaves)
           "litho": ("f_end", "l_start", "l_end"),
           "etch": ("l_end", "e_start", "t_out")}


def stage_table(run, t_start=WARMUP_H, t_end=None):
    """One row per tool group, for the window [t_start, t_end] (default: after the warm-up, to the end of the run).

    utilization     tool-hours spent processing, as a share of the tool-hours the group has (tools x window)
    queue_lots      average number of lots waiting for a tool
    queue_h         mean wait of the lots that started processing in the window
    in_group_lots   average number of lots in the group, waiting or processing
    in_group_h      mean time in the group of the lots that left it in the window (wait + process)"""
    t_end = run.hours if t_end is None else t_end
    L, win = run.lots, t_end - t_start
    rows = {}
    for s, (arr, start, end) in _GROUPS.items():
        tools = int(run.cfg[s + "_tools"])
        wait_started = _in_window(L[start], t_start, t_end)
        left = _in_window(L[end], t_start, t_end)
        rows[s] = {"tools": tools,
                   "utilization": _overlap(L[start], L[end], t_start, t_end).sum() / (tools * win),
                   "queue_lots": _overlap(L[arr], L[start], t_start, t_end).sum() / win,
                   "queue_h": float(np.mean((L[start] - L[arr])[wait_started])) if wait_started.any() else float("nan"),
                   "in_group_lots": _overlap(L[arr], L[end], t_start, t_end).sum() / win,
                   "in_group_h": float(np.mean((L[end] - L[arr])[left])) if left.any() else float("nan")}
    return pd.DataFrame(rows).T


def stage_summary(runs, t_start=WARMUP_H, t_end=None):
    """stage_table() averaged over the replications."""
    tabs = [stage_table(r, t_start, t_end) for r in runs]
    return sum(tabs) / len(tabs)


# ---------------------------------------------------------------- warm-up
def _wip_bins(run, bin_h):
    edges = np.arange(0.0, run.hours + 1e-9, bin_h)
    L = run.lots
    return np.array([reference_wip(L["t_in"], L["t_out"], a, a + bin_h) for a in edges[:-1]]), edges[:-1] + bin_h / 2


def welch(runs, bin_h=10.0, w=6):
    """Welch's method. Average the number of lots in the line over bins of bin_h hours in every replication, average the
    replications bin by bin, then smooth with a moving average of half-width w (a shorter, centered window near the start).
    Returns (bin centers, the average over replications, the smoothed average)."""
    ys = []
    for r in runs:
        y, t = _wip_bins(r, bin_h)
        ys.append(y)
    ybar = np.mean(ys, axis=0)
    m = len(ybar)
    sm = np.empty(m - w)
    for i in range(m - w):                    # i = 0, 1, 2, ...  (bin i + 1 in Welch's notation)
        h = min(i, w)
        sm[i] = ybar[i - h:i + h + 1].mean()
    return t, ybar, np.concatenate([sm, np.full(w, np.nan)])


# ---------------------------------------------------------------- tables
def replication_table(runs, t_start=WARMUP_H, t_end=None):
    """One row per replication, measured on the window [t_start, t_end]: throughput (lots/h), mean cycle time (h), average lots
    in the line, and Little's ratio L / (throughput x cycle time)."""
    rows = []
    for r in runs:
        te = r.hours if t_end is None else t_end
        L = r.lots
        th = reference_throughput(L["t_out"], t_start, te)
        ct = reference_cycle_time(L["t_in"], L["t_out"], t_start, te)
        wip = reference_wip(L["t_in"], L["t_out"], t_start, te)
        rows.append({"seed": r.seed, "throughput": th, "cycle_time": ct, "wip": wip, "littles_ratio": wip / (th * ct)})
    return pd.DataFrame(rows)


def maintenance_table(runs, t_start=WARMUP_H, t_end=None):
    """One row per replication: furnace calibrations started in the window and finished by the end of the run (a calibration
    is logged when it ends, so one still under way at the end is not counted), how many were triggered by the drift estimate,
    lots processed while the furnace drift was beyond what the controller can correct (off target), and the mean cycle time."""
    rows = []
    for r in runs:
        te = r.hours if t_end is None else t_end
        L, c = r.lots, r.calibrations
        done = _in_window(L["f_end"], t_start, te)
        cal = c[(c["t_start"] >= t_start) & (c["t_start"] < te)]
        rows.append({"seed": r.seed, "calibrations": len(cal), "predictive": int((cal["reason"] == "predictive").sum()),
                     "lots_run": int(done.sum()), "off_target": int((L["off_target"].to_numpy() & done).sum()),
                     "cycle_time": reference_cycle_time(L["t_in"], L["t_out"], t_start, te)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
def plot_run(run, t_start=0.0, t_end=None, warmup=WARMUP_H, title=None):
    """Lots in the line, and lots waiting at each tool group, against time, for one replication."""
    import matplotlib.pyplot as plt
    t_end = run.hours if t_end is None else t_end
    L = run.lots

    def count(a, b):
        ev = [(t, 1) for t in L[a].dropna()] + [(t, -1) for t in L[b].dropna()]
        ev.sort()
        t = np.array([0.0] + [e[0] for e in ev])
        n = np.concatenate([[0], np.cumsum([e[1] for e in ev])])
        return t, n
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    t, n = count("t_in", "t_out")
    axes[0].step(t, n, where="post", color="#2F4F4F", lw=1.0)
    axes[0].set_ylabel("lots in the line")
    for s, c in zip(STAGES, ["#1f77b4", "#d62728", "#2ca02c"]):
        arr, start, _ = _GROUPS[s]
        t, n = count(arr, start)
        axes[1].step(t, n, where="post", color=c, lw=1.0, label=s)
    axes[1].set_ylabel("lots waiting")
    axes[1].set_xlabel("time (h)")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)
    for ax in axes:
        if warmup:
            ax.axvspan(0, warmup, color="0.88", zorder=0)
        ax.set_xlim(t_start, t_end)
        ax.grid(alpha=0.3)
    axes[0].set_title(title or f"One replication (seed {run.seed}); grey: the warm-up")
    fig.tight_layout()
    return fig


def plot_welch(w, warmup=None, title="Welch's plot: lots in the line, average of the replications"):
    import matplotlib.pyplot as plt
    t, ybar, sm = w
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(t, ybar, color="0.7", lw=1.0, label="average over replications, per bin")
    ax.plot(t, sm, color="#d62728", lw=2.0, label="Welch moving average")
    if warmup:
        ax.axvline(warmup, color="#1f77b4", ls="--", label=f"warm-up {warmup:g} h")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("lots in the line")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
