"""Build the Module 6 slide deck (media/m6/m6-slides.html) for Digital Twins for Semiconductor Manufacturing.

Every number and point on the slides is computed here from the course's line model (content/m6/line.py), the lab's helpers
(content/m6/line_tools.py) and the reference functions of the Module 6 check, so the slides agree with the lab notebook.
Run from anywhere: python3 tools/build_dt_m6_media.py
"""
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m6"), str(ROOT / "checks")]
import line as LN  # noqa: E402
import line_tools as LT  # noqa: E402
from line import BASE_LINE, line_config  # noqa: E402
from dt_m6_check import reference_calibrate, reference_ci95, reference_cycle_time, personal_value  # noqa: E402

T = ROOT / "tools"
OUT = ROOT / "media" / "m6"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, INK, MUTED, LINE, TINT, OK, BAD, BLUE, GOLD, GREEN = "#99004C", "#24313D", "#6A626B", "#E4DCE0", "#F8F4F6", "#2E7D4F", "#B3261E", "#2F5E8E", "#A86B00", "#2E7D4F"
HOURS, WARM = LT.HOURS, LT.WARMUP_H

# ------------------------------------------------------------------ numbers used on the slides (the lab's example: seeds 1 to 20)
RUNS = LT.run_replications()
RUN1 = RUNS[0]
CT = np.array([reference_cycle_time(r.lots["t_in"], r.lots["t_out"], WARM, HOURS) for r in RUNS])
CT_M, CT_H = reference_ci95(CT)
REP = LT.replication_table(RUNS)
THR_M = REP["throughput"].mean()
WIP_M = REP["wip"].mean()
RATIO_M, RATIO_H = reference_ci95(REP["littles_ratio"])
STAGE = LT.stage_summary(RUNS)
WELCH = LT.welch(RUNS, 10, 6)
HW = {n: reference_ci95(CT[:n])[1] for n in (5, 10, 20)}
N_NEEDED = math.ceil(20 * (CT_H / 0.2) ** 2)


def whatif(**changes):
    rs = LT.run_replications(line_config(**changes))
    ct = np.array([reference_cycle_time(r.lots["t_in"], r.lots["t_out"], WARM, HOURS) for r in rs])
    d = CT - ct
    return {"ct": ct, "mean": ct.mean(), "half": reference_ci95(ct)[1], "diff": d, "dmean": d.mean(), "dhalf": reference_ci95(d)[1]}


WI = {"etch 10 % faster": whatif(etch_h=1.08), "furnace 10 % faster": whatif(furnace_h=1.62), "lithography 10 % faster": whatif(litho_h=0.9)}
WI_ETCH, WI_LITHO = WI["etch 10 % faster"], WI["lithography 10 % faster"]
assert WI_ETCH["diff"].min() > 0 and WI_LITHO["diff"].min() > 0
assert WI_LITHO["mean"] + WI_LITHO["half"] < CT_M - CT_H < CT_M + CT_H and WI_ETCH["mean"] + WI_ETCH["half"] > CT_M - CT_H

SWEEP = {}
for lam in (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93):
    rs = LT.run_replications(line_config(arrival_rate=lam))
    ct = [reference_cycle_time(r.lots["t_in"], r.lots["t_out"], WARM, HOURS) for r in rs]
    SWEEP[lam] = (reference_ci95(ct), LT.stage_summary(rs).loc["litho", "utilization"])

POLICY = {}
for name, cfg in (("none", line_config()), ("preventive", line_config(maintenance="preventive", interval_runs=50)),
                  ("predictive", line_config(maintenance="predictive"))):
    rs = LT.run_replications(cfg, trigger=reference_calibrate)
    mt = LT.maintenance_table(rs)
    fr = np.concatenate([r.calibrations["fault_runs"].dropna().to_numpy() for r in rs]) if name != "none" else np.array([])
    POLICY[name] = {"cal": reference_ci95(mt["calibrations"]), "off": reference_ci95(mt["off_target"]), "off_max": int(mt["off_target"].max()),
                    "lots": mt["lots_run"].mean(), "ct": reference_ci95(mt["cycle_time"]), "fault_med": float(np.median(fr)) if len(fr) else None}
assert POLICY["predictive"]["off_max"] == 0 and POLICY["preventive"]["off"][0] > 10


def tube_path(policy, n=170, fault_run=112, seed=3):
    """One furnace tube under a policy, with the thermocouple failing from a chosen run: (rows, calibration runs)."""
    tube = LN._Tube(0, np.random.default_rng([6060, seed, 10]), 600.0)
    tube.fault_u[:] = 1.0
    tube.fault_u[fault_run - 1] = 0.0
    rows, cal = [], []
    for r in range(1, n + 1):
        drift = tube.run()
        rows.append((r, drift, tube.d_hat))
        if (policy == "preventive" and tube.runs >= 50) or (policy == "predictive" and reference_calibrate(tube.recent[-3:], 2.0, 3)):
            cal.append(r)
            tube.reset()
    return rows, cal


TP = {p: tube_path(p) for p in ("preventive", "predictive")}
TP_OFF = {p: sum(1 for r, d, e in rows if d > LN.COMP_LIMIT_NM) for p, (rows, cal) in TP.items()}
assert TP_OFF["predictive"] == 0 and TP_OFF["preventive"] > 5

# Little: pause-and-predict
VAL_417 = personal_value(417)


# ------------------------------------------------------------------ SVG helpers
def txt(x, y, s, size=20, fill=INK, anchor="middle", weight=400, extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{extra}>{s}</text>'


def arrow_def(mid, color=MUTED):
    return (f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0 0L10 5L0 10z" fill="{color}"/></marker></defs>')


def axes(s, X_, Y_, x0, x1, y0, y1, xt, yt, xl, yl, yfmt="{:.0f}", xfmt=None):
    for v in yt:
        s.append(f'<line x1="{x0}" y1="{Y_(v):.1f}" x2="{x1}" y2="{Y_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y_(v) + 7, yfmt.format(v), 19, MUTED, "end"))
    for v in xt:
        s.append(txt(X_(v), y0 + 28, xfmt(v) if xfmt else f"{v:g}", 19, MUTED))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt((x0 + x1) / 2, y0 + 56, xl, 20, MUTED))
    s.append(txt(22, (y0 + y1) / 2, yl, 20, MUTED, extra=f' transform="rotate(-90 22 {(y0 + y1) / 2:.1f})"'))


def poly(xs, ys, X_, Y_, color, width=3.5, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{" ".join(f"{X_(a):.1f},{Y_(b):.1f}" for a, b in zip(xs, ys))}" fill="none" stroke="{color}" stroke-width="{width}"{d}/>'


def legend(s, x, y, items, size=18):
    for k, (kind, color, label) in enumerate(items):
        yy = y + 26 * k
        if kind == "line":
            s.append(f'<line x1="{x}" y1="{yy - 6}" x2="{x + 30}" y2="{yy - 6}" stroke="{color}" stroke-width="4"/>')
        elif kind == "dash":
            s.append(f'<line x1="{x}" y1="{yy - 6}" x2="{x + 30}" y2="{yy - 6}" stroke="{color}" stroke-width="3" stroke-dasharray="7 5"/>')
        elif kind == "dot":
            s.append(f'<circle cx="{x + 15}" cy="{yy - 6}" r="6" fill="{color}"/>')
        elif kind == "sq":
            s.append(f'<rect x="{x + 9}" y="{yy - 12}" width="12" height="12" fill="{color}"/>')
        s.append(txt(x + 40, yy, label, size, INK, "start"))


def step_points(times_up, times_down, t_end, step=0.0):
    """The step function (number in the line) as a list of (t, n) points, for a polyline."""
    ev = sorted([(t, 1) for t in times_up if t == t] + [(t, -1) for t in times_down if t == t])
    pts, n = [(0.0, 0)], 0
    for t, d in ev:
        if t > t_end:
            break
        pts.append((t, n))
        n += d
        pts.append((t, n))
    pts.append((t_end, n))
    return pts


# ------------------------------------------------------------------ figures
def line_fig(w=1130, h_=250, mid="a1"):
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 1130 250" role="img" aria-label="The line: lots are released, wait for and use the furnace, then lithography, then etch, then leave; each tool group has a queue in front of it">',
         arrow_def(mid, MAROON)]
    groups = [(160, "furnace", "2 tubes · 1.8 h", BLUE), (470, "lithography", "1 tool · 1.0 h", BAD), (780, "etch", "2 chambers · 1.2 h", GREEN)]
    s.append(txt(60, 98, "lots", 20, MUTED))
    s.append(f'<line x1="22" y1="120" x2="{groups[0][0] - 54}" y2="120" stroke="{MAROON}" stroke-width="4" marker-end="url(#{mid})"/>')
    for x, name, sub, col in groups:
        for k in range(3):                                   # the queue: waiting lots
            s.append(f'<rect x="{x - 46 + 14 * k}" y="{100}" width="10" height="40" rx="2" fill="{MUTED}" opacity="{0.9 - 0.2 * k}"/>')
        s.append(f'<rect x="{x}" y="64" width="190" height="112" rx="10" fill="#fff" stroke="{col}" stroke-width="3"/>')
        s.append(txt(x + 95, 108, name, 25, INK, weight=600))
        s.append(txt(x + 95, 144, sub, 19, MUTED))
    for a, b in ((groups[0][0] + 192, groups[1][0] - 50), (groups[1][0] + 192, groups[2][0] - 50)):
        s.append(f'<line x1="{a}" y1="120" x2="{b}" y2="120" stroke="{MAROON}" stroke-width="4" marker-end="url(#{mid})"/>')
    s.append(f'<line x1="{groups[2][0] + 192}" y1="120" x2="{groups[2][0] + 240}" y2="120" stroke="{MAROON}" stroke-width="4" marker-end="url(#{mid})"/>')
    s.append(txt(groups[2][0] + 236, 98, "lots out", 20, MUTED, "start"))
    s.append(f'<rect x="12" y="188" width="1106" height="46" rx="8" fill="{TINT}" stroke="{LINE}"/>')
    s.append(txt(565, 218, "one boundary around the whole line: lots in (λ), lots out (throughput), time in the line (W), lots in the line (L)", 19, INK))
    s.append("</svg>")
    return "".join(s)


def where_fig(w=620, h_=390):
    """Per tool group, the mean time a lot spends: processing and waiting."""
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Mean time in each tool group, split into processing and waiting; lithography has the longest wait">']
    x0, x1 = 150, w - 24
    X_ = lambda v: x0 + v / 4.2 * (x1 - x0)
    names = [("furnace", BASE_LINE["furnace_h"]), ("litho", BASE_LINE["litho_h"]), ("etch", BASE_LINE["etch_h"])]
    for i, (g, proc) in enumerate(names):
        y = 70 + i * 82
        wait = float(STAGE.loc[g, "queue_h"])
        s.append(txt(x0 - 12, y + 30, "lithography" if g == "litho" else g, 21, INK, "end", 600))
        s.append(f'<rect x="{X_(0)}" y="{y}" width="{X_(proc) - X_(0):.1f}" height="44" fill="{BLUE}"/>')
        s.append(f'<rect x="{X_(proc):.1f}" y="{y}" width="{X_(wait) - X_(0):.1f}" height="44" fill="{MAROON}"/>')
        s.append(txt(X_(proc + wait) + 8, y + 30, f"{proc:.1f} + {wait:.2f} h", 19, INK, "start"))
    for v in range(0, 5):
        s.append(f'<line x1="{X_(v):.1f}" y1="{50}" x2="{X_(v):.1f}" y2="{300}" stroke="{LINE}"/>')
        s.append(txt(X_(v), 322, f"{v}", 19, MUTED))
    s.append(txt((x0 + x1) / 2, 352, "mean time in the group (h)", 20, MUTED))
    legend(s, x0 + 20, 26, [("sq", BLUE, "processing"), ("sq", MAROON, "waiting for a tool")], 17)
    s.append("</svg>")
    return "".join(s)


def _wip_series(L, t0, t1):
    """Number of lots in the line as a step function over [t0, t1], as (t, n) points, plus the area (lot-hours) counted slice by slice."""
    t_in = np.asarray(L["t_in"], float)
    t_out = np.asarray(L["t_out"], float)
    ev = sorted([(t, 1) for t in t_in] + [(t, -1) for t in t_out if t == t])
    cur, series = 0, []
    for t, d in ev:
        series.append((t, cur))
        cur += d
        series.append((t, cur))
    before = [n for t, n in series if t <= t0]
    n_start = before[-1] if before else 0
    pts = [(t0, n_start)] + [(t, n) for t, n in series if t0 < t < t1]
    pts.append((t1, pts[-1][1]))
    area = sum((pts[i + 1][0] - pts[i][0]) * pts[i][1] for i in range(len(pts) - 1))
    return pts, area


def _lot_hours(L, t0, t1):
    """The same area counted lot by lot: hours each lot spends in the line inside the window."""
    t_in = np.asarray(L["t_in"], float)
    t_out = np.asarray(L["t_out"], float)
    t_out = np.where(np.isnan(t_out), 1e18, t_out)
    return float(np.clip(np.minimum(t_out, t1) - np.maximum(t_in, t0), 0, None).sum())


def little_fig(w=620, h_=390):
    """WIP step function for a typical 40 hours of seed 1, with the two ways of counting the area."""
    L = RUN1.lots
    target = float(REP.loc[0, "wip"])
    best = None
    for t0 in range(200, 940, 10):
        pts, area = _wip_series(L, float(t0), float(t0 + 40))
        mx = max(n for t, n in pts)
        gap = abs(area / 40 - target)
        if mx <= 18 and gap <= 0.5 and (best is None or gap < best[3]):
            best = (float(t0), pts, area, gap)
    assert best is not None, "no typical 40 h window"
    t0, series, area_val, _gap = best
    t1 = t0 + 40.0
    lot_h = _lot_hours(L, t0, t1)
    assert abs(lot_h - area_val) < 1e-6, (lot_h, area_val)
    avg = area_val / 40.0
    x0, x1, y0, y1 = 80, w - 20, h_ - 70, 24
    X_ = lambda v: x0 + (v - t0) / (t1 - t0) * (x1 - x0)
    Y_ = lambda v: y0 - v / 18 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="The number of lots in the line over 40 hours as a staircase; the area under it is the average number of lots times the window, and also the sum of the hours each lot spends in the line">']
    axes(s, X_, Y_, x0, x1, y0, y1, [t0 + 10 * k for k in range(5)], [0, 4, 8, 12, 16], "time (h)", "lots in the line")
    area = " ".join(f"{X_(t):.1f},{Y_(n):.1f}" for t, n in series)
    s.append(f'<polygon points="{X_(t0):.1f},{y0} {area} {X_(t1):.1f},{y0}" fill="{MAROON}" opacity="0.18"/>')
    s.append(poly([t for t, n in series], [n for t, n in series], X_, Y_, MAROON, 3))
    s.append(poly([t0, t1], [avg, avg], X_, Y_, INK, 2.5, "8 6"))
    s.append(f'<rect x="{x1 - 170}" y="{Y_(avg) - 34:.1f}" width="166" height="26" rx="4" fill="#fff" opacity="0.85"/>')
    s.append(txt(x1 - 8, Y_(avg) - 14, f"average L = {avg:.1f}", 18, INK, "end", 600))
    s.append("</svg>")
    return "".join(s), avg, area_val, t0


def sweep_fig(w=620, h_=390):
    x0, x1, y0, y1 = 92, w - 24, h_ - 70, 24
    X_ = lambda v: x0 + (v - 0.25) / 0.70 * (x1 - x0)
    Y_ = lambda v: y0 - (v - 3) / 10 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Mean cycle time against the load on lithography: it rises slowly at first and then steeply as utilization passes 0.85">']
    axes(s, X_, Y_, x0, x1, y0, y1, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [4, 6, 8, 10, 12], "utilization of lithography", "mean cycle time (h)", xfmt=lambda v: f"{v:.1f}")
    lams = sorted(SWEEP)
    xs = [SWEEP[l][1] for l in lams]
    ys = [SWEEP[l][0][0] for l in lams]
    s.append(poly(xs, ys, X_, Y_, MAROON, 3.5))
    for l in lams:
        (m, h), u = SWEEP[l]
        s.append(f'<line x1="{X_(u):.1f}" y1="{Y_(m - h):.1f}" x2="{X_(u):.1f}" y2="{Y_(m + h):.1f}" stroke="{MAROON}" stroke-width="2"/>')
        s.append(f'<circle cx="{X_(u):.1f}" cy="{Y_(m):.1f}" r="5" fill="{MAROON}"/>')
    s.append(poly([0.25, 0.95], [4.0, 4.0], X_, Y_, MUTED, 2, "7 5"))
    s.append(txt(x1 - 4, Y_(4.0) + 22, "processing time alone, 4.0 h", 16, MUTED, "end"))
    s.append(txt(X_(SWEEP[0.85][1]) - 10, Y_(SWEEP[0.85][0][0]) - 24, "the lab's line: 0.84", 17, INK, "end", 600))
    s.append("</svg>")
    return "".join(s)


def wip_fig(w=1130, h_=300):
    L = RUN1.lots
    x0, x1, y0, y1 = 84, w - 20, h_ - 66, 20
    X_ = lambda v: x0 + v / HOURS * (x1 - x0)
    Y_ = lambda v: y0 - v / 16 * (y0 - y1)
    pts = step_points(L["t_in"], L["t_out"], HOURS)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Lots in the line over 1000 hours for one replication: the line starts empty, fills within the first hours and then moves up and down around seven lots; the first 200 hours are the warm-up">']
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 200, 400, 600, 800, 1000], [0, 4, 8, 12, 16], "time (h)", "lots in the line")
    s.append(f'<rect x="{X_(0)}" y="{y1}" width="{X_(WARM) - X_(0):.1f}" height="{y0 - y1}" fill="#000" opacity="0.07"/>')
    s.append(txt(X_(WARM / 2), y1 + 22, "warm-up", 18, MUTED, "middle", 600))
    s.append(poly([p[0] for p in pts], [p[1] for p in pts], X_, Y_, MAROON, 1.6))
    w_mean = float(REP.loc[0, "wip"])
    s.append(poly([WARM, HOURS], [w_mean, w_mean], X_, Y_, INK, 2.5, "8 6"))
    s.append(f'<rect x="{x1 - 292}" y="{Y_(w_mean) - 34:.1f}" width="288" height="26" rx="4" fill="#fff" opacity="0.9"/>')
    s.append(txt(x1 - 8, Y_(w_mean) - 14, f"average after the warm-up: {w_mean:.1f}", 18, INK, "end", 600))
    s.append("</svg>")
    return "".join(s)


def welch_fig(w=620, h_=390):
    t, raw, sm = WELCH
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 24
    X_ = lambda v: x0 + v / 400 * (x1 - x0)
    Y_ = lambda v: y0 - v / 10 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Welch plot: the average number of lots in the line over 20 replications rises from about 4 in the first bin and levels off near 7 after about 120 hours">']
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 100, 200, 300, 400], [0, 2, 4, 6, 8, 10], "time (h)", "lots in the line")
    m = t <= 400
    s.append(poly(t[m], raw[m], X_, Y_, "#BBB", 2))
    mm = m & ~np.isnan(sm)
    s.append(poly(t[mm], sm[mm], X_, Y_, MAROON, 3.5))
    s.append(f'<line x1="{X_(WARM):.1f}" y1="{y0}" x2="{X_(WARM):.1f}" y2="{y1}" stroke="{BLUE}" stroke-width="2.5" stroke-dasharray="7 5"/>')
    legend(s, x0 + 8, y0 - 84, [("line", "#BBB", "replication average"), ("line", MAROON, "smoothed"), ("dash", BLUE, "warm-up 200 h")], 17)
    s.append("</svg>")
    return "".join(s)


def reps_fig(w=620, h_=390):
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 24
    X_ = lambda v: x0 + (v - 0.5) / 20 * (x1 - x0)
    Y_ = lambda v: y0 - (v - 5.5) / 4.5 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Mean cycle time of each of the 20 replications, from 6.1 to 9.1 hours, with their mean and 95 percent interval">']
    axes(s, X_, Y_, x0, x1, y0, y1, [1, 5, 10, 15, 20], [6, 7, 8, 9, 10], "replication (seed)", "mean cycle time (h)")
    s.append(f'<rect x="{x0}" y="{Y_(CT_M + CT_H):.1f}" width="{x1 - x0}" height="{Y_(CT_M - CT_H) - Y_(CT_M + CT_H):.1f}" fill="{MAROON}" opacity="0.15"/>')
    s.append(poly([0.5, 20.5], [CT_M, CT_M], X_, Y_, MAROON, 2.5))
    for i, v in enumerate(CT, start=1):
        s.append(f'<circle cx="{X_(i):.1f}" cy="{Y_(v):.1f}" r="5.5" fill="{BLUE}"/>')
    legend(s, x0 + 160, y1 + 22, [("dot", BLUE, "one replication"), ("line", MAROON, f"mean {CT_M:.2f} h, interval ± {CT_H:.2f}")], 17)
    s.append("</svg>")
    return "".join(s)


def bottleneck_fig(w=620, h_=390):
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Utilization and mean wait of the three tool groups: lithography has the highest utilization and the longest wait">']
    x0 = 150
    s.append(txt(x0 + 120, 40, "utilization", 21, MUTED, "middle", 600))
    s.append(txt(x0 + 330, 40, "mean wait (h)", 21, MUTED, "middle", 600))
    for i, g in enumerate(("furnace", "litho", "etch")):
        y = 70 + i * 92
        u, q = float(STAGE.loc[g, "utilization"]), float(STAGE.loc[g, "queue_h"])
        s.append(txt(x0 - 12, y + 32, "lithography" if g == "litho" else g, 21, INK, "end", 600))
        s.append(f'<rect x="{x0}" y="{y}" width="{240 * 0.62:.1f}" height="46" fill="{TINT}" stroke="{LINE}"/>')
        s.append(f'<rect x="{x0}" y="{y}" width="{u * 240 * 0.62:.1f}" height="46" fill="{BAD if g == "litho" else BLUE}"/>')
        s.append(txt(x0 + 240 * 0.62 + 8, y + 32, f"{u:.2f}", 20, INK, "start", 600))
        s.append(f'<rect x="{x0 + 250}" y="{y}" width="{q / 2.6 * 160:.1f}" height="46" fill="{MAROON}"/>')
        s.append(txt(x0 + 250 + q / 2.6 * 160 + 8, y + 32, f"{q:.2f}", 20, INK, "start", 600))
    s.append(txt(x0 + 60, 360, "share of tool-hours in use", 17, MUTED))
    s.append("</svg>")
    return "".join(s)


def whatif_fig(w=620, h_=390):
    x0, x1, y0, y1 = 92, w - 20, h_ - 90, 24
    names = ["etch 10 % faster", "furnace 10 % faster", "lithography 10 % faster"]
    labels = ["etch", "furnace", "lithography"]
    Y_ = lambda v: y0 - v / 1.8 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Paired mean reduction in cycle time for a 10 percent faster process time at etch, furnace and lithography, with 95 percent intervals: 0.14, 0.43 and 1.35 hours">']
    for v in (0, 0.5, 1.0, 1.5):
        s.append(f'<line x1="{x0}" y1="{Y_(v):.1f}" x2="{x1}" y2="{Y_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y_(v) + 7, f"{v:.1f}", 19, MUTED, "end"))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt(22, (y0 + y1) / 2, "mean cycle time saved (h)", 20, MUTED, extra=f' transform="rotate(-90 22 {(y0 + y1) / 2:.1f})"'))
    for i, (n, lab) in enumerate(zip(names, labels)):
        cx = x0 + 90 + i * 160
        d = WI[n]
        col = BAD if lab == "lithography" else BLUE
        s.append(f'<rect x="{cx - 38}" y="{Y_(d["dmean"]):.1f}" width="76" height="{y0 - Y_(d["dmean"]):.1f}" fill="{col}"/>')
        s.append(f'<line x1="{cx}" y1="{Y_(d["dmean"] - d["dhalf"]):.1f}" x2="{cx}" y2="{Y_(d["dmean"] + d["dhalf"]):.1f}" stroke="{INK}" stroke-width="3"/>')
        s.append(txt(cx, Y_(d["dmean"] + d["dhalf"]) - 10, f"{d['dmean']:.2f} h", 20, INK, "middle", 600))
        s.append(txt(cx, y0 + 30, lab, 20, INK, "middle", 600))
        s.append(txt(cx, y0 + 54, f"{BASE_LINE[{'etch': 'etch_h', 'furnace': 'furnace_h', 'lithography': 'litho_h'}[lab]]:.1f} h process", 16, MUTED))
    s.append("</svg>")
    return "".join(s)


def pair_fig(w=620, h_=390):
    x0, x1, y0, y1 = 92, w - 20, h_ - 60, 24
    Y_ = lambda v: y0 - (v - 5.5) / 4.5 * (y0 - y1)
    xa, xb = x0 + 140, x1 - 140
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Mean cycle time of the 20 seeds for the baseline and for the line with a 10 percent faster etch; the cycle times differ by up to 3 hours between seeds, and every seed gets a little faster, by 0.13 to 0.15 hours">']
    for v in (6, 7, 8, 9, 10):
        s.append(f'<line x1="{x0}" y1="{Y_(v):.1f}" x2="{x1}" y2="{Y_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y_(v) + 7, f"{v}", 19, MUTED, "end"))
    s.append(txt(22, (y0 + y1) / 2, "mean cycle time (h)", 20, MUTED, extra=f' transform="rotate(-90 22 {(y0 + y1) / 2:.1f})"'))
    for a, b in zip(CT, WI_ETCH["ct"]):
        s.append(f'<line x1="{xa}" y1="{Y_(a):.1f}" x2="{xb}" y2="{Y_(b):.1f}" stroke="{MAROON}" stroke-width="2" opacity="0.7"/>')
        s.append(f'<circle cx="{xa}" cy="{Y_(a):.1f}" r="4.5" fill="{BLUE}"/>')
        s.append(f'<circle cx="{xb}" cy="{Y_(b):.1f}" r="4.5" fill="{BLUE}"/>')
    s.append(txt(xa, y0 + 32, "baseline", 20, INK, "middle", 600))
    s.append(txt(xb, y0 + 32, "etch 10 % faster", 20, INK, "middle", 600))
    s.append("</svg>")
    return "".join(s)


def tube_fig(w=1130, h_=420):
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="One furnace tube whose thermocouple starts to fail at run 112, under a preventive rule (calibrate every 50 runs) and a predictive rule (calibrate when the drift estimate has been beyond 2 nm for 3 runs): the preventive rule lets the drift pass the 5 nm limit for 17 runs, the predictive rule calibrates 12 runs after the failure starts">']
    x0, x1 = 90, w - 20
    X_ = lambda v: x0 + v / 170 * (x1 - x0)
    for k, (policy, label) in enumerate((("preventive", "preventive: calibrate every 50 runs"), ("predictive", "predictive: alarm at 2.0 nm for 3 runs"))):
        rows, cal = TP[policy]
        y0 = 175 + k * 205
        y1 = y0 - 140
        Y_ = lambda v, y0=y0, y1=y1: y0 - v / 9 * (y0 - y1)
        for v in (0, 2, 5, 8):
            s.append(f'<line x1="{x0}" y1="{Y_(v):.1f}" x2="{x1}" y2="{Y_(v):.1f}" stroke="{LINE}"/>')
            s.append(txt(x0 - 10, Y_(v) + 7, f"{v}", 18, MUTED, "end"))
        s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
        s.append(poly([0, 170], [LN.COMP_LIMIT_NM] * 2, X_, Y_, BAD, 2, "7 5"))
        s.append(poly([0, 170], [2.0] * 2, X_, Y_, GOLD, 2, "7 5"))
        # segments between calibrations
        segs, cur = [], []
        for r, d, e in rows:
            cur.append((r, d, e))
            if r in cal:
                segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        for seg in segs:
            xs = [a for a, b, c in seg]
            s.append(poly(xs, [b for a, b, c in seg], X_, Y_, BLUE, 3))
            s.append(poly(xs, [c for a, b, c in seg], X_, Y_, MAROON, 2, "5 3"))
        for r in cal:
            s.append(f'<line x1="{X_(r):.1f}" y1="{y0}" x2="{X_(r):.1f}" y2="{y1 - 6}" stroke="{GREEN}" stroke-width="3"/>')
        s.append(f'<line x1="{X_(112):.1f}" y1="{y0}" x2="{X_(112):.1f}" y2="{y1 + 20}" stroke="{MUTED}" stroke-width="2" stroke-dasharray="3 4"/>')
        s.append(txt(x0 + 6, y1 - 8, label, 20, INK, "start", 600))
        off = TP_OFF[policy]
        s.append(txt(x1 - 4, y1 - 8, f"calibrations: {len(cal)} · runs beyond 5 nm: {off}", 19, BAD if off else GREEN, "end", 600))
    s.append(txt((x0 + x1) / 2, h_ - 8, "run", 19, MUTED))
    s.append(txt(X_(112) + 6, 402, "thermocouple starts to fail (run 112)", 16, MUTED, "start"))
    for k in range(2):
        y0k = 175 + k * 205
        Yk = lambda v, y0k=y0k: y0k - v / 9 * 140
        s.append(txt(x0 + 8, Yk(5.0) - 6, "5 nm: the controller can no longer correct", 16, BAD, "start"))
        s.append(txt(x0 + 8, Yk(2.0) - 6, "2 nm alarm", 16, GOLD, "start"))
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ slides
FOOT = "Module 6 · Toolset twin: queues and maintenance"
slides = []


def slide(body, cls="slide"):
    n = len(slides) + 1
    foot = ('<div class="foot"><span>SemiAcademy · ISEP-CSU</span><span>CC BY 4.0</span></div>' if n == 1
            else f'<div class="foot"><span>{FOOT}</span><span>{n}</span></div>')
    slides.append(f'<!-- {n} -->\n<section class="{cls}">\n{body}\n  {foot}\n</section>\n')


def reveal(ans):
    return f'  <div class="reveal"><button type="button">Show the answer</button>\n    <div class="ans">{ans}</div></div>'


def two(fig, items, width=620, size=23):
    lis = "".join(f'<li style="font-size:{size}px">{i}</li>' for i in items)
    return f'<div class="cols"><div style="flex:0 0 {width}px">{fig}</div><div><ul style="padding-left:28px">{lis}</ul></div></div>'


LITTLE_FIG, LITTLE_AVG, LITTLE_AREA, LITTLE_T0 = little_fig()
PROC = BASE_LINE["furnace_h"] + BASE_LINE["litho_h"] + BASE_LINE["etch_h"]
WAIT = float(STAGE["queue_h"].sum())
LIT = STAGE.loc["litho"]

slide(f"""  <div class="kicker">Digital Twins for Semiconductor Manufacturing · Module 6</div>
  <h1>Toolset twin: queues and maintenance</h1>
  <p style="font-size:30px;max-width:960px">From one furnace to a line: where lots wait, what a change to one tool does to the whole line, and what maintenance costs.</p>
  <div style="margin-top:6px">{line_fig(1060, 235, "a0")}</div>""", "slide title-slide")

slide("""  <h2>By the end of this module you can</h2>
  <ul>
    <li>measure <strong>throughput, cycle time and work in progress</strong> at the same boundary, and check them with <strong>Little's law</strong>;</li>
    <li>explain why <strong>queues grow faster than load</strong>, and find the <strong>bottleneck</strong> of a line;</li>
    <li>use and explain a supplied <strong>discrete-event simulation</strong> of lots, tools and queues in <strong>SimPy</strong>, and implement its measurement and maintenance functions;</li>
    <li>choose a <strong>warm-up</strong>, run <strong>replications</strong> and report a mean with a <strong>confidence interval</strong>;</li>
    <li>compare a <strong>what-if change</strong> by the mean and spread over replications, on the same random numbers;</li>
    <li>compare <strong>preventive</strong> and <strong>predictive</strong> maintenance using the Module 5 drift estimate, and say what a three-tool model leaves out.</li>
  </ul>
  <p class="small" style="margin-top:12px">Course outcome 6. The lab runs a supplied SimPy model of a furnace, lithography and etch line.</p>""")

slide(f"""  <h2>In a line, nearly half of a lot's time is waiting</h2>
  {two(where_fig(), [f"The lab's line: a lot needs {PROC:.1f} hours of processing in all, and spends {CT_M:.1f} hours in the line.", f"The other {WAIT:.1f} hours are spent in queues, and {float(STAGE.loc['litho', 'queue_h']):.1f} of them in front of <strong>lithography</strong>.", "A twin of one tool cannot see this: the wait depends on the other lots, the tools they use and how many tools there are.", "Simulated line, mean of 20 replications after the warm-up."], size=22)}""")

slide(f"""  <h2>Five numbers, one boundary</h2>
  <table style="font-size:23px;width:100%">
    <tr><th>name</th><th>what it counts</th><th>unit</th></tr>
    <tr><td><strong>queue</strong></td><td>lots waiting for a tool</td><td>lots</td></tr>
    <tr><td><strong>utilization</strong></td><td>tool-hours spent processing, as a share of the tool-hours available</td><td>0 to 1</td></tr>
    <tr><td><strong>throughput</strong> λ</td><td>lots that leave per hour</td><td>lots/h</td></tr>
    <tr><td><strong>cycle time</strong> W</td><td>time from entering to leaving: waiting and processing</td><td>h</td></tr>
    <tr><td><strong>work in progress</strong> L</td><td>average number of lots inside the boundary</td><td>lots</td></tr>
  </table>
  <ul style="font-size:22px;margin-top:14px">
    <li>The boundary can be one tool, a tool group or the whole line. Throughput, cycle time and work in progress must describe <strong>the same one</strong>, over the same window.</li>
    <li>In the lab: throughput {THR_M:.2f} lots/h, cycle time {CT_M:.2f} h, {WIP_M:.1f} lots in the line (the whole line, after the warm-up).</li>
  </ul>""")

slide(f"""  <h2>Little's law</h2>
  <div class="eq" style="font-size:34px">L = λ · W</div>
  {two(LITTLE_FIG, ["The staircase is the number of lots in the line. Its <strong>area</strong> can be counted two ways: slice by slice of time, which gives L times the window, or lot by lot, which gives the sum of the hours each lot spends in the line.", "Little (1961) proved it for any arrival pattern and any service times. It needs only the same boundary and a steady state.", f"Lab line, hours {LITTLE_T0:.0f} to {LITTLE_T0 + 40:.0f} of one replication: an area of {LITTLE_AREA:.0f} lot-hours over 40 hours is {LITTLE_AVG:.1f} lots; adding up the hours of each lot inside the window gives the same {LITTLE_AREA:.0f}."], size=21)}""")

LAM = THR_M
W_LIT = float(LIT["in_group_h"])
L_LIT = float(LIT["in_group_lots"])
assert abs(LAM * CT_M / WIP_M - 1) < 0.01 and abs(LAM * W_LIT / L_LIT - 1) < 0.01
assert LAM * W_LIT < 0.6 * WIP_M
slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>Little's law</h2>
  <p>The line releases 0.85 lots per hour and lots leave at {LAM:.3f} lots per hour. A lot spends {CT_M:.2f} hours in the line on average, and {W_LIT:.2f} hours in the lithography group, waiting and processing together.</p>
  <p><strong>Predict:</strong> how many lots are in the line on average, and how many are in the lithography group? Then: what would you get by multiplying {LAM:.3f} by lithography's {W_LIT:.2f} hours and comparing it with the lots in the <em>whole line</em>?</p>
{reveal(f"Whole line: L = λ · W = {LAM:.3f} × {CT_M:.2f} = {LAM * CT_M:.1f} lots; the simulation measures {WIP_M:.1f}. Lithography group: {LAM:.3f} × {W_LIT:.2f} = {LAM * W_LIT:.1f} lots, and the simulation measures {L_LIT:.1f}. Mixing the two, {LAM * W_LIT:.1f} against the line's {WIP_M:.1f} lots, gives numbers that do not match: the law has not failed, it was applied to two different systems. L, λ and W must describe the same boundary.")}""")

slide(f"""  <h2>Waiting grows faster than load</h2>
  {two(sweep_fig(), ["For one tool, the mean wait in the queue is about <strong>(variability) × (utilization) × (process time)</strong>: Wq ≈ (c_a² + c_s²)/2 · ρ/(1 − ρ) · t_e (Kingman, 1961; Hopp and Spearman, 2011).", "The factor ρ/(1 − ρ) is 1 at 50 % utilization, 4 at 80 %, 9 at 90 %, 19 at 95 %.", f"In the simulated line the cycle time rises from {SWEEP[0.5][0][0]:.1f} h at a load of 0.5 to {SWEEP[0.85][0][0]:.1f} h at 0.84 and {SWEEP[0.93][0][0]:.1f} h at 0.92, and the spread between replications grows with it.", "Small changes at a busy tool matter much more than at an idle one."], size=21)}""")

slide(f"""  <h2>The line in this module</h2>
  <div>{line_fig(1000, 221, "a2")}</div>
  <table style="font-size:19px;width:100%;margin-top:4px">
    <tr><th>tool group</th><th>tools</th><th>time per lot</th><th>variation (c)</th><th>capacity (lots/h)</th><th>utilization at 0.85 lots/h</th></tr>
    <tr><td>furnace</td><td>2 tubes</td><td>1.8 h</td><td>0.30</td><td>{2 / 1.8:.2f}</td><td>{float(STAGE.loc['furnace', 'utilization']):.2f}</td></tr>
    <tr><td>lithography</td><td>1 tool</td><td>1.0 h</td><td>0.50</td><td>{1 / 1.0:.2f}</td><td>{float(STAGE.loc['litho', 'utilization']):.2f}</td></tr>
    <tr><td>etch</td><td>2 chambers</td><td>1.2 h</td><td>0.40</td><td>{2 / 1.2:.2f}</td><td>{float(STAGE.loc['etch', 'utilization']):.2f}</td></tr>
  </table>
  <p class="small" style="margin-top:8px">Lots arrive at random times, one product, first in first out, no re-entrant flow. The furnace time covers load, 90 min of oxidation and unload. Lithography is the bottleneck: the smallest capacity, the highest utilization.</p>""")

slide("""  <h2>Discrete-event simulation with SimPy</h2>
  <div class="cols">
    <div style="flex:0 0 500px"><ul style="padding-left:24px">
      <li style="font-size:22px">The state changes only at <strong>events</strong>: a lot arrives, a tool becomes free, a process ends.</li>
      <li style="font-size:22px">The simulation keeps a list of future events and jumps to the next one: hours of waiting cost no computing time.</li>
      <li style="font-size:22px">SimPy writes each lot as a <strong>process</strong> (a Python generator) and each tool group as a <strong>resource</strong> with a queue.</li>
      <li style="font-size:22px">Random numbers make every run different: each run is a <strong>replication</strong>, started from a seed.</li>
    </ul></div>
    <div style="flex:1"><pre style="background:#F8F4F6;border-left:6px solid #99004C;padding:16px 20px;font-size:19px;line-height:1.5;margin:0;font-family:Consolas,'Courier New',monospace">def lot(i):
    tube = yield tubes.get()
    yield env.timeout(furnace_time[i])
    yield tubes.put(tube)

    with litho.request() as req:
        yield req            # wait here
        yield env.timeout(litho_time[i])

    with etch.request() as req:
        yield req
        yield env.timeout(etch_time[i])</pre>
    <p class="small" style="margin-top:10px">Simplified from the lab's line.py; SimPy 4.1.2, MIT license.</p></div>
  </div>""")

slide(f"""  <h2>One replication</h2>
  <div>{wip_fig()}</div>
  <ul style="font-size:22px">
    <li>The line starts <strong>empty</strong>, fills within a day, and then moves around seven lots. The average after the warm-up is {float(REP.loc[0, 'wip']):.1f} lots in this replication, {WIP_M:.1f} over the 20.</li>
    <li>One replication is one random run of the line. Everything measured from it, a mean cycle time or an average WIP, has a spread that only several replications can show.</li>
  </ul>""")

slide(f"""  <h2>Warm-up: when does the line settle?</h2>
  {two(welch_fig(), ["The first lots meet no queue, so the first hours are not typical of the running line.", "<strong>Welch's method</strong> (1983): average the replications bin by bin, smooth with a moving average, and read where the curve levels off.", f"Here the curve rises from {WELCH[1][0]:.0f} lots in the first 10 hours and levels near 7 after about 120 hours. The lab discards the first <strong>200 hours</strong>.", "The bias of keeping the warm-up is small beside the noise of 20 replications, and discarding it costs a fifth of the run: a cheap safeguard."], size=21)}""")

slide(f"""  <h2>Replications and the confidence interval</h2>
  {two(reps_fig(), [f"The 20 replications give mean cycle times from {CT.min():.1f} to {CT.max():.1f} hours: one run says little.", "They are independent runs of one line, so their mean estimates the line's long-run value, and <strong>mean ± t · s / √n</strong> is a 95 % confidence interval.", f"With n = 20, t = {stats.t.ppf(0.975, 19):.2f}: mean {CT_M:.2f} h, half-width {CT_H:.2f} h.", f"The half-width shrinks about as 1/√n: {HW[5]:.2f} h with 5 replications, {HW[10]:.2f} with 10, {HW[20]:.2f} with 20."], size=21)}""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>How many replications?</h2>
  <p>With 20 replications the half-width of the interval for the mean cycle time is {CT_H:.2f} hours.</p>
  <p><strong>Predict:</strong> about how many replications would it take to bring the half-width down to 0.2 hours?</p>
{reveal(f"The half-width goes as 1/√n, so halving it takes four times as many replications. From {CT_H:.2f} to 0.2 is a factor of {CT_H / 0.2:.2f}, which takes about {CT_H / 0.2:.2f}² × 20 = {N_NEEDED} replications (a few fewer, because t also falls as n grows). Precision is expensive: ten times the precision costs a hundred times the runs.")}""")

slide(f"""  <h2>Where the time goes: the bottleneck</h2>
  {two(bottleneck_fig(), [f"<strong>Lithography</strong> has the highest utilization, {float(LIT['utilization']):.2f}, and the longest wait, {float(LIT['queue_h']):.1f} h.", f"The furnace, at {float(STAGE.loc['furnace', 'utilization']):.2f} with two tubes, waits {float(STAGE.loc['furnace', 'queue_h']):.1f} h. Etch, at {float(STAGE.loc['etch', 'utilization']):.2f}, almost never makes a lot wait.", "The bottleneck is where a change to a process time buys the most cycle time; elsewhere it buys little.", "Mean of 20 replications, after the warm-up."], size=22)}""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>Which 10 % is worth more?</h2>
  <p>Two changes, each shortening a process time by 10 %: the etch time from 1.2 to 1.08 hours, or the lithography time from 1.0 to 0.9 hours. The mean cycle time is {CT_M:.1f} hours.</p>
  <p><strong>Predict:</strong> which change shortens the cycle time more, and by roughly how much?</p>
{reveal(f"Lithography, by {WI_LITHO['dmean']:.2f} h (interval ± {WI_LITHO['dhalf']:.2f}). Etch saves {WI_ETCH['dmean']:.2f} h, about what its shorter process time removes ({0.12:.2f} h). Lithography is the bottleneck: speeding it shortens its queue as well as its process time.")}""")

slide(f"""  <h2>What-if changes, compared by mean and spread</h2>
  {two(whatif_fig(), [f"Each bar is the <strong>paired</strong> mean reduction in cycle time over 20 seeds, with its 95 % interval.", f"Etch: {WI_ETCH['dmean']:.2f} h. Furnace: {WI['furnace 10 % faster']['dmean']:.2f} h. Lithography: {WI_LITHO['dmean']:.2f} h ± {WI_LITHO['dhalf']:.2f}.", f"The faster lithography line's interval ({WI_LITHO['mean'] - WI_LITHO['half']:.1f} to {WI_LITHO['mean'] + WI_LITHO['half']:.1f} h) does not overlap the baseline's ({CT_M - CT_H:.1f} to {CT_M + CT_H:.1f} h); the etch line's does.", "A change to a tool that is not the bottleneck is real but small."], size=21)}""")

slide(f"""  <h2>Same random numbers, paired differences</h2>
  {two(pair_fig(), [f"Between seeds, the baseline's cycle time ranges over about {CT.max() - CT.min():.0f} hours. The etch change saves {WI_ETCH['diff'].min():.2f} to {WI_ETCH['diff'].max():.2f} hours.", "Replications with the same seed use the same lots, arrival times and unit process times in both lines. Each seed gives one difference, and the 20 differences go through the same interval formula.", f"Paired: {WI_ETCH['dmean']:.3f} ± {WI_ETCH['dhalf']:.3f} h, positive in all 20. Unpaired, the two intervals (± {CT_H:.2f} h each) overlap and cannot tell.", "Say which you did: paired differences are sharper, and only valid when both lines see the same random numbers."], size=21)}""")

slide(f"""  <h2>Maintenance with the Module 5 drift estimate</h2>
  <div>{tube_fig()}</div>
  <p class="small" style="margin-top:0">Blue: the tube's true drift; maroon: the Module 5 estimate from one monitor wafer per run; green: calibrations (8 hours out of service each). The same tube and the same noise under both rules; the thermocouple starts to fail at run 112.</p>""")

slide(f"""  <h2>Three policies on the line, 20 replications each</h2>
  <table style="font-size:18px;width:100%">
    <tr><th>policy</th><th>completed calibrations</th><th>off-target lots</th><th>most in one replication</th><th>failing tube ran (median runs)</th><th>cycle time (h)</th></tr>
    <tr><td>none</td><td>0</td><td style="white-space:nowrap">{POLICY['none']['off'][0]:.0f} ± {POLICY['none']['off'][1]:.0f} of {POLICY['none']['lots']:.0f}</td><td>{POLICY['none']['off_max']}</td><td>-</td><td style="white-space:nowrap">{POLICY['none']['ct'][0]:.2f} ± {POLICY['none']['ct'][1]:.2f}</td></tr>
    <tr><td>preventive, every 50 runs</td><td>{POLICY['preventive']['cal'][0]:.1f} ± {POLICY['preventive']['cal'][1]:.1f}</td><td>{POLICY['preventive']['off'][0]:.0f} ± {POLICY['preventive']['off'][1]:.0f}</td><td>{POLICY['preventive']['off_max']}</td><td>{POLICY['preventive']['fault_med']:.0f}</td><td style="white-space:nowrap">{POLICY['preventive']['ct'][0]:.2f} ± {POLICY['preventive']['ct'][1]:.2f}</td></tr>
    <tr><td>predictive, 2.0 nm for 3 runs</td><td>{POLICY['predictive']['cal'][0]:.1f} ± {POLICY['predictive']['cal'][1]:.1f}</td><td>{(f"{POLICY['predictive']['off'][0]:.0f} ± {POLICY['predictive']['off'][1]:.0f}" if POLICY['predictive']['off_max'] else 'none observed')}</td><td>{POLICY['predictive']['off_max']}</td><td>{POLICY['predictive']['fault_med']:.0f}</td><td style="white-space:nowrap">{POLICY['predictive']['ct'][0]:.2f} ± {POLICY['predictive']['ct'][1]:.2f}</td></tr>
  </table>
  <ul style="font-size:20px;margin-top:10px">
    <li>The predictive rule makes fewer calibrations than the preventive rule, because it calibrates each tube when its own estimate says so. No off-target lot was observed in these 20 replications, which does not establish zero risk.</li>
    <li>Calibrations cost capacity: one tube out for 8 hours leaves the other short, and cycle time rises from {POLICY['none']['ct'][0]:.1f} to {POLICY['predictive']['ct'][0]:.1f} h. Whether that pays depends on the cost of an off-target lot, which the model does not know.</li>
  </ul>
  <p class="small" style="margin-top:6px">Model assumptions: compensation does not lengthen the furnace step, and a calibration also repairs a failing thermocouple.</p>""")

slide("""  <h2>What a three-tool model leaves out</h2>
  <div class="cols">
    <div class="card"><h3>Re-entrant flow</h3><p>A lot returns to lithography and etch for every layer, dozens of times. The bottleneck depends on the layer mix, and lots of different ages compete for the same tools.</p></div>
    <div class="card"><h3>Many products and priorities</h3><p>Different recipes, set-ups between them, rush lots that jump the queue.</p></div>
  </div>
  <div class="cols" style="margin-top:18px">
    <div class="card"><h3>Batching rules</h3><p>A real furnace processes many lots at once, and the rule for filling a batch changes the queues.</p></div>
    <div class="card"><h3>Other things that go wrong</h3><p>Random breakdowns, operators, transport, rework loops. The model has drift and calibration only.</p></div>
  </div>
  <p class="small" style="margin-top:14px">Each can move the bottleneck. The model answers the questions it was built for, such as how queues respond to load and to one change; it needs data from a real line before it is trusted for more.</p>""")

slide("""  <h2>Summary</h2>
  <ul style="font-size:24px">
    <li>Throughput, cycle time and work in progress are tied by <strong>L = λW</strong>, when all three describe the same boundary.</li>
    <li>Queues grow faster than load: the busiest tool, the <strong>bottleneck</strong>, decides where a change matters.</li>
    <li>A discrete-event simulation gives each statistic as a <strong>mean with an interval</strong>, from replications that start after a warm-up.</li>
    <li>Compare what-if changes on <strong>the same random numbers</strong>, and report the paired difference with its interval.</li>
    <li>The Module 5 drift estimate turns maintenance from a calendar into a decision for each tube, and the twin shows what that costs.</li>
  </ul>
  <div class="eq" style="font-size:21px;margin-top:10px"><strong>Next:</strong> the lab, with five functions, 20 replications, two what-if changes and three maintenance policies.</div>""")

slide("""  <h2>Credits and license</h2>
  <p>These slides were written for SemiAcademy's <em>Digital Twins for Semiconductor Manufacturing</em> course, developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). License: <strong>CC BY 4.0</strong>, https://creativecommons.org/licenses/by/4.0/.</p>
  <p style="font-size:19px">Sources: J. D. C. Little, "A proof for the queuing formula: L = λW", Operations Research 9(3), 383-387 (1961), and "Little's law as viewed on its 50th anniversary", Operations Research 59(3), 536-549 (2011); W. J. Hopp and M. L. Spearman, <em>Factory Physics</em>, 3rd ed., Waveland Press (2011); J. F. C. Kingman, "The single server queue in heavy traffic", Mathematical Proceedings of the Cambridge Philosophical Society 57(4), 902-904 (1961); P. D. Welch, "The statistical analysis of simulation results", in S. S. Lavenberg (ed.), <em>The Computer Performance Modeling Handbook</em>, Academic Press, 268-328 (1983); A. M. Law, <em>Simulation Modeling and Analysis</em>, 5th ed., McGraw-Hill (2015); SimPy 4.1.2 (MIT license), simpy.readthedocs.io; SEMI E10, Specification for Definition and Measurement of Equipment Reliability, Availability, and Maintainability (RAM) and Utilization. Cited, not reproduced.</p>
  <p class="small">Every data value on these slides is simulated with the course's line model (line.py 1.0.0, seeds 1 to 20). Source: github.com/CSU-Physics/digital-twins-notebooks (media/m6/m6-slides.html).</p>""")

# ------------------------------------------------------------------ assemble
head_html = (T / "deck_head.html").read_text()
head_html = head_html.replace("<title>Module 1 Slides: From Bits to Qubits</title>", "<title>Module 6 Slides: Toolset Twin</title>")
assert "Toolset Twin" in head_html
head_html = head_html.replace("</style>", "  code { font-family: Consolas, \"Courier New\", monospace; font-size:0.92em; }\n</style>", 1)
bar = (T / "deck_bar.html").read_text().replace("1 / 19", f"1 / {len(slides)}")
ctrl = (T / "deck_ctrl.js").read_text()
html = head_html + "\n" + "\n".join(slides) + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
for bad in ("—", "–", "${", "`"):
    assert bad not in html, bad
(OUT / "m6-slides.html").write_text(html)
print("wrote", OUT / "m6-slides.html", len(slides), "slides,", len(html), "bytes")
print("CT", round(CT_M, 3), round(CT_H, 3), "THR", round(THR_M, 3), "WIP", round(WIP_M, 3), "ratio", round(RATIO_M, 4), round(RATIO_H, 4), "HW", {k: round(v, 3) for k, v in HW.items()}, "N", N_NEEDED)
print("whatif", {k: (round(v["dmean"], 3), round(v["dhalf"], 3)) for k, v in WI.items()})
print("policy", {k: {a: (b if not isinstance(b, tuple) else tuple(round(x, 2) for x in b)) for a, b in v.items()} for k, v in POLICY.items()})
print("tube", {p: (cal, TP_OFF[p]) for p, (rows, cal) in TP.items()}, "little avg", round(LITTLE_AVG, 2), "val417", VAL_417)
