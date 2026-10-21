"""Build the Module 2 slide deck (media/m2/m2-slides.html) for Digital Twins for Semiconductor Manufacturing.

Every number and figure on the slides is computed here: Part A numbers from the SECOM data (content/m2/data) with the
lab's helpers and the reference functions of the Module 2 check, Part B numbers from the simulated records of the
course's furnace (content/m2/furnace.py, example key 2026). The slides therefore agree with the lab notebook.
Run from anywhere: python3 tools/build_dt_m2_media.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m2"), str(ROOT / "checks")]
from furnace import ON_TIME_H, fab_records  # noqa: E402
from secom_tools import evaluate, load_secom, time_split  # noqa: E402
from dt_m2_check import (hold_means, reference_align, reference_keep_signals,  # noqa: E402
                         reference_missing_fraction, reference_s2n)

T = ROOT / "tools"
OUT = ROOT / "media" / "m2"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, INK, MUTED, LINE, TINT, OK, BAD = "#99004C", "#24313D", "#6A626B", "#E4DCE0", "#F8F4F6", "#2E7D4F", "#B3261E"

# ------------------------------------------------------------------ Part A numbers (SECOM)
X, y, t = load_secom()
X_train, X_test, y_train, y_test = time_split(X, y, t)
NTR = len(X_train)
month = t.dt.to_period("M")
MONTHS = [(str(p), int(y[month == p].size), int(y[month == p].sum())) for p in sorted(month.unique())]
miss = reference_missing_fraction(X_train)
kept = reference_keep_signals(X_train)
share = X_train.apply(lambda c: c.dropna().value_counts(normalize=True).iloc[0] if c.notna().any() else 1.0)
N_DROP_MISS, N_DROP_CONST = int((miss > 0.5).sum()), int(((miss <= 0.5) & (share >= 0.95)).sum())
scores = reference_s2n(X_train[kept], y_train).sort_values(ascending=False)
top20 = list(scores.index[:20])
honest = evaluate(X_train, y_train, X_test, y_test, top20)
top20_all = list(reference_s2n(X[kept], y).sort_values(ascending=False).index[:20])
leaky = evaluate(X_train, y_train, X_test, y_test, top20_all)
rng = np.random.default_rng(2026)
SHUF = []
for i in range(5):
    ys = pd.Series(rng.permutation(y.to_numpy()), index=y.index, name="fail")
    a = evaluate(X_train, ys.iloc[:NTR], X_test, ys.iloc[NTR:], reference_s2n(X_train[kept], ys.iloc[:NTR]).nlargest(20).index)["auc"]
    b = evaluate(X_train, ys.iloc[:NTR], X_test, ys.iloc[NTR:], reference_s2n(X[kept], ys).nlargest(20).index)["auc"]
    SHUF.append((a, b))
S059 = [(str(p), float(X["s059"][month == p].mean()), float(y[month == p].mean())) for p in sorted(month.unique())]
S059_TR = float(reference_s2n(X_train[["s059"]], y_train).iloc[0])
S059_TE = float(reference_s2n(X_test[["s059"]], y_test).iloc[0])
A = dict(n=len(y), p=X.shape[1], fails=int(y.sum()), acc=float(np.mean(y == 0)), ntr=NTR, ftr=int(y_train.sum()), nte=len(y_test),
         fte=int(y_test.sum()), miss_all=float(X_train.isna().to_numpy().mean()), miss0=int((miss == 0).sum()), kept=len(kept),
         top=scores.index[0], top_s=float(scores.iloc[0]), honest=honest["auc"], leaky=leaky["auc"], h_found=honest["fails_found"],
         l_found=leaky["fails_found"], overlap=len(set(top20) & set(top20_all)), sh_h=float(np.mean([a for a, _ in SHUF])),
         sh_l=float(np.mean([b for _, b in SHUF])), acc_test=float(np.mean(y_test == 0)), h_acc=honest["accuracy"])
assert A["kept"] == 440 and f"{A['top_s']:.3f}" == "0.363" and f"{A['acc']:.3f}" == "0.934"
assert f"{A['honest']:.2f}" == "0.56" and f"{A['leaky']:.2f}" == "0.72"

# ------------------------------------------------------------------ Part B numbers (simulated, key 2026)
rec = fab_records(2026)
table = reference_align(rec["runs"], rec["wafers"], rec["metrology"])
hm = hold_means(rec["trace"])
table["hold_mean_C"] = table["run_id"].map(hm)
on_time = table[table["delay_h"] <= ON_TIME_H]
R = float(np.corrcoef(on_time["hold_mean_C"], on_time["thickness_nm"])[0, 1])
LATE = table[table["delay_h"] > ON_TIME_H].iloc[0]
MISSING = table[table["thickness_nm"].isna()].iloc[0]
SLOPE = float(np.polyfit(on_time["hold_mean_C"], on_time["thickness_nm"], 1)[0])
B = dict(r=R, late=LATE["run_id"], late_h=float(LATE["delay_h"]), missing=MISSING["run_id"], n_on=len(on_time),
         dmin=float(on_time["delay_h"].min()), dmax=float(on_time["delay_h"].max()), slope=SLOPE,
         th_min=float(on_time["thickness_nm"].min()), th_max=float(on_time["thickness_nm"].max()))
assert f"{R:.3f}" == "0.661" and B["n_on"] == 22
# inner join: the correlation does not change, but the table loses a run without saying so
INNER_N = int(table["thickness_nm"].notna().sum())

# an illustrative data budget (example numbers, not a statement about any real tool)
EX_SIG, EX_HZ, EX_MIN = 40, 1, 130
EX_VALUES = EX_SIG * EX_HZ * EX_MIN * 60


# ------------------------------------------------------------------ SVG helpers
def txt(x, y, s, size=20, fill=INK, anchor="middle", weight=400, extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{extra}>{s}</text>'


def box(x, y, w, h_, title, sub, on=True, title_size=22, sub_size=18):
    stroke, fill = (MAROON, "#FBF3F7") if on else (LINE, "#FFFFFF")
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h_}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
    out += txt(x + w / 2, y + 32, title, title_size, INK, weight=600)
    for k, ln in enumerate(sub):
        out += txt(x + w / 2, y + 58 + 23 * k, ln, sub_size, MUTED)
    return out


def arrow_def(mid, color=MUTED):
    return (f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0 0L10 5L0 10z" fill="{color}"/></marker></defs>')


def line(x1, y1, x2, y2, mid, color=MUTED, dash="", width=3, both=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    start = f' marker-start="url(#{mid})"' if both else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d} marker-end="url(#{mid})"{start}/>'


def axes(s, X_, Y_, x0, x1, y0, y1, xt, yt, xl, yl, yfmt="{:.0f}", xfmt=None):
    for v in yt:
        s.append(f'<line x1="{x0}" y1="{Y_(v):.1f}" x2="{x1}" y2="{Y_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y_(v) + 7, yfmt.format(v), 19, MUTED, "end"))
    for v in xt:
        s.append(txt(X_(v), y0 + 28, xfmt(v) if xfmt else f"{v:g}", 19, MUTED))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt((x0 + x1) / 2, y0 + 56, xl, 20, MUTED))
    s.append(txt(22, (y0 + y1) / 2, yl, 20, MUTED, extra=f' transform="rotate(-90 22 {(y0 + y1) / 2:.1f})"'))


# ------------------------------------------------------------------ figures
def title_fig():
    w, h_ = 700, 200
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" aria-hidden="true">', arrow_def("ti", MAROON)]
    names = [("MES", "runs, wafers"), ("Furnace", "traces"), ("Metrology", "thickness")]
    for k, (a, b) in enumerate(names):
        y_ = 4 + k * 66
        s.append(f'<rect x="10" y="{y_}" width="210" height="58" rx="10" fill="#FFFFFF" stroke="{LINE}" stroke-width="2.5"/>')
        s.append(txt(115, y_ + 25, a, 20, INK, weight=600))
        s.append(txt(115, y_ + 47, b, 16, MUTED))
        s.append(line(226, y_ + 29, 472, 100, "ti", MAROON, width=2.5))
    s.append(box(480, 55, 210, 90, "Twin", ["joined, checked data"], on=True))
    s.append("</svg>")
    return "".join(s)


def flow_fig():
    w, h_ = 1130, 250
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="A wafer moves from tool to tool: clean, oxidation, lithography, etch, deposition and so on, with metrology tools measuring between steps">',
         arrow_def("pf", MUTED)]
    steps = ["Clean", "Oxidation", "Lithography", "Etch", "Deposition", "Implant"]
    bw, gap = 156, 32
    for k, name in enumerate(steps):
        x = 6 + k * (bw + gap)
        s.append(box(x, 30, bw, 70, name, [], on=(name == "Oxidation"), title_size=20))
        if k < len(steps) - 1:
            s.append(line(x + bw + 3, 65, x + bw + gap - 3, 65, "pf", MUTED))
    for k, (name, what) in enumerate([("Metrology", "oxide thickness"), ("Metrology", "line width, overlay"), ("Metrology", "etch depth")]):
        x = 6 + (k + 1) * (bw + gap) - 8           # under oxidation, lithography and etch
        s.append(f'<rect x="{x}" y="150" width="{bw + 16}" height="64" rx="10" fill="#FFFFFF" stroke="{MAROON}" stroke-dasharray="6 5" stroke-width="2"/>')
        s.append(txt(x + (bw + 16) / 2, 177, name, 19, INK, weight=600))
        s.append(txt(x + (bw + 16) / 2, 202, what, 16, MUTED))
        s.append(line(x + (bw + 16) / 2, 104, x + (bw + 16) / 2, 146, "pf", MAROON, dash="5 4", width=2))
    s.append(txt(1124, 245, "hundreds of steps and many passes through the same tool types", 17, MUTED, "end"))
    s.append("</svg>")
    return "".join(s)


def path_fig():
    w, h_ = 1130, 420
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Data path: the furnace controller sends detailed traces to FDC by Interface A and events to the MES host by SECS/GEM over HSMS; the ellipsometer sends results to the MES; a data store joins them for the twin">',
         arrow_def("dp", MAROON), arrow_def("dq", MUTED)]
    s.append(box(10, 40, 250, 100, "Furnace", ["controller, sensors"], on=False))
    s.append(box(10, 250, 250, 100, "Ellipsometer", ["metrology tool"], on=False))
    s.append(box(440, 20, 250, 100, "FDC system", ["traces, summaries"], on=False))
    s.append(box(440, 250, 250, 100, "MES / host", ["runs, wafers, recipes"], on=False))
    s.append(box(870, 125, 250, 120, "Data store", ["joined records", "for the twin"], on=True))
    s.append(line(265, 75, 435, 70, "dp", MAROON))
    s.append(txt(350, 58, "Interface A (EDA)", 17, MAROON, weight=600))
    s.append(line(265, 120, 435, 270, "dp", MAROON))
    s.append(txt(318, 150, "SECS/GEM (HSMS)", 17, MAROON, "start", 600, extra=' transform="rotate(41 318 150)"'))
    s.append(line(265, 315, 435, 315, "dq", MUTED))
    s.append(txt(350, 340, "SECS/GEM", 17, MUTED, weight=600))
    s.append(line(695, 70, 865, 160, "dp", MAROON))
    s.append(txt(800, 92, "run_id, step, time", 17, INK, extra=' dx="30"'))
    s.append(line(695, 300, 865, 215, "dp", MAROON))
    s.append(txt(820, 290, "run_id, wafer_id", 17, INK, extra=' dx="20"'))
    s.append(txt(565, 385, "Each arrow can be late, lose records or carry a wrong ID; each connection is also a way in for an attacker.", 19, INK, weight=600))
    s.append("</svg>")
    return "".join(s)


def trace_fig():
    w, h_ = 620, 380
    x0, x1, y0, y1 = 96, w - 34, h_ - 70, 44
    r1 = rec["trace"][rec["trace"]["run_id"] == rec["runs"]["run_id"].iloc[0]]
    lo, hi = 998.0, 1002.0
    X_ = lambda v: x0 + v / 3600 * (x1 - x0)
    Y_ = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="The 60-minute hold trace of one run, a reading every 10 s, with its mean">']
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 600, 1200, 1800, 2400, 3000, 3600], [998, 999, 1000, 1001, 1002], "time from the start of the hold (s)", "zone temperature (°C)")
    for a, b in zip(r1["time_s"], r1["zone_temp_C"]):
        if not np.isnan(b):
            s.append(f'<circle cx="{X_(a):.1f}" cy="{Y_(b):.1f}" r="2.6" fill="{INK}" fill-opacity="0.65"/>')
    m = float(r1["zone_temp_C"].mean())
    s.append(f'<line x1="{x0}" y1="{Y_(m):.1f}" x2="{x1}" y2="{Y_(m):.1f}" stroke="{MAROON}" stroke-width="4"/>')
    s.append(f'<line x1="{x1 - 250}" y1="{y1 - 22}" x2="{x1 - 214}" y2="{y1 - 22}" stroke="{MAROON}" stroke-width="4"/>')
    s.append(txt(x1 - 206, y1 - 15, f"hold mean {m:.2f} °C", 20, MAROON, "start", 600))
    s.append("</svg>")
    return "".join(s), m


def join_fig():
    w, h_ = 1130, 360
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="metrology joins wafers on wafer_id, which gives run_id; the run averages join runs on run_id; the trace means join runs on run_id">',
         arrow_def("jn", MAROON)]

    def tab(x, name, cols, on=False, wd=236):
        y, hh = 10, 44 + 30 * len(cols)
        out = f'<rect x="{x}" y="{y}" width="{wd}" height="{hh}" rx="8" fill="{"#FBF3F7" if on else "#FFFFFF"}" stroke="{MAROON if on else LINE}" stroke-width="2.5"/>'
        out += txt(x + wd / 2, y + 30, name, 21, INK, weight=600)
        for k, c in enumerate(cols):
            key = c.startswith("*")
            out += txt(x + 16, y + 64 + 30 * k, c.lstrip("*"), 18, MAROON if key else MUTED, "start", 600 if key else 400)
        return out

    def num(x, y, n):
        return f'<circle cx="{x}" cy="{y}" r="15" fill="{MAROON}"/>' + txt(x, y + 6, n, 17, "#FFFFFF", weight=700)

    s.append(tab(6, "metrology", ["*wafer_id", "measured_at", "thickness_nm"]))
    s.append(tab(296, "wafers (MES)", ["*wafer_id", "*run_id", "slot"]))
    s.append(tab(586, "runs (MES)", ["*run_id", "end_time", "start_time"], on=True))
    s.append(tab(888, "trace", ["*run_id", "time_s", "zone_temp_C"]))
    s.append(line(246, 75, 290, 75, "jn", MAROON, both=True))
    s.append(num(268, 120, "1"))
    s.append(line(536, 105, 580, 105, "jn", MAROON))
    s.append(num(558, 150, "2"))
    s.append(line(884, 75, 826, 75, "jn", MAROON))
    s.append(num(855, 120, "3"))
    notes = ["metrology joins wafers on wafer_id: each result gets the run_id of its wafer.",
             "average thickness_nm and the latest measured_at by run_id, then a left join onto runs: every run stays.",
             "the hold mean of each trace, by run_id, joins runs the same way."]
    for k, n in enumerate(notes):
        s.append(num(22, 222 + 40 * k, str(k + 1)))
        s.append(txt(48, 228 + 40 * k, n, 19, INK, "start"))
    s.append(txt(6, 350, "Key columns in maroon. The ellipsometer knows the wafer, not the run: the MES wafer history links the two.", 18, MUTED, "start"))
    s.append("</svg>")
    return "".join(s)


def timeline_fig():
    w, h_ = 1130, 380
    runs = rec["runs"]
    t0 = runs["start_time"].min().floor("D")
    days = (runs["end_time"].max() - t0) / pd.Timedelta(days=1) + 1.6
    x0, x1, y0 = 40, w - 20, 300
    X_ = lambda ts: x0 + ((ts - t0) / pd.Timedelta(days=1)) / days * (x1 - x0)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Timeline of 24 runs: each run end and the arrival of its metrology; one arrives after more than a day, one never arrives">',
         arrow_def("tl", MUTED), arrow_def("tlb", BAD)]
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    for d in range(int(days) + 1):
        xd = X_(t0 + pd.Timedelta(days=d))
        s.append(f'<line x1="{xd:.1f}" y1="{y0}" x2="{xd:.1f}" y2="{y0 + 8}" stroke="{INK}" stroke-width="2"/>')
        s.append(txt(xd + 4, y0 + 30, (t0 + pd.Timedelta(days=d)).strftime("%d %b"), 18, MUTED, "start"))
    j = 0
    for k, r in table.iterrows():
        xe = X_(r["end_time"])
        if np.isnan(r["delay_h"]):
            lane = 70
        elif r["delay_h"] > ON_TIME_H:
            lane = 30
        else:
            lane = 120 + (j % 4) * 44
            j += 1
        s.append(f'<rect x="{xe - 4:.1f}" y="{lane - 10}" width="8" height="20" fill="{INK}"/>')
        if np.isnan(r["delay_h"]):
            s.append(txt(xe + 10, lane + 7, "✕ no metrology", 17, BAD, "start", 600))
            continue
        xm = X_(r["last_measured_at"])
        late = r["delay_h"] > ON_TIME_H
        s.append(f'<line x1="{xe + 5:.1f}" y1="{lane}" x2="{xm:.1f}" y2="{lane}" stroke="{BAD if late else MUTED}" stroke-width="{3 if late else 2}" marker-end="url(#{"tlb" if late else "tl"})"/>')
        if late:
            s.append(txt(xm + 8, lane + 7, f"{r['delay_h']:.0f} h: late", 17, BAD, "start", 600))
    s.append(f'<rect x="{x0}" y="{h_ - 34}" width="8" height="20" fill="{INK}"/>')
    s.append(txt(x0 + 16, h_ - 18, "end of a run", 18, INK, "start"))
    s.append(f'<line x1="{x0 + 190}" y1="{h_ - 24}" x2="{x0 + 240}" y2="{h_ - 24}" stroke="{MUTED}" stroke-width="2" marker-end="url(#tl)"/>')
    s.append(txt(x0 + 250, h_ - 18, f"its last metrology record: {B['dmin']:.1f} to {B['dmax']:.1f} h later when on time", 18, INK, "start"))
    s.append("</svg>")
    return "".join(s)


def scatter_fig():
    w, h_ = 600, 380
    x0, x1, y0, y1 = 96, w - 16, h_ - 70, 20
    xs, ys_ = on_time["hold_mean_C"], on_time["thickness_nm"]
    lo_x, hi_x = 998.5, 1001.5
    lo_y, hi_y = 39.0, 40.4
    X_ = lambda v: x0 + (v - lo_x) / (hi_x - lo_x) * (x1 - x0)
    Y_ = lambda v: y0 - (v - lo_y) / (hi_y - lo_y) * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Average thickness against mean hold temperature for the 22 on-time runs, rising with temperature">']
    axes(s, X_, Y_, x0, x1, y0, y1, [999, 1000, 1001], [39.0, 39.4, 39.8, 40.2], "mean hold temperature (°C)", "average thickness (nm)", "{:.1f}")
    for a, b in zip(xs, ys_):
        s.append(f'<circle cx="{X_(a):.1f}" cy="{Y_(b):.1f}" r="7" fill="{MAROON}" fill-opacity="0.85"/>')
    s.append(f'<rect x="{X_(LATE["hold_mean_C"]) - 7:.1f}" y="{Y_(LATE["thickness_nm"]) - 7:.1f}" width="14" height="14" fill="none" stroke="{INK}" stroke-width="2.5"/>')
    s.append(txt(x0 + 14, y1 + 22, f"r = {R:.3f} over {B['n_on']} on-time runs", 20, INK, "start", 600))
    s.append(txt(x0 + 14, y1 + 48, "square: the late run", 18, MUTED, "start"))
    s.append("</svg>")
    return "".join(s)


def missing_fig():
    w, h_ = 600, 380
    x0, x1, y0, y1 = 96, w - 16, h_ - 70, 20
    edges = [0, 1e-12, 0.05, 0.2, 0.5, 0.9, 1.0001]
    labels = ["none", "up to 5%", "5–20%", "20–50%", "50–90%", "over 90%"]
    counts = [int(((miss >= a) & (miss < b)).sum()) if k else int((miss == 0).sum()) for k, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]
    counts[1] = int(((miss > 0) & (miss <= 0.05)).sum())
    counts[2] = int(((miss > 0.05) & (miss <= 0.2)).sum())
    counts[3] = int(((miss > 0.2) & (miss <= 0.5)).sum())
    counts[4] = int(((miss > 0.5) & (miss <= 0.9)).sum())
    counts[5] = int((miss > 0.9).sum())
    assert sum(counts) == X.shape[1]
    top = 450
    Y_ = lambda v: y0 - v / top * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Number of SECOM signals by fraction of missing values in the training records">']
    for v in range(0, top + 1, 100):
        s.append(f'<line x1="{x0}" y1="{Y_(v):.1f}" x2="{x1}" y2="{Y_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y_(v) + 7, v, 19, MUTED, "end"))
    bw = (x1 - x0) / len(counts)
    for k, (c, lab) in enumerate(zip(counts, labels)):
        cx = x0 + bw * (k + 0.5)
        s.append(f'<rect x="{cx - 34:.1f}" y="{Y_(c):.1f}" width="68" height="{y0 - Y_(c):.1f}" fill="{BAD if k >= 4 else MAROON}" rx="3"/>')
        s.append(txt(cx, Y_(c) - 10, c, 20, INK, weight=600))
        s.append(txt(cx, y0 + 28, lab, 16, MUTED))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt((x0 + x1) / 2, y0 + 56, "share of training values missing", 20, MUTED))
    s.append(txt(22, (y0 + y1) / 2, "number of signals", 20, MUTED, extra=f' transform="rotate(-90 22 {(y0 + y1) / 2:.1f})"'))
    s.append("</svg>")
    return "".join(s), counts


def months_fig():
    w, h_ = 600, 380
    x0, x1, y0, y1 = 96, w - 16, h_ - 70, 30
    top = 0.25
    Y_ = lambda v: y0 - v / top * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Share of fails by month: July 22 percent, August 9, September 3, October 6">']
    for v in (0, 0.05, 0.10, 0.15, 0.20, 0.25):
        s.append(f'<line x1="{x0}" y1="{Y_(v):.1f}" x2="{x1}" y2="{Y_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y_(v) + 7, f"{v:.0%}", 19, MUTED, "end"))
    bw = (x1 - x0) / len(MONTHS)
    names = {"2008-07": "Jul", "2008-08": "Aug", "2008-09": "Sep", "2008-10": "Oct"}
    for k, (p, n, f) in enumerate(MONTHS):
        cx = x0 + bw * (k + 0.5)
        v = f / n
        s.append(f'<rect x="{cx - 44:.1f}" y="{Y_(v):.1f}" width="88" height="{y0 - Y_(v):.1f}" fill="{MAROON}" rx="3"/>')
        s.append(txt(cx, Y_(v) - 10, f"{v:.1%}", 20, INK, weight=600))
        s.append(txt(cx, y0 + 28, f"{names[p]} ({n})", 18, MUTED))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt((x0 + x1) / 2, y0 + 56, "month of 2008 (number of records)", 20, MUTED))
    s.append(txt(22, (y0 + y1) / 2, "share of records that failed", 20, MUTED, extra=f' transform="rotate(-90 22 {(y0 + y1) / 2:.1f})"'))
    s.append("</svg>")
    return "".join(s)


def auc_fig():
    w, h_ = 620, 400
    x0, x1, y0, y1 = 96, w - 16, h_ - 80, 30
    lo, hi = 0.4, 0.8
    Y_ = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    bars = [("always pass", 0.5, MUTED), ("training only", A["honest"], MAROON), ("all records", A["leaky"], BAD)]
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Test AUC: always pass 0.50, ranked on training records {A["honest"]:.2f}, ranked on all records {A["leaky"]:.2f}">']
    for v in (0.4, 0.5, 0.6, 0.7, 0.8):
        s.append(f'<line x1="{x0}" y1="{Y_(v):.1f}" x2="{x1}" y2="{Y_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y_(v) + 7, f"{v:.1f}", 19, MUTED, "end"))
    bw = (x1 - x0) / len(bars)
    for k, (lab, v, col) in enumerate(bars):
        cx = x0 + bw * (k + 0.5)
        s.append(f'<rect x="{cx - 52:.1f}" y="{Y_(v):.1f}" width="104" height="{y0 - Y_(v):.1f}" fill="{col}" rx="3"/>')
        s.append(txt(cx, Y_(v) - 10, f"{v:.2f}", 21, INK, weight=600))
        s.append(txt(cx, y0 + 28, lab, 18, MUTED))
    s.append(txt(x0 + bw * 2.5, y0 + 50, "(leakage)", 17, BAD))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt(22, (y0 + y1) / 2, "AUC on the test records", 20, MUTED, extra=f' transform="rotate(-90 22 {(y0 + y1) / 2:.1f})"'))
    s.append("</svg>")
    return "".join(s)


def s059_fig():
    w, h_ = 620, 380
    x0, x1, y0, y1 = 96, w - 80, h_ - 70, 30
    lo, hi = -2, 14
    Y_ = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    Yf = lambda v: y0 - v / 0.25 * (y0 - y1)
    names = ["Jul", "Aug", "Sep", "Oct"]
    X_ = lambda k: x0 + (x1 - x0) * (k + 0.5) / 4
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Monthly mean of s059 and monthly share of fails both fall from July to October">']
    for v in (0, 4, 8, 12):
        s.append(f'<line x1="{x0}" y1="{Y_(v):.1f}" x2="{x1}" y2="{Y_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y_(v) + 7, v, 19, MAROON, "end"))
    for v in (0, 0.1, 0.2):
        s.append(txt(x1 + 10, Yf(v) + 7, f"{v:.0%}", 19, INK, "start"))
    s.append(f'<polyline points="{" ".join(f"{X_(k):.1f},{Y_(m):.1f}" for k, (_, m, _) in enumerate(S059))}" fill="none" stroke="{MAROON}" stroke-width="4"/>')
    s.append(f'<polyline points="{" ".join(f"{X_(k):.1f},{Yf(f):.1f}" for k, (_, _, f) in enumerate(S059))}" fill="none" stroke="{INK}" stroke-width="3" stroke-dasharray="8 6"/>')
    for k, (_, m, f) in enumerate(S059):
        s.append(f'<circle cx="{X_(k):.1f}" cy="{Y_(m):.1f}" r="6" fill="{MAROON}"/>')
        s.append(f'<rect x="{X_(k) - 5:.1f}" y="{Yf(f) - 5:.1f}" width="10" height="10" fill="{INK}"/>')
        s.append(txt(X_(k), y0 + 28, names[k], 19, MUTED))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt(x0 + 10, y1 - 8, "mean of s059 (solid)", 18, MAROON, "start", 600))
    s.append(txt(x1, y1 - 8, "share of fails (dashed)", 18, INK, "end", 600))
    s.append(txt((x0 + x1) / 2, y0 + 56, "month of 2008", 20, MUTED))
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ slides
FOOT = "Module 2 · The fab as a data system"
slides = []


def slide(body, cls="slide"):
    n = len(slides) + 1
    foot = ('<div class="foot"><span>SemiAcademy · ISEP-CSU</span><span>CC BY 4.0</span></div>' if n == 1
            else f'<div class="foot"><span>{FOOT}</span><span>{n}</span></div>')
    slides.append(f'<!-- {n} -->\n<section class="{cls}">\n{body}\n  {foot}\n</section>\n')


def reveal(ans):
    return f'  <div class="reveal"><button type="button">Show the answer</button>\n    <div class="ans">{ans}</div></div>'


TRACE_SVG, M1 = trace_fig()
MISS_SVG, MISS_COUNTS = missing_fig()

slide(f"""  <div class="kicker">Digital Twins for Semiconductor Manufacturing · Module 2</div>
  <h1>The fab as a data system</h1>
  <p style="font-size:30px;max-width:860px">Where a twin's data come from, how they are joined, and how to judge what they can show.</p>
  <div style="margin-top:22px">{title_fig()}</div>""", "slide title-slide")

slide("""  <h2>By the end of this module you can</h2>
  <ul>
    <li>name the main <strong>records</strong> of a fab: equipment traces, metrology results and the MES run history;</li>
    <li>say what <strong>SECS/GEM</strong>, <strong>Interface A</strong> and <strong>OPC UA</strong> are for;</li>
    <li>join records by <strong>run and wafer IDs</strong>, and handle late and missing results;</li>
    <li>screen signals for missing values and near-constant values, and rank them <strong>inside the training records only</strong>;</li>
    <li>explain <strong>class imbalance</strong>, <strong>data leakage</strong> and why an association is not shown to be a cause;</li>
    <li>name integrity and security risks of connected tools, from <strong>NIST IR 8356</strong>.</li>
  </ul>""")

slide(f"""  <h2>A wafer's path through the fab</h2>
  <div style="margin-top:4px">{flow_fig()}</div>
  <p class="small" style="margin-top:14px">Each process tool works on lots of wafers and writes records about every run. Metrology tools measure some wafers between steps. The furnace in this course grows the oxide; an ellipsometer then measures its thickness on monitor wafers.</p>""")

slide("""  <h2>Three kinds of record</h2>
  <table style="font-size:22px;width:100%">
    <tr><th style="width:250px">Record</th><th>What it holds</th><th style="width:300px">In this course</th></tr>
    <tr><td><strong>Equipment trace</strong> (for FDC)</td><td>sensor readings during each recipe step: temperatures, pressures, flows, powers</td><td>the zone temperature every 10 s of the hold</td></tr>
    <tr><td><strong>Metrology result</strong></td><td>a measurement of a wafer after a step: thickness, line width, overlay, defects</td><td>oxide thickness of three monitor wafers per run</td></tr>
    <tr><td><strong>MES history</strong></td><td>which lot and wafers ran on which tool, with which recipe, and when</td><td>runs and wafers tables</td></tr>
  </table>
  <p class="small" style="margin-top:16px">FDC is fault detection and classification: software that watches traces for runs that look abnormal. The MES is the manufacturing execution system, which tracks every lot through the fab.</p>""")

slide("""  <h2>How tools talk to the fab</h2>
  <div class="cols">
    <div class="card"><h3>SECS/GEM</h3><p style="font-size:20px">SECS-II (SEMI E5) defines the messages; GEM (E30) defines how a tool behaves toward the host: states, events, alarms, recipes, data collection. HSMS (E37) carries the messages over TCP/IP. GEM300 adds carriers, process jobs and wafer tracking (E87, E40, E90, E94).</p></div>
    <div class="card"><h3>Interface A (EDA)</h3><p style="font-size:20px">SEMI's Equipment Data Acquisition standards (E120, E125, E132, E134, E164): web services with SOAP/XML over HTTP or HTTPS, for collecting detailed equipment data. It does not control the tool and runs alongside SECS/GEM.</p></div>
    <div class="card"><h3>OPC UA</h3><p style="font-size:20px">IEC 62541, a platform-independent standard for industrial data exchange used across manufacturing. It has security built in: messages can be signed, or signed and encrypted, with certificates and user authentication.</p></div>
  </div>
  <p class="small" style="margin-top:14px">Sources: SEMI Standards Watch, &ldquo;Intro to SEMI communication standards&rdquo; (2022); Cimetrix, SEMI EDA / Interface A overview; OPC Foundation. The standards are described here, not reproduced.</p>""")

slide(f"""  <h2>One run's data path</h2>
  <div style="margin-top:4px">{path_fig()}</div>""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>How much data is one run?</h2>
  <p>An example tool logs {EX_SIG} signals, {EX_HZ} reading per second each, for a {EX_MIN}-minute run. A fab runs this recipe 6 times a day on 10 such tools.</p>
  <p><strong>Predict:</strong> how many values per run, and per day for those tools?</p>
{reveal(f"Per run: {EX_SIG} × {EX_HZ} × {EX_MIN} × 60 = {EX_VALUES:,} values. Per day: {EX_VALUES:,} × 6 × 10 = {EX_VALUES * 60:,} values, for one recipe on ten tools. These are example numbers, not a statement about a real tool; real tools differ widely. That volume is why FDC systems keep <strong>summary indicators</strong> per step, such as the mean or the slope.")}""")

slide(f"""  <h2>Sampling, clocks and summaries</h2>
  <div class="cols">
    <div style="flex:0 0 620px">{TRACE_SVG}</div>
    <div>
      <ul style="padding-left:28px">
        <li style="font-size:23px">The trace counts seconds from the start of the hold; the MES writes clock times. Clock time = hold_start + time_s.</li>
        <li style="font-size:23px">Two systems' clocks can disagree: align them before joining by time.</li>
        <li style="font-size:23px">A missed reading is NaN, not 0: the mean skips it.</li>
        <li style="font-size:23px">The twin uses one number per run: the <strong>hold mean</strong>, {M1:.2f} °C here.</li>
      </ul>
    </div>
  </div>""")

slide(f"""  <h2>Joining the records by ID</h2>
  <div>{join_fig()}</div>""")

slide(f"""  <h2>Late and missing results</h2>
  <div>{timeline_fig()}</div>
  <p class="small" style="margin-top:0">Simulated records, key 2026. The twin can use a result only if it arrives within {ON_TIME_H:.0f} h, before the next lots are dispatched. Run {B['late']} arrived after {B['late_h']:.0f} h; run {B['missing']} has none.</p>""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>Left join or inner join?</h2>
  <p>An inner join keeps only rows that match in both tables; a left join keeps every run and fills NaN where there is no result.</p>
  <p><strong>Predict:</strong> with an inner join, how many rows does the table of 24 runs have, and what does the fab lose?</p>
{reveal(f"{INNER_N} rows: run {B['missing']} disappears without a message. The correlation over on-time runs does not change, but the table no longer shows that a result is missing, and nobody follows up the missing measurement. A left join keeps the run with NaN thickness: unknown, not zero.")}""")

slide(f"""  <h2>What the joined table shows</h2>
  <div class="cols">
    <div style="flex:0 0 600px">{scatter_fig()}</div>
    <div>
      <p style="font-size:24px">Runs that held hotter grew more oxide: about {B['slope']:.2f} nm per °C here, on oxide near 40 nm.</p>
      <p style="font-size:24px">The correlation is computed over the {B['n_on']} runs whose result arrived on time.</p>
      <p style="font-size:24px">In this simulator the link is a cause, by construction, and Module 3's physics says why. Data alone show only an association.</p>
    </div>
  </div>""")

slide(f"""  <h2>SECOM: real records from a fab</h2>
  <div class="cols">
    <div style="flex:0 0 600px">{MISS_SVG}</div>
    <div>
      <ul style="padding-left:28px">
        <li style="font-size:23px">{A['n']:,} production records, {A['p']} anonymized signals, a pass or fail line test; {A['fails']} fails.</li>
        <li style="font-size:23px">{A['miss_all']:.1%} of the training values are missing; only {A['miss0']} signals have none.</li>
        <li style="font-size:23px">Screening on the training records keeps {A['kept']}: {N_DROP_MISS} dropped for more than half missing, {N_DROP_CONST} as near-constant.</li>
      </ul>
      <p class="small">McCann and Johnston (2008), UCI Machine Learning Repository, doi 10.24432/C54305, CC BY 4.0.</p>
    </div>
  </div>""")

slide(f"""  <h2>The accuracy trap</h2>
  <div class="eq" style="font-size:30px">always predict pass: accuracy = 1463 / 1567 = {A['acc']:.3f}</div>
  <ul>
    <li>It finds <strong>none</strong> of the {A['fails']} fails, the records the fab cares about.</li>
    <li>Report numbers that look at each class: the share of fails found, the <strong>balanced accuracy</strong> (mean of the two classes' shares called correctly, 0.5 here), the <strong>AUC</strong> (0.5 is chance).</li>
    <li>Compare every model with this baseline. A model that does not beat it is a valid finding.</li>
  </ul>""")

slide(f"""  <h2>Split first, and keep time order</h2>
  <div class="cols">
    <div style="flex:0 0 600px">{months_fig()}</div>
    <div>
      <ul style="padding-left:28px">
        <li style="font-size:23px">Training: the earliest {A['ntr']} records ({A['ftr']} fails). Test: the latest {A['nte']} ({A['fte']} fails).</li>
        <li style="font-size:23px">The fab changed: 22% fails in July, 3% in September.</li>
        <li style="font-size:23px">Everything that learns from data (filling in, scaling, choosing signals, fitting) uses the training records only.</li>
      </ul>
    </div>
  </div>""")

slide(f"""  <h2>Leakage: the ranking saw the test labels</h2>
  <div class="cols">
    <div style="flex:0 0 620px">{auc_fig()}</div>
    <div>
      <p style="font-size:23px">Top 20 signals by S2N, same classifier, same test records. Only the ranking differs.</p>
      <p style="font-size:23px">Ranked on training records: AUC {A['honest']:.2f}, {A['h_found']} of {A['fte']} fails found. Ranked on all records: AUC {A['leaky']:.2f}, {A['l_found']} found.</p>
      <p style="font-size:23px">With shuffled labels (no real link), means over 5 shuffles: {A['sh_h']:.2f} honest, {A['sh_l']:.2f} leaky.</p>
    </div>
  </div>""")

slide("""  <div class="kicker">Pause and predict</div>
  <h2>Random features, picked with all the data</h2>
  <p>scikit-learn's guide to common pitfalls makes 200 samples with 10,000 random features and random labels. It keeps the 25 features most related to the labels, using all 200 samples, then splits into training and test sets.</p>
  <p><strong>Predict:</strong> the test accuracy, when chance is 0.5?</p>
""" + reveal("About <strong>0.76</strong> in the guide, from nothing: the selection saw the test samples. Selecting inside the training set, or inside a pipeline, gives about 0.5, as it should. Put every step that learns from data in a <code>Pipeline</code>, so that cross-validation repeats it inside every fold."))

slide(f"""  <h2>Associated is not shown to be a cause</h2>
  <div class="cols">
    <div style="flex:0 0 620px">{s059_fig()}</div>
    <div>
      <p style="font-size:23px">{A['top']} ranks first in the training records (S2N {S059_TR:.3f}), and almost nothing in the test records ({S059_TE:.3f}).</p>
      <p style="font-size:23px">It falls month by month, and so does the share of fails: something that changed over time moved both.</p>
      <p style="font-size:23px">A screened signal is a lead for an engineer to investigate, not a cause. SECOM does not even say what it measures.</p>
    </div>
  </div>""")

slide("""  <h2>Integrity and security of connected tools</h2>
  <div class="cols">
    <div class="card"><h3>NIST IR 8356 (2025)</h3><p style="font-size:20px">Risks: many low-cost sensors that may give untrustworthy data or have vulnerabilities; all data about an object in one place; displays that can be manipulated; remote control while the display hides the change. Measures include hashes and error detection to verify data, encryption, access governance and a zero-trust approach.</p></div>
    <div class="card"><h3>In the fab</h3><p style="font-size:20px">SEMI E187 (2022) sets a minimum cybersecurity baseline for fab equipment: operating system support, network security, endpoint protection, security monitoring. E188 covers malware-free equipment integration. SEMI cites a 2018 ransomware incident that paused a major foundry's production.</p></div>
  </div>
  <p class="small" style="margin-top:14px">Data ownership: equipment data are produced on a supplier's tool inside a fab. Who may read and use them is set by contracts and access policies, not by the data themselves.</p>""")

slide("""  <h2>Summary</h2>
  <ul>
    <li>A fab's records come from separate systems: traces, metrology, MES history. A twin must join them by run and wafer IDs.</li>
    <li>Keep every run with a left join; mark results as on time, late or missing.</li>
    <li>Split by time first; screen, rank and fit on the training records only, ideally inside a pipeline.</li>
    <li>With rare fails, accuracy misleads: report results for each class and compare with a baseline.</li>
    <li>A screened association is a lead, not a cause. Every connection is also a security risk.</li>
  </ul>
  <div class="eq" style="font-size:25px;margin-top:16px"><strong>Next:</strong> the Module 2 lab. You screen SECOM, see leakage happen, and join your own furnace records.</div>""")

slide("""  <h2>Credits and license</h2>
  <p>These slides were written for SemiAcademy's <em>Digital Twins for Semiconductor Manufacturing</em> course, developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). License: <strong>CC BY 4.0</strong>, https://creativecommons.org/licenses/by/4.0/.</p>
  <p style="font-size:20px">Data: SECOM, M. McCann and A. Johnston (2008), UCI Machine Learning Repository, https://doi.org/10.24432/C54305, CC BY 4.0. Sources: NIST IR 8356 (2025), a U.S. government work; SEMI Standards Watch (2022) on SEMI communication standards and on SEMI E187 and E188; Cimetrix overview of SEMI EDA (Interface A); OPC Foundation; scikit-learn, &ldquo;Common pitfalls and recommended practices&rdquo;. Standards are described, not reproduced.</p>
  <p class="small">The furnace records are simulated with the course's model (furnace.py 1.2.0, example key 2026). Source: github.com/CSU-Physics/digital-twins-notebooks (media/m2/m2-slides.html).</p>""")

# ------------------------------------------------------------------ assemble
head_html = (T / "deck_head.html").read_text()
head_html = head_html.replace("<title>Module 1 Slides: From Bits to Qubits</title>", "<title>Module 2 Slides: The Fab as a Data System</title>")
assert "The Fab as a Data System" in head_html
extra_css = ("  code { font-family: Consolas, \"Courier New\", monospace; font-size:0.92em; }\n</style>")
head_html = head_html.replace("</style>", extra_css, 1)
bar = (T / "deck_bar.html").read_text().replace("1 / 19", f"1 / {len(slides)}")
ctrl = (T / "deck_ctrl.js").read_text()
html = head_html + "\n" + "\n".join(slides) + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
for bad in ("—", "${", "`"):
    assert bad not in html, bad
(OUT / "m2-slides.html").write_text(html)
print("wrote", OUT / "m2-slides.html", len(slides), "slides,", len(html), "bytes")
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in A.items()})
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in B.items()}, "inner rows", INNER_N, "missing bins", MISS_COUNTS)
print("shuffles", [(round(a, 3), round(b, 3)) for a, b in SHUF], "s059", [(p, round(m, 2), round(f, 3)) for p, m, f in S059])
