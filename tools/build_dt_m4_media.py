"""Build the Module 4 slide deck (media/m4/m4-slides.html) for Digital Twins for Semiconductor Manufacturing.

Every number and point on the slides is computed here from the course's furnace model (content/m4/furnace.py), the
lab's helpers (content/m4/vm_tools.py) and the reference functions of the Module 4 check, so the slides agree with the
lab notebook. Run from anywhere: python3 tools/build_dt_m4_media.py
"""
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import legendre as LEG
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m4"), str(ROOT / "checks")]
from furnace import VM_SHIFT_MIN, textbook_rates, vm_runs  # noqa: E402
from vm_tools import ALPHAS, FEATURES, block_folds, cv_rmse, physics_prediction, scores, time_split  # noqa: E402
from dt_m4_check import reference_baseline, reference_hybrid, reference_outside, reference_ridge  # noqa: E402

T = ROOT / "tools"
OUT = ROOT / "media" / "m4"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, INK, MUTED, LINE, TINT, OK, BAD, BLUE = "#99004C", "#24313D", "#6A626B", "#E4DCE0", "#F8F4F6", "#2E7D4F", "#B3261E", "#2F5E8E"

# ------------------------------------------------------------------ numbers used on the slides (example key 2026)
RUNS = vm_runs(2026)
MEAS, SHIFT = RUNS["measured"], RUNS["shifted"]
TRAIN, TEST = time_split(MEAS)
CV, GROUPS = block_folds(TRAIN)
XTR, YTR, XTE, YTE = TRAIN[FEATURES], TRAIN["thickness_nm"], TEST[FEATURES], TEST["thickness_nm"]
F5 = ["ox_time_min", "T_center_C", "h2o_ppm", "o2_slm", "slot"]
def _f5(t):
    x = t[F5].copy()
    x["T_center_C"] = x["T_center_C"] - 1000.0          # degrees above 1000 C, so the intercept is a thickness
    return x


LR5 = LinearRegression().fit(_f5(TRAIN), YTR)
C5 = dict(zip(F5, LR5.coef_))
W = TEST.iloc[[12]]
W_PRED = float(LR5.predict(_f5(W))[0])
C5R = {k: round(float(v), 4) for k, v in C5.items()}
I5R = round(float(LR5.intercept_), 2)
W_HAND = I5R + sum(C5R[k] * float(_f5(W)[k].iloc[0]) for k in F5)
OLS = dict(zip(FEATURES, LinearRegression().fit(XTR, YTR).coef_))
R10 = reference_ridge(10.0).fit(XTR, YTR)
RID10 = dict(zip(FEATURES, R10[-1].coef_ / R10[0].scale_))
ZCORR = TRAIN[["T_top_C", "T_center_C", "T_bottom_C"]].corr().to_numpy()
BASE = np.asarray(reference_baseline(MEAS, 3), dtype=float)[MEAS["run_index"].to_numpy() > 40]
CV_R = {a: cv_rmse(reference_ridge(a), XTR, YTR, CV, GROUPS) for a in ALPHAS}
A_R = min(CV_R, key=CV_R.get)
RIDGE = reference_ridge(A_R).fit(XTR, YTR)
PH_TR, PH_TE, PH_SH = physics_prediction(TRAIN), physics_prediction(TEST), physics_prediction(SHIFT)
ERR_TR = YTR.to_numpy() - PH_TR
CV_H = {a: cv_rmse(reference_ridge(a), XTR, ERR_TR, CV, GROUPS) for a in ALPHAS}
A_H = min(CV_H, key=CV_H.get)
HY_TE = reference_hybrid(XTR, YTR, PH_TR, XTE, PH_TE, A_H)
HY_SH = reference_hybrid(XTR, YTR, PH_TR, SHIFT[FEATURES], PH_SH, A_H)
LAST3 = MEAS[MEAS["run_index"] > 57]["thickness_nm"].mean()
S_TE = {"baseline": scores(YTE, BASE), "physics alone": scores(YTE, PH_TE), "data-driven": scores(YTE, RIDGE.predict(XTE)), "hybrid": scores(YTE, HY_TE)}
S_SH = {"baseline": scores(SHIFT["thickness_nm"], np.full(len(SHIFT), LAST3)), "physics alone": scores(SHIFT["thickness_nm"], PH_SH),
        "data-driven": scores(SHIFT["thickness_nm"], RIDGE.predict(SHIFT[FEATURES])), "hybrid": scores(SHIFT["thickness_nm"], HY_SH)}
FLEX = {}
for a in (1e-6, 10.0):
    m = make_pipeline(PolynomialFeatures(3), StandardScaler(), Ridge(alpha=a))
    FLEX[a] = (cv_rmse(m, XTR, YTR, KFold(5, shuffle=True, random_state=0)), cv_rmse(m, XTR, YTR, CV, GROUPS),
               scores(YTE, m.fit(XTR, YTR).predict(XTE))["rmse"])
N_FLAG_SH = int(reference_outside(XTR, SHIFT[FEATURES]).sum())
LO_T, HI_T = XTR["ox_time_min"].min(), XTR["ox_time_min"].max()
LIM_T = (LO_T - 0.1 * (HI_T - LO_T), HI_T + 0.1 * (HI_T - LO_T))
assert f"{S_TE['baseline']['rmse']:.2f}" == "4.31" and f"{S_SH['hybrid']['rmse']:.2f}" == "0.96"
assert S_SH["hybrid"]["rmse"] < S_SH["data-driven"]["rmse"] / 2 and FLEX[1e-6][0] < FLEX[1e-6][1] / 2

# overfitting toy: a Deal-Grove-shaped curve with noise, polynomial fits of degree 1 to 9 (seed 11)
B_TB, BA_TB = textbook_rates(1000)
A_TB = B_TB / BA_TB
TRUE = lambda t: A_TB / 2 * (np.sqrt(1 + 4 * B_TB * t / A_TB ** 2) - 1) + 4.0
_rng = np.random.default_rng(11)
TOY_T = np.sort(_rng.uniform(20, 200, 12)); TOY_Y = TRUE(TOY_T) + _rng.normal(0, 1.5, 12)
TOY_TV = _rng.uniform(20, 200, 200); TOY_YV = TRUE(TOY_TV) + _rng.normal(0, 1.5, 200)
SC = lambda t: (np.asarray(t) - 110.0) / 90.0
TOY_FIT = {d: LEG.legfit(SC(TOY_T), TOY_Y, d) for d in range(1, 10)}
TOY_ERR = {d: (float(np.sqrt(np.mean((LEG.legval(SC(TOY_T), c) - TOY_Y) ** 2))), float(np.sqrt(np.mean((LEG.legval(SC(TOY_TV), c) - TOY_YV) ** 2))))
           for d, c in TOY_FIT.items()}
BEST_D = min(TOY_ERR, key=lambda d: TOY_ERR[d][1])
assert BEST_D == 2

Q_RES = np.array([0.4, -0.8, 0.2, 0.6])          # a pause-and-predict RMSE
Q_RMSE = math.sqrt(float(np.mean(Q_RES ** 2)))


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
        elif kind == "ring":
            s.append(f'<circle cx="{x + 15}" cy="{yy - 6}" r="6" fill="#fff" stroke="{color}" stroke-width="2.5"/>')
        elif kind == "sq":
            s.append(f'<rect x="{x + 9}" y="{yy - 12}" width="12" height="12" fill="{color}"/>')
        s.append(txt(x + 40, yy, label, size, INK, "start"))


# ------------------------------------------------------------------ figures
def boat_fig(w=640, h_=210, label=True):
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="A boat of 25 wafers: five are measured, the others get a virtual measurement">']
    s.append(f'<rect x="10" y="40" width="{w - 20}" height="110" rx="10" fill="{TINT}" stroke="{LINE}" stroke-width="2"/>')
    n = 25
    for i in range(n):
        x = 30 + i * (w - 60) / (n - 1)
        measured = i in (1, 7, 13, 18, 24)
        if measured:
            s.append(f'<rect x="{x - 5:.1f}" y="55" width="10" height="80" rx="3" fill="{MAROON}"/>')
        else:
            s.append(f'<rect x="{x - 5:.1f}" y="55" width="10" height="80" rx="3" fill="#fff" stroke="{MUTED}" stroke-width="2" stroke-dasharray="4 3"/>')
    if label:
        s.append(f'<rect x="30" y="172" width="14" height="14" fill="{MAROON}"/>')
        s.append(txt(52, 185, "measured by the metrology tool", 18, INK, "start"))
        s.append(f'<rect x="{w / 2 + 30:.0f}" y="172" width="14" height="14" fill="#fff" stroke="{MUTED}" stroke-width="2" stroke-dasharray="4 3"/>')
        s.append(txt(w / 2 + 52, 185, "virtual measurement", 18, INK, "start"))
        s.append(txt(w / 2, 26, "one furnace run: the boat, top to bottom", 18, MUTED))
    s.append("</svg>")
    return "".join(s)


def flow_fig():
    w, h_ = 1130, 300
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Trace data and recipe settings feed a model that predicts every wafer; sampled metrology trains, monitors and validates the model">',
         arrow_def("v1", MAROON), arrow_def("v2", INK)]
    boxes = [(10, 60, 300, "furnace trace summaries", "temperatures, moisture, flow"), (430, 60, 290, "virtual-metrology model", "trained on measured wafers"),
             (840, 60, 280, "predicted thickness", "for every wafer, at once")]
    for x, y, wd, a, b in boxes:
        s.append(f'<rect x="{x}" y="{y}" width="{wd}" height="100" rx="10" fill="#fff" stroke="{MAROON}" stroke-width="2.5"/>')
        s.append(txt(x + wd / 2, y + 42, a, 21, INK, weight=600))
        s.append(txt(x + wd / 2, y + 72, b, 18, MUTED))
    s.append(f'<line x1="312" y1="110" x2="424" y2="110" stroke="{MAROON}" stroke-width="4" marker-end="url(#v1)"/>')
    s.append(f'<line x1="722" y1="110" x2="834" y2="110" stroke="{MAROON}" stroke-width="4" marker-end="url(#v1)"/>')
    s.append(f'<rect x="310" y="215" width="530" height="66" rx="10" fill="{TINT}" stroke="{LINE}" stroke-width="2"/>')
    s.append(txt(575, 244, "sampled metrology: a few wafers per run, some runs", 19, INK, weight=600))
    s.append(txt(575, 270, "train · monitor · validate · catch flagged wafers", 18, MUTED))
    s.append(f'<line x1="575" y1="213" x2="575" y2="166" stroke="{INK}" stroke-width="3" marker-end="url(#v2)"/>')
    s.append("</svg>")
    return "".join(s)


def scatter_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    X_ = lambda v: x0 + (v - 78) / 24 * (x1 - x0)
    Y_ = lambda v: y0 - (v - 54) / 18 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Measured thickness of 300 wafers against oxidation time: about half a nanometre more per minute, with a spread of one to two nanometres at each time">']
    axes(s, X_, Y_, x0, x1, y0, y1, [80, 85, 90, 95, 100], [56, 60, 64, 68, 72], "oxidation time (min)", "measured thickness (nm)")
    for a, b in zip(MEAS["ox_time_min"], MEAS["thickness_nm"]):
        s.append(f'<circle cx="{X_(a):.1f}" cy="{Y_(b):.1f}" r="3.5" fill="{MAROON}" fill-opacity="0.55"/>')
    s.append(txt(x0 + 12, y1 + 22, "60 runs × 5 wafers (simulated, key 2026)", 18, MUTED, "start"))
    s.append("</svg>")
    return "".join(s)


def overfit_fig():
    w, h_ = 1130, 380
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Polynomial fits of degree 1, 2 and 9 to 12 noisy points, and the training and test errors for degrees 1 to 9: training error falls, test error is lowest at degree 2 and rises steeply">']
    # left: data and fits
    x0, x1, y0, y1 = 86, 530, h_ - 70, 20
    X_ = lambda v: x0 + (v - 20) / 180 * (x1 - x0)
    Y_ = lambda v: y0 - (v - 10) / 100 * (y0 - y1)
    axes(s, X_, Y_, x0, x1, y0, y1, [20, 65, 110, 155, 200], [10, 30, 50, 70, 90, 110], "oxidation time (min)", "thickness (nm)")
    tt = np.linspace(20, 200, 300)
    for d, color, dash in ((1, MUTED, "8 6"), (2, OK, ""), (9, BAD, "")):
        yy = np.clip(LEG.legval(SC(tt), TOY_FIT[d]), 10, 110)
        s.append(poly(tt, yy, X_, Y_, color, 3.5, dash))
    for a, b in zip(TOY_T, TOY_Y):
        s.append(f'<circle cx="{X_(a):.1f}" cy="{Y_(b):.1f}" r="6" fill="{INK}"/>')
    legend(s, x0 + 230, y0 - 88, [("dash", MUTED, "degree 1"), ("line", OK, "degree 2"), ("line", BAD, "degree 9"), ("dot", INK, "12 training points")], 17)
    # right: errors against degree
    u0, u1 = 660, w - 20
    U_ = lambda dg: u0 + (dg - 1) / 8 * (u1 - u0)
    V_ = lambda v: y0 - (math.log10(v) + 0.7) / 2.0 * (y0 - y1)
    for v in (0.2, 0.5, 1, 2, 5, 10, 20):
        s.append(f'<line x1="{u0}" y1="{V_(v):.1f}" x2="{u1}" y2="{V_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(u0 - 10, V_(v) + 7, f"{v:g}", 18, MUTED, "end"))
    for dg in range(1, 10):
        s.append(txt(U_(dg), y0 + 28, f"{dg}", 18, MUTED))
    s.append(f'<line x1="{u0}" y1="{y0}" x2="{u1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt((u0 + u1) / 2, y0 + 56, "polynomial degree (model complexity)", 20, MUTED))
    s.append(txt(u0 - 52, (y0 + y1) / 2, "RMSE (nm, log scale)", 20, MUTED, extra=f' transform="rotate(-90 {u0 - 52} {(y0 + y1) / 2:.1f})"'))
    ds = list(range(1, 10))
    s.append(poly(ds, [min(TOY_ERR[d][0], 20) for d in ds], U_, V_, INK, 3.5))
    s.append(poly(ds, [min(TOY_ERR[d][1], 20) for d in ds], U_, V_, MAROON, 3.5))
    for d in ds:
        s.append(f'<circle cx="{U_(d):.1f}" cy="{V_(min(TOY_ERR[d][0], 20)):.1f}" r="5" fill="{INK}"/>')
        s.append(f'<circle cx="{U_(d):.1f}" cy="{V_(min(TOY_ERR[d][1], 20)):.1f}" r="5" fill="{MAROON}"/>')
    legend(s, u0 + 20, y1 + 24, [("line", INK, "training error"), ("line", MAROON, "test error (new points)")], 17)
    s.append("</svg>")
    return "".join(s)


def split_fig():
    w, h_ = 1130, 250
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="67 runs in time order: runs 1 to 40 train and tune in five blocks of eight runs, runs 41 to 60 are the test set, runs 61 to 66 are the new recipe, run 67 is not measured">',
         arrow_def("s1", INK)]
    x0, x1 = 20, w - 20
    X_ = lambda r: x0 + (r - 1) / 67 * (x1 - x0)
    cols = {0: "#EAD0DD", 1: "#F3E2EA"}
    for b in range(5):
        a, c = 1 + 8 * b, 9 + 8 * b
        s.append(f'<rect x="{X_(a):.1f}" y="50" width="{X_(c) - X_(a) - 2:.1f}" height="70" fill="{cols[b % 2]}" stroke="{MAROON}" stroke-width="1.5"/>')
        s.append(txt((X_(a) + X_(c)) / 2, 92, f"block {b + 1}", 17, INK))
    s.append(f'<rect x="{X_(41):.1f}" y="50" width="{X_(61) - X_(41) - 2:.1f}" height="70" fill="#E3EBF3" stroke="{BLUE}" stroke-width="2"/>')
    s.append(txt((X_(41) + X_(61)) / 2, 92, "test: runs 41 to 60", 18, INK, weight=600))
    s.append(f'<rect x="{X_(61):.1f}" y="50" width="{X_(67) - X_(61) - 2:.1f}" height="70" fill="#FBE9E7" stroke="{BAD}" stroke-width="2"/>')
    s.append(txt((X_(61) + X_(67)) / 2, 92, "140 min", 16, BAD, weight=600))
    s.append(f'<rect x="{X_(67):.1f}" y="50" width="{X_(68) - X_(67) - 2:.1f}" height="70" fill="#fff" stroke="{MUTED}" stroke-width="2" stroke-dasharray="4 3"/>')
    s.append(txt((X_(1) + X_(41)) / 2, 36, "training runs 1 to 40: fit, and tune with folds of whole runs", 18, MAROON, weight=600))
    s.append(txt(X_(64), 36, "new recipe", 17, BAD, weight=600))
    s.append(txt(x1, 142, "run 67:", 16, MUTED, "end"))
    s.append(txt(x1, 162, "not measured", 16, MUTED, "end"))
    s.append(f'<line x1="{x0}" y1="200" x2="{x1 - 4}" y2="200" stroke="{INK}" stroke-width="2.5" marker-end="url(#s1)"/>')
    s.append(txt((x0 + x1) / 2, 232, "time: every model is trained on the past and tested on the future", 19, MUTED))
    s.append("</svg>")
    return "".join(s)


def folds_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 80, 30
    Y_ = lambda v: y0 - v / 1.6 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Error of the flexible model: random wafer folds report a small error, run-block folds and the test runs show a much larger one">']
    for v in (0, 0.4, 0.8, 1.2, 1.6):
        s.append(f'<line x1="{x0}" y1="{Y_(v):.1f}" x2="{x1}" y2="{Y_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y_(v) + 7, f"{v:.1f}", 19, MUTED, "end"))
    s.append(txt(22, (y0 + y1) / 2, "RMSE (nm)", 20, MUTED, extra=f' transform="rotate(-90 22 {(y0 + y1) / 2:.1f})"'))
    labels = ["random wafer folds", "run-block folds", "test runs"]
    vals = FLEX[1e-6]
    colors = [MUTED, MAROON, BLUE]
    bw = 110
    for k, (lab, v, c) in enumerate(zip(labels, vals, colors)):
        cx = x0 + 70 + k * 165
        s.append(f'<rect x="{cx:.1f}" y="{Y_(v):.1f}" width="{bw}" height="{y0 - Y_(v):.1f}" fill="{c}"/>')
        s.append(txt(cx + bw / 2, Y_(v) - 10, f"{v:.2f}", 20, INK, weight=600))
        s.append(txt(cx + bw / 2, y0 + 26, lab.split(" ")[0] + " " + lab.split(" ")[1], 18, MUTED))
        if len(lab.split(" ")) > 2:
            s.append(txt(cx + bw / 2, y0 + 48, lab.split(" ")[2], 18, MUTED))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt((x0 + x1) / 2, h_ - 6, "flexible model: degree-3 products, penalty 10⁻⁶", 18, MUTED))
    s.append("</svg>")
    return "".join(s)


def baseline_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    X_ = lambda v: x0 + (v - 40) / 21 * (x1 - x0)
    Y_ = lambda v: y0 - (v - 48) / 24 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Test runs: measured thickness and the baseline, which lags behind and misses the run-to-run changes">']
    axes(s, X_, Y_, x0, x1, y0, y1, [41, 45, 50, 55, 60], [48, 52, 56, 60, 64, 68, 72], "run (time order)", "thickness (nm)")
    per = TEST.groupby("run_index")
    br = pd.Series(BASE, index=TEST.index).groupby(TEST["run_index"]).first()
    s.append(poly(br.index, br.values, X_, Y_, MUTED, 3.5))
    for a, b in zip(TEST["run_index"], YTE):
        s.append(f'<circle cx="{X_(a):.1f}" cy="{Y_(b):.1f}" r="3.8" fill="{MAROON}"/>')
    legend(s, x0 + 20, y0 - 40, [("dot", MAROON, "measured wafers"), ("line", MUTED, "baseline: mean of the 3 previous runs")])
    s.append("</svg>")
    return "".join(s)


def resid_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    X_ = lambda v: x0 + (v - 78) / 24 * (x1 - x0)
    Y_ = lambda v: y0 - v / 12 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Measured minus physics for the training wafers: all several nanometres above zero, changing smoothly">']
    axes(s, X_, Y_, x0, x1, y0, y1, [80, 85, 90, 95, 100], [0, 2, 4, 6, 8, 10, 12], "oxidation time (min)", "measured − physics (nm)")
    for a, b, hh in zip(TRAIN["ox_time_min"], ERR_TR, TRAIN["h2o_ppm"]):
        c = MAROON if hh > 7 else BLUE
        s.append(f'<circle cx="{X_(a):.1f}" cy="{Y_(b):.1f}" r="3.8" fill="{c}" fill-opacity="0.75"/>')
    legend(s, x0 + 20, y0 - 40, [("dot", MAROON, "moisture above 7 ppm"), ("dot", BLUE, "moisture below 7 ppm")], 17)
    s.append("</svg>")
    return "".join(s)


def compare_fig():
    w, h_ = 1130, 330
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Errors of four models on the test runs and on the new recipe, log scale">']
    x0, x1, y0, y1 = 100, w - 20, h_ - 70, 30
    Y_ = lambda v: y0 - (math.log10(v) + 0.5) / 2.0 * (y0 - y1)
    for v in (0.5, 1, 2, 5, 10, 20):
        s.append(f'<line x1="{x0}" y1="{Y_(v):.1f}" x2="{x1}" y2="{Y_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y_(v) + 7, f"{v:g}", 19, MUTED, "end"))
    s.append(txt(26, (y0 + y1) / 2, "RMSE (nm, log)", 20, MUTED, extra=f' transform="rotate(-90 26 {(y0 + y1) / 2:.1f})"'))
    names = ["baseline", "physics alone", "data-driven", "hybrid"]
    gw = (x1 - x0) / 4
    for k, n in enumerate(names):
        cx = x0 + gw * k + gw / 2
        for j, (S, c) in enumerate(((S_TE, BLUE), (S_SH, BAD))):
            v = S[n]["rmse"]
            bx = cx - 70 + j * 74
            s.append(f'<rect x="{bx:.1f}" y="{Y_(v):.1f}" width="66" height="{y0 - Y_(v):.1f}" fill="{c}"/>')
            s.append(txt(bx + 33, Y_(v) - 8, f"{v:.2f}", 18, INK, weight=600))
        s.append(txt(cx, y0 + 28, n, 20, INK, weight=600))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    legend(s, x1 - 330, y1 + 4, [("sq", BLUE, "test runs 41 to 60 (80 to 100 min)"), ("sq", BAD, "new recipe (140 min)")], 17)
    s.append("</svg>")
    return "".join(s)


def shift_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    X_ = lambda v: x0 + (v - 70) / 80 * (x1 - x0)
    Y_ = lambda v: y0 - (v - 50) / 45 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Thickness against oxidation time: training runs from 80 to 100 minutes and the new recipe at 140 minutes; in the zoomed box, the data-driven line overshoots the new runs and the hybrid curve passes through them">']
    s.append(f'<rect x="{X_(LIM_T[0]):.1f}" y="{y1}" width="{X_(LIM_T[1]) - X_(LIM_T[0]):.1f}" height="{y0 - y1}" fill="{TINT}"/>')
    axes(s, X_, Y_, x0, x1, y0, y1, [70, 90, 110, 130, 150], [50, 60, 70, 80, 90], "oxidation time (min)", "thickness (nm)")
    s.append(txt((X_(LIM_T[0]) + X_(LIM_T[1])) / 2, y0 - 12, "range check", 17, MUTED))
    ref = TRAIN.iloc[[0]].copy()
    for c in FEATURES:
        if c not in ("ox_time_min", "slot"):
            ref[c] = SHIFT[c].mean()
    ref["slot"] = 65
    tt = np.arange(70, 151, 1.0)
    rows = pd.concat([ref.assign(ox_time_min=t) for t in tt], ignore_index=True)
    lin = RIDGE.predict(rows[FEATURES])
    hyb = reference_hybrid(XTR, YTR, PH_TR, rows[FEATURES], physics_prediction(rows), A_H)
    s.append(poly(tt, lin, X_, Y_, BLUE, 3.5, "8 6"))
    s.append(poly(tt, hyb, X_, Y_, MAROON, 3.5))
    for a, b in zip(TRAIN["ox_time_min"], YTR):
        s.append(f'<circle cx="{X_(a):.1f}" cy="{Y_(b):.1f}" r="3" fill="{INK}" fill-opacity="0.5"/>')
    for b in SHIFT["thickness_nm"]:
        s.append(f'<rect x="{X_(VM_SHIFT_MIN) - 4:.1f}" y="{Y_(b) - 4:.1f}" width="8" height="8" fill="{BAD}"/>')
    # zoom box around the new recipe
    m = float(SHIFT["thickness_nm"].mean())
    za, zb, zc, zd = 134.0, 146.0, m - 4.0, m + 4.0
    bx0, bx1, by0, by1 = x0 + 14, x0 + 204, y1 + 140, y1 + 4
    ZX = lambda v: bx0 + (v - za) / (zb - za) * (bx1 - bx0)
    ZY = lambda v: by0 - (v - zc) / (zd - zc) * (by0 - by1)
    s.append(f'<rect x="{X_(za):.1f}" y="{Y_(zd):.1f}" width="{X_(zb) - X_(za):.1f}" height="{Y_(zc) - Y_(zd):.1f}" fill="none" stroke="{MUTED}" stroke-width="1.5"/>')
    s.append(f'<line x1="{bx1}" y1="{by1}" x2="{X_(za):.1f}" y2="{Y_(zd):.1f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="3 3"/>')
    s.append(f'<rect x="{bx0}" y="{by1}" width="{bx1 - bx0}" height="{by0 - by1}" fill="#fff" stroke="{MUTED}" stroke-width="1.5"/>')
    zt = tt[(tt >= za) & (tt <= zb)]
    zl, zh = lin[(tt >= za) & (tt <= zb)], hyb[(tt >= za) & (tt <= zb)]
    s.append(f'<clipPath id="zc"><rect x="{bx0}" y="{by1}" width="{bx1 - bx0}" height="{by0 - by1}"/></clipPath><g clip-path="url(#zc)">')
    s.append(poly(zt, zl, ZX, ZY, BLUE, 3, "8 6"))
    s.append(poly(zt, zh, ZX, ZY, MAROON, 3))
    for k, b in enumerate(SHIFT["thickness_nm"]):
        s.append(f'<rect x="{ZX(VM_SHIFT_MIN) - 4 + (k % 5 - 2) * 3:.1f}" y="{ZY(b) - 4:.1f}" width="8" height="8" fill="{BAD}"/>')
    s.append("</g>")
    s.append(txt(bx1 - 6, by0 - 8, f"zoom: {za:.0f} to {zb:.0f} min", 14, MUTED, "end"))
    legend(s, x0 + 300, y0 - 112, [("dot", INK, "training wafers"), ("sq", BAD, "new recipe, measured"), ("dash", BLUE, "data-driven model"), ("line", MAROON, "hybrid model")], 17)
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ slides
FOOT = "Module 4 · Virtual metrology for oxide thickness"
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


slide(f"""  <div class="kicker">Digital Twins for Semiconductor Manufacturing · Module 4</div>
  <h1>Virtual metrology for oxide thickness</h1>
  <p style="font-size:30px;max-width:900px">Predict the wafers nobody measured, validate the model without fooling yourself, and know when not to trust it.</p>
  <div style="margin-top:14px">{boat_fig()}</div>""", "slide title-slide")

slide("""  <h2>By the end of this module you can</h2>
  <ul>
    <li>explain <strong>virtual metrology</strong>, and why sampled metrology continues alongside it;</li>
    <li>fit <strong>linear and ridge regression</strong> in a scikit-learn pipeline, with scaling inside it;</li>
    <li>use <strong>training, tuning and test</strong> data, with cross-validation that keeps whole runs together and a test set in time order;</li>
    <li>compare a model with a <strong>simple baseline</strong>, and build a <strong>hybrid</strong>: the Deal-Grove prediction plus a learned correction;</li>
    <li>score a model on a <strong>shifted condition</strong>, and flag predictions outside the training range.</li>
  </ul>
  <p class="small" style="margin-top:16px">This is course outcome 4. The lab builds the three models for your own simulated furnace.</p>""")

slide(f"""  <h2>What virtual metrology is</h2>
  <div style="margin-top:4px">{flow_fig()}</div>
  <p class="small" style="margin-top:14px">Virtual metrology predicts a wafer's measurement from the tool's sensor data and from the actual measurements of sampled wafers (Kang et al., 2009). The sampled metrology never stops: it is how the model is trained, checked and corrected.</p>""")

slide(f"""  <h2>What the twin knows before anyone measures</h2>
  <table style="font-size:21px;width:100%">
    <tr><th>Feature</th><th>Source</th><th>Example wafer (run 43, slot 65)</th></tr>
    <tr><td>oxidation time</td><td>recipe, set by the run-to-run controller</td><td>{W['ox_time_min'].iloc[0]:.0f} min</td></tr>
    <tr><td>zone temperatures: top, center, bottom</td><td>thermocouple traces, mean over the oxidation step</td><td>{W['T_top_C'].iloc[0]:.2f}, {W['T_center_C'].iloc[0]:.2f}, {W['T_bottom_C'].iloc[0]:.2f} °C</td></tr>
    <tr><td>moisture in the oxygen</td><td>gas analyzer</td><td>{W['h2o_ppm'].iloc[0]:.1f} ppm</td></tr>
    <tr><td>oxygen flow</td><td>mass-flow controller</td><td>{W['o2_slm'].iloc[0]:.2f} slm</td></tr>
    <tr><td>slot</td><td>wafer position in the 125-slot boat</td><td>{int(W['slot'].iloc[0])}</td></tr>
  </table>
  <p class="small" style="margin-top:12px">The target: the oxide thickness, measured on five sampled wafers per run (measured: {W['thickness_nm'].iloc[0]:.2f} nm). Simulated data, example key 2026.</p>""")

slide(f"""  <h2>Thickness and its main driver</h2>
  {two(scatter_fig(), ["Most of the spread comes from the oxidation time: about half a nanometre per minute.", "At one oxidation time, the wafers still spread over a nanometre or two: temperature, moisture and position.", "No single feature explains everything: the twin needs a model.", "Wafers of one run share their run's conditions."])}""")

slide(f"""  <h2>A linear model with five features</h2>
  <div class="eq" style="font-size:25px">thickness ≈ {I5R:.2f} + {C5R['ox_time_min']:.4f}·t + {C5R['T_center_C']:.4f}·(T_center − 1000) + {C5R['h2o_ppm']:.4f}·H₂O + {C5R['o2_slm']:.4f}·O₂ + {C5R['slot']:.4f}·slot</div>
  <ul style="font-size:23px">
    <li>Least squares on the 200 training wafers. Units: nm per minute, per °C, per ppm, per slm, per slot.</li>
    <li>One more minute: +{C5['ox_time_min']:.2f} nm. One degree hotter: +{C5['T_center_C']:.2f} nm. One ppm more moisture: +{C5['h2o_ppm']:.2f} nm.</li>
    <li>The oxygen flow barely matters: in the simulator it has no effect at all.</li>
    <li>The intercept is the thickness the model gives at zero for every feature: here only a reference point, far outside the data.</li>
  </ul>""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>Predict one wafer</h2>
  <p>Run 43, slot 65: t = {W['ox_time_min'].iloc[0]:.0f} min, T_center = {W['T_center_C'].iloc[0]:.2f} °C, H₂O = {W['h2o_ppm'].iloc[0]:.1f} ppm, O₂ = {W['o2_slm'].iloc[0]:.2f} slm, slot 65.</p>
  <p><strong>Predict:</strong> the thickness, with the coefficients of the previous slide.</p>
{reveal(f"{I5R:.2f} + {C5R['ox_time_min']:.4f}×{W['ox_time_min'].iloc[0]:.0f} + {C5R['T_center_C']:.4f}×({W['T_center_C'].iloc[0] - 1000:.2f}) + {C5R['h2o_ppm']:.4f}×{W['h2o_ppm'].iloc[0]:.1f} + {C5R['o2_slm']:.4f}×{W['o2_slm'].iloc[0]:.2f} + {C5R['slot']:.4f}×65 = {W_HAND:.2f} nm. The wafer measured {W['thickness_nm'].iloc[0]:.2f} nm: the model is off by {W['thickness_nm'].iloc[0] - W_HAND:+.2f} nm, inside its typical error of about half a nanometre.")}""")

slide(f"""  <h2>Correlated features and ridge regression</h2>
  <div class="eq" style="font-size:28px">minimize &nbsp; Σ (yᵢ − ŷᵢ)² + α · Σ w_j²</div>
  <table style="font-size:21px;width:100%">
    <tr><th>Zone temperature coefficient (nm/°C)</th><th>top</th><th>center</th><th>bottom</th><th>sum</th></tr>
    <tr><td>least squares, all 7 features</td><td>{OLS['T_top_C']:.3f}</td><td>{OLS['T_center_C']:.3f}</td><td>{OLS['T_bottom_C']:.3f}</td><td>{OLS['T_top_C'] + OLS['T_center_C'] + OLS['T_bottom_C']:.3f}</td></tr>
    <tr><td>ridge, α = 10 on scaled features</td><td>{RID10['T_top_C']:.3f}</td><td>{RID10['T_center_C']:.3f}</td><td>{RID10['T_bottom_C']:.3f}</td><td>{RID10['T_top_C'] + RID10['T_center_C'] + RID10['T_bottom_C']:.3f}</td></tr>
  </table>
  <p style="font-size:22px;margin-top:14px">The three zones move together (correlation {ZCORR[0, 1]:.2f} to {ZCORR[1, 2]:.2f}), so least squares can split their shared effect into a negative and a large positive coefficient. The penalty pulls them toward each other. Because it compares coefficients, the features must be <strong>scaled first</strong>: StandardScaler, then Ridge, in one pipeline.</p>""")

slide(f"""  <h2>Overfitting: more complex is not better</h2>
  <div>{overfit_fig()}</div>
  <p class="small" style="margin-top:0">Twelve noisy points from a growth curve. The degree-9 polynomial has the smallest training error and a test error {TOY_ERR[9][1] / TOY_ERR[2][1]:.0f} times that of degree 2: it fits the noise. Here degree {BEST_D} predicts new points best.</p>""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>Which error tells you about new wafers?</h2>
  <p>Model A has a training RMSE of {TOY_ERR[9][0]:.2f} nm. Model B has {TOY_ERR[2][0]:.2f} nm. Both were fitted to the same 12 points.</p>
  <p><strong>Predict:</strong> which one will do better on new wafers, and how would you find out?</p>
{reveal(f"You cannot tell from the training error: it only rewards following the points. Score both on data the fit never saw. Here A is the degree-9 polynomial, with a test RMSE of {TOY_ERR[9][1]:.1f} nm, and B the degree-2 one, with {TOY_ERR[2][1]:.2f} nm.")}""")

slide(f"""  <h2>Training, tuning and test data</h2>
  <div>{split_fig()}</div>
  <ul style="font-size:22px">
    <li><strong>Training runs</strong> fit the model. <strong>Tuning</strong> choices, such as α, use cross-validation inside the training runs.</li>
    <li>The <strong>test runs</strong> come later in time and are used once, at the end. Their score is the estimate of how the model will do.</li>
  </ul>""")

slide("""  <h2>Cross-validation, done properly</h2>
  <ul>
    <li>Split the training runs into five folds; fit on four, validate on the fifth; rotate; average the five errors.</li>
    <li><strong>Everything learned from data goes inside the fold:</strong> the scaler too. A pipeline makes that automatic: cross_val_score refits the whole pipeline on each training fold.</li>
    <li><strong>Keep whole runs together.</strong> Each fold here is a block of 8 consecutive runs, so no run is split and neighbouring runs stay together.</li>
    <li>Choose the α with the smallest cross-validated error, then refit on all training runs.</li>
  </ul>
  <p class="small" style="margin-top:12px">scikit-learn: Pipeline, StandardScaler, Ridge, GroupKFold, cross_val_score, GridSearchCV.</p>""")

slide(f"""  <h2>Why the folds hold whole runs</h2>
  {two(folds_fig(), [f"A flexible model (all products of the features up to degree 3) with almost no penalty.", f"Random wafer folds: {FLEX[1e-6][0]:.2f} nm. It has seen four wafers of each validation run, so it can look up the fifth.", f"Run-block folds: {FLEX[1e-6][1]:.2f} nm; the test runs agree: {FLEX[1e-6][2]:.2f} nm.", f"With run blocks, a larger penalty wins: α = 10 gives {FLEX[10.0][1]:.2f} nm."], size=22)}""")

slide(f"""  <h2>First, a baseline</h2>
  {two(baseline_fig(), ["Baseline: the mean thickness of all wafers measured in the 3 runs before this one.", "It uses only finished, measured runs: never the run itself.", f"Test RMSE: {S_TE['baseline']['rmse']:.2f} nm. It follows slow changes, but not each run's own oxidation time.", "A model that cannot beat it is not worth running; one that ties it is a valid finding."], size=22)}""")

slide(f"""  <h2>Physics plus a learned correction</h2>
  {two(resid_fig(), [f"Textbook Deal-Grove at each run's center-zone reading and time: right shape, wrong level. Every wafer is thicker, by {S_TE['physics alone']['bias']:.1f} nm on average.", "The error is smooth and systematic: the thin-oxide regime, moisture, this furnace's constants.", "Hybrid = physics prediction + ridge model of (measured − physics).", "The correction is learned and cross-validated like any other model."], size=22)}""")

slide(f"""  <h2>Test runs and a new recipe</h2>
  <div>{compare_fig()}</div>
  <p class="small" style="margin-top:4px">Inside the training range the data-driven and hybrid models tie ({S_TE['data-driven']['rmse']:.2f} and {S_TE['hybrid']['rmse']:.2f} nm), close to the noise no trace explains. On the 140-minute recipe the hybrid's error grows to {S_SH['hybrid']['rmse']:.2f} nm, the data-driven model's to {S_SH['data-driven']['rmse']:.2f} nm.</p>""")

slide(f"""  <h2>Why the new recipe separates them</h2>
  {two(shift_fig(), ["The data-driven model extends a straight line in time beyond 100 minutes.", "Oxide growth slows as the oxide thickens, so the line overshoots, every wafer in the same direction.", "The hybrid takes the time dependence from Deal-Grove, which holds at 140 minutes; only its small correction is extrapolated.", f"All {N_FLAG_SH} wafers of the new recipe fall outside the range check."], size=22)}""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>Compute an RMSE</h2>
  <p>A model's errors (measured minus predicted) on four wafers are {", ".join(f"{v:+.1f}" for v in Q_RES)} nm.</p>
  <p><strong>Predict:</strong> the RMSE.</p>
{reveal(f"RMSE = √(({' + '.join(f'{v:.1f}²' for v in Q_RES)}) / 4) = √({float(np.mean(Q_RES ** 2)):.3f}) = {Q_RMSE:.3f} nm. Squaring makes the largest error count most, which is why RMSE is quoted with the largest error alongside it.")}""")

slide(f"""  <h2>When not to trust a prediction</h2>
  <ul>
    <li><strong>Range check:</strong> flag a wafer if any feature is outside its training range, widened by 10 % of the range at each end. For oxidation time here: {LIM_T[0]:.0f} to {LIM_T[1]:.0f} minutes.</li>
    <li>A flag is a request to measure, not proof of an error. Industry systems refine the idea: Cheng and co-workers' automatic virtual metrology adds a <strong>similarity index</strong> (distance of a wafer's process data from the training data) and a <strong>reliance index</strong> for each prediction.</li>
    <li>A range check misses problems that keep every feature in range, such as a slowly drifting thermocouple: Module 5.</li>
    <li><strong>Claim only what the data support:</strong> on simulated data, a good score shows the models recover the course's simulator, not a real furnace.</li>
  </ul>""")

slide("""  <h2>Summary</h2>
  <ul>
    <li>Virtual metrology predicts every wafer from trace data; sampled metrology keeps training, checking and correcting it.</li>
    <li>Ridge regression in a pipeline: scale, then penalize; tune α with cross-validation inside the training runs.</li>
    <li>Keep whole runs together and test in time order, or the scores will flatter the model.</li>
    <li>Beat a simple baseline first. A hybrid of physics and a learned correction ties the data-driven model inside the range and holds up better outside it.</li>
    <li>Flag wafers outside the training range and send them to metrology.</li>
  </ul>
  <div class="eq" style="font-size:25px;margin-top:16px"><strong>Next:</strong> the Module 4 lab. You build the three models, score them on a new recipe and predict a run nobody measured.</div>""")

slide("""  <h2>Credits and license</h2>
  <p>These slides were written for SemiAcademy's <em>Digital Twins for Semiconductor Manufacturing</em> course, developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). License: <strong>CC BY 4.0</strong>, https://creativecommons.org/licenses/by/4.0/.</p>
  <p style="font-size:20px">Sources: P. Kang et al., "A virtual metrology system for semiconductor manufacturing", Expert Systems with Applications 36(10), 12554–12561 (2009); F.-T. Cheng, Y.-T. Chen and Y.-C. Su, US Patent 7,593,912, on the reliance and similarity indexes of automatic virtual metrology; scikit-learn documentation on cross-validation and pipelines. Cited, not reproduced.</p>
  <p class="small">Every data value on these slides is simulated with the course's furnace model (furnace.py 1.4.0, example key 2026). Source: github.com/CSU-Physics/digital-twins-notebooks (media/m4/m4-slides.html).</p>""")

# ------------------------------------------------------------------ assemble
head_html = (T / "deck_head.html").read_text()
head_html = head_html.replace("<title>Module 1 Slides: From Bits to Qubits</title>", "<title>Module 4 Slides: Virtual Metrology for Oxide Thickness</title>")
assert "Virtual Metrology for Oxide Thickness" in head_html
head_html = head_html.replace("</style>", "  code { font-family: Consolas, \"Courier New\", monospace; font-size:0.92em; }\n</style>", 1)
bar = (T / "deck_bar.html").read_text().replace("1 / 19", f"1 / {len(slides)}")
ctrl = (T / "deck_ctrl.js").read_text()
html = head_html + "\n" + "\n".join(slides) + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
for bad in ("—", "${", "`"):
    assert bad not in html, bad
(OUT / "m4-slides.html").write_text(html)
print("wrote", OUT / "m4-slides.html", len(slides), "slides,", len(html), "bytes")
print("test", {k: round(v["rmse"], 3) for k, v in S_TE.items()}, "shift", {k: round(v["rmse"], 3) for k, v in S_SH.items()})
print("flex", {k: tuple(round(x, 3) for x in v) for k, v in FLEX.items()}, "toy", {d: tuple(round(x, 2) for x in e) for d, e in TOY_ERR.items()})
print("W pred", round(W_PRED, 2), "q rmse", round(Q_RMSE, 3), "limits", LIM_T, "alphas", A_R, A_H)
