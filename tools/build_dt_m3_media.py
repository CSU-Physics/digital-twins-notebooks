"""Build the Module 3 slide deck (media/m3/m3-slides.html) for Digital Twins for Semiconductor Manufacturing.

Every number and curve on the slides is computed here from the course's furnace model (content/m3/furnace.py), the
lab's fitting helpers (content/m3/oxide_tools.py) and the reference functions of the Module 3 check, so the slides
agree with the lab notebook. Run from anywhere: python3 tools/build_dt_m3_media.py
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m3"), str(ROOT / "checks")]
from furnace import OX_NEW_MIN, OX_SIGMA_NM, OX_XI_NM, THIN_XI_NM, oxidation_runs, textbook_rates  # noqa: E402
from oxide_tools import fit_deal_grove, linear_form_fit, predict  # noqa: E402
from dt_m3_check import reference_tau, reference_thickness  # noqa: E402

T = ROOT / "tools"
OUT = ROOT / "media" / "m3"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, INK, MUTED, LINE, TINT, OK, BAD = "#99004C", "#24313D", "#6A626B", "#E4DCE0", "#F8F4F6", "#2E7D4F", "#B3261E"

# ------------------------------------------------------------------ numbers used on the slides
B_TB, BA_TB = textbook_rates(1000)
A_TB = B_TB / BA_TB
TAU25 = reference_tau(25.0, B_TB, BA_TB)
X120 = float(reference_thickness(120.0, B_TB, BA_TB, 25.0))
RUNS = oxidation_runs(2026)
CAL, NEW, THIN = RUNS["calibration"], RUNS["new"], RUNS["thin"]
FIT = fit_deal_grove(CAL["time_min"], CAL["thickness_nm"], OX_SIGMA_NM, OX_XI_NM)
B3, BA3 = linear_form_fit(CAL["time_min"].iloc[[0, 5, 11]], CAL["thickness_nm"].iloc[[0, 5, 11]], OX_XI_NM)
P270 = predict(FIT, OX_NEW_MIN)
P1440 = predict(FIT, 1440)
FIT_THIN = dict(FIT, xi=THIN_XI_NM)
PT = predict(FIT_THIN, THIN["time_min"])
Z_THIN = (THIN["thickness_nm"].to_numpy() - PT["x"]) / PT["se_pred"]
RATIO_1100 = textbook_rates(1100)[0] / B_TB
X2H = {Tc: float(reference_thickness(120.0, *textbook_rates(Tc), 25.0)) for Tc in (900, 1000, 1100)}
X2H_WET = {Tc: float(reference_thickness(120.0, *textbook_rates(Tc, "wet"), 25.0)) for Tc in (900, 1000, 1100)}
assert f"{TAU25:.1f}" == "37.0" and f"{P270['x'][0]:.2f}" == "143.41" and f"{1.96 * P270['se_pred'][0]:.2f}" == "1.03"
assert all(z > 5 for z in Z_THIN)
SE_Q = (0.30, 0.50)
SE_Q_PRED = math.sqrt(SE_Q[0] ** 2 + SE_Q[1] ** 2)


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


def band(xs, lo, hi, X_, Y_, color, opacity):
    pts = [f"{X_(a):.1f},{Y_(b):.1f}" for a, b in zip(xs, hi)] + [f"{X_(a):.1f},{Y_(b):.1f}" for a, b in zip(xs[::-1], lo[::-1])]
    return f'<polygon points="{" ".join(pts)}" fill="{color}" fill-opacity="{opacity}" stroke="none"/>'


def legend(s, x, y, items):
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
        else:
            s.append(f'<rect x="{x}" y="{yy - 14}" width="30" height="14" fill="{color}" fill-opacity="{kind}"/>')
        s.append(txt(x + 40, yy, label, 18, INK, "start"))


# ------------------------------------------------------------------ figures
def title_fig():
    w, h_ = 640, 230
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" aria-hidden="true">', arrow_def("t1", MAROON)]
    s.append(f'<rect x="10" y="20" width="620" height="50" fill="#FFFFFF" stroke="{LINE}" stroke-width="2"/>')
    s.append(txt(320, 52, "O₂ gas", 20, MUTED))
    s.append(f'<rect x="10" y="70" width="620" height="60" fill="#FBF3F7" stroke="{MAROON}" stroke-width="2.5"/>')
    s.append(txt(320, 108, "SiO₂: the oxide grows from the bottom", 20, INK, weight=600))
    s.append(f'<rect x="10" y="130" width="620" height="80" fill="#EEF0F3" stroke="{LINE}" stroke-width="2"/>')
    s.append(txt(320, 178, "Si wafer", 20, MUTED))
    for xx in (40, 95, 545, 600):
        s.append(f'<line x1="{xx}" y1="40" x2="{xx}" y2="124" stroke="{MAROON}" stroke-width="3" marker-end="url(#t1)"/>')
    s.append("</svg>")
    return "".join(s)


def flux_fig():
    w, h_ = 1130, 300
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Three steps in series: oxygen reaches the oxide surface, diffuses through the oxide, reacts at the silicon interface">',
         arrow_def("f1", MAROON)]
    zones = [(10, 300, "gas", "#FFFFFF", LINE), (310, 420, "oxide, thickness x", "#FBF3F7", MAROON), (740, 380, "silicon", "#EEF0F3", LINE)]
    for x, wd, lab, fill, stroke in zones:
        s.append(f'<rect x="{x}" y="40" width="{wd}" height="150" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>')
        s.append(txt(x + wd / 2, 70, lab, 21, INK, weight=600))
    s.append(f'<line x1="120" y1="130" x2="330" y2="130" stroke="{MAROON}" stroke-width="4" marker-end="url(#f1)"/>')
    s.append(f'<line x1="340" y1="130" x2="720" y2="130" stroke="{MAROON}" stroke-width="4" marker-end="url(#f1)"/>')
    s.append(txt(200, 118, "1. reaches the surface", 18, MAROON))
    s.append(txt(525, 118, "2. diffuses through the oxide", 18, MAROON))
    s.append(txt(860, 118, "3. reacts: Si + O₂ → SiO₂", 18, MAROON, "start"))
    s.append(txt(525, 236, "thin oxide: step 3 limits the rate → x grows like (B/A)·t", 20, INK, weight=600))
    s.append(txt(525, 268, "thick oxide: step 2 limits the rate → x grows like √(B·t)", 20, INK, weight=600))
    s.append("</svg>")
    return "".join(s)


def limits_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    X_ = lambda t: x0 + t / 1440 * (x1 - x0)
    Y_ = lambda v: y0 - v / 450 * (y0 - y1)
    t = np.linspace(0, 1440, 200)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Deal-Grove growth of a bare wafer at 1000 C with its linear and parabolic limits">']
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 240, 480, 720, 960, 1200, 1440], [0, 100, 200, 300, 400], "time (min)", "oxide thickness (nm)")
    tl = t[BA_TB * t <= 450]
    s.append(poly(tl, BA_TB * tl, X_, Y_, MUTED, 3, "8 6"))
    s.append(poly(t, np.sqrt(B_TB * t), X_, Y_, INK, 3, "3 5"))
    s.append(poly(t, reference_thickness(t, B_TB, BA_TB, 0.0), X_, Y_, MAROON, 4.5))
    legend(s, x0 + 230, y0 - 80, [("line", MAROON, "Deal-Grove, bare wafer"), ("dash", MUTED, "linear limit (B/A)·t"), ("dash", INK, "parabolic limit √(B·t)")])
    s.append("</svg>")
    return "".join(s)


def tau_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    X_ = lambda t: x0 + (t + 90) / 330 * (x1 - x0)
    Y_ = lambda v: y0 - v / 130 * (y0 - y1)
    t = np.linspace(-TAU25, 240, 200)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="The curve of a bare wafer, shifted: a wafer that starts at 25 nm behaves as if it had already oxidized for tau minutes">']
    axes(s, X_, Y_, x0, x1, y0, y1, [-90, -60, -30, 0, 60, 120, 180, 240], [0, 25, 50, 75, 100, 125], "time of this run (min)", "oxide thickness (nm)")
    s.append(poly(t, reference_thickness(t + TAU25, B_TB, BA_TB, 0.0), X_, Y_, MAROON, 4.5))
    s.append(f'<line x1="{X_(0):.1f}" y1="{y0}" x2="{X_(0):.1f}" y2="{y1}" stroke="{INK}" stroke-width="2"/>')
    s.append(f'<line x1="{X_(-TAU25):.1f}" y1="{Y_(25):.1f}" x2="{X_(0):.1f}" y2="{Y_(25):.1f}" stroke="{INK}" stroke-dasharray="6 5" stroke-width="2"/>')
    s.append(f'<circle cx="{X_(0):.1f}" cy="{Y_(25):.1f}" r="7" fill="{INK}"/>')
    s.append(f'<circle cx="{X_(-TAU25):.1f}" cy="{Y_(0):.1f}" r="7" fill="{MUTED}"/>')
    s.append(txt(X_(0) - 8, Y_(25) - 14, f"τ = {TAU25:.0f} min", 20, INK, "end", 600))
    s.append(txt(X_(0) + 14, Y_(25) + 30, "x_i = 25 nm at t = 0", 19, INK, "start"))
    s.append("</svg>")
    return "".join(s)


def temp_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 96, w - 20, h_ - 70, 20
    temps = np.arange(850, 1201, 10)
    X_ = lambda v: x0 + (v - 850) / 350 * (x1 - x0)
    Y_ = lambda v: y0 - (math.log10(v) - 0) / 3.2 * (y0 - y1)
    dry = [float(reference_thickness(120.0, *textbook_rates(Tc), 0.0)) for Tc in temps]
    wet = [float(reference_thickness(120.0, *textbook_rates(Tc, "wet"), 0.0)) for Tc in temps]
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Oxide grown in 2 hours on a bare wafer from 850 to 1200 C, dry and wet, on a log scale">']
    for v in (1, 10, 100, 1000):
        s.append(f'<line x1="{x0}" y1="{Y_(v):.1f}" x2="{x1}" y2="{Y_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y_(v) + 7, f"{v:g}", 19, MUTED, "end"))
    for v in (900, 1000, 1100, 1200):
        s.append(txt(X_(v), y0 + 28, f"{v}", 19, MUTED))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt((x0 + x1) / 2, y0 + 56, "temperature (°C)", 20, MUTED))
    s.append(txt(22, (y0 + y1) / 2, "oxide after 2 h (nm, log scale)", 20, MUTED, extra=f' transform="rotate(-90 22 {(y0 + y1) / 2:.1f})"'))
    s.append(poly(temps, wet, X_, Y_, INK, 4, "8 6"))
    s.append(poly(temps, dry, X_, Y_, MAROON, 4.5))
    legend(s, x0 + 300, y0 - 50, [("line", MAROON, "dry O₂"), ("dash", INK, "wet (H₂O)")])
    s.append("</svg>")
    return "".join(s)


def cal_fig(with_fit=False, with_textbook=True):
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    X_ = lambda t: x0 + t / 520 * (x1 - x0)
    Y_ = lambda v: y0 - v / 230 * (y0 - y1)
    t = np.linspace(0, 520, 200)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Twelve calibration measurements from 30 to 480 minutes with the model curve">']
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 120, 240, 360, 480], [0, 50, 100, 150, 200], "oxidation time (min)", "oxide thickness (nm)")
    items = []
    if with_textbook:
        s.append(poly(t, reference_thickness(t, B_TB, BA_TB, OX_XI_NM), X_, Y_, MUTED, 3, "8 6"))
        items.append(("dash", MUTED, "textbook constants"))
    if with_fit:
        s.append(poly(t, reference_thickness(t, FIT["B"], FIT["BA"], OX_XI_NM), X_, Y_, MAROON, 3.5))
        items.append(("line", MAROON, "calibrated fit"))
    for a, b in zip(CAL["time_min"], CAL["thickness_nm"]):
        s.append(f'<circle cx="{X_(a):.1f}" cy="{Y_(b):.1f}" r="6" fill="{INK}"/>')
    items.append(("dot", INK, "12 calibration runs (σ = 0.5 nm)"))
    legend(s, x0 + 170, y0 - 28 - 26 * (len(items) - 1), items)
    s.append("</svg>")
    return "".join(s)


def ellipse_fig():
    w, h_ = 600, 390
    x0, x1, y0, y1 = 100, w - 20, h_ - 70, 20
    cov = FIT["cov"]
    vals, vecs = np.linalg.eigh(cov)
    ang = np.linspace(0, 2 * np.pi, 120)
    bx, by = FIT["B"], FIT["BA"]
    sB, sA = 4.5 * FIT["se_B"], 4.5 * FIT["se_BA"]
    X_ = lambda v: x0 + (v - (bx - sB)) / (2 * sB) * (x1 - x0)
    Y_ = lambda v: y0 - (v - (by - sA)) / (2 * sA) * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="The joint uncertainty of B and B/A is a narrow tilted ellipse: the two estimates are strongly negatively correlated">']
    xt = [int(round(bx - 3 * FIT["se_B"])), int(round(bx)), int(round(bx + 3 * FIT["se_B"]))]
    yt = [by - 3 * FIT["se_BA"], by, by + 3 * FIT["se_BA"]]
    axes(s, X_, Y_, x0, x1, y0, y1, xt, yt, "B (nm²/min)", "B/A (nm/min)", "{:.3f}")
    for k, col in ((2.45, MUTED), (1.52, MAROON)):          # sqrt of chi2(2) at 95% and 68%
        e = (vecs @ np.diag(np.sqrt(vals)) @ np.vstack([np.cos(ang), np.sin(ang)])) * k
        s.append(poly(bx + e[0], by + e[1], X_, Y_, col, 3))
    s.append(f'<circle cx="{X_(bx):.1f}" cy="{Y_(by):.1f}" r="6" fill="{INK}"/>')
    legend(s, x0 + 220, y1 + 24, [("line", MAROON, "68% region"), ("line", MUTED, "95% region")])
    s.append(txt(x0 + 220, y1 + 90, f"correlation {FIT['corr']:.2f}", 19, INK, "start", 600))
    s.append("</svg>")
    return "".join(s)


def resid_fig():
    w, h_ = 600, 330
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    X_ = lambda t: x0 + t / 520 * (x1 - x0)
    Y_ = lambda v: y0 - (v + 3) / 6 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Residuals of the fit in units of sigma, all within plus or minus 2">']
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 120, 240, 360, 480], [-3, -2, -1, 0, 1, 2, 3], "oxidation time (min)", "residual / σ")
    s.append(f'<rect x="{x0}" y="{Y_(2):.1f}" width="{x1 - x0}" height="{Y_(-2) - Y_(2):.1f}" fill="{TINT}"/>')
    s.append(f'<line x1="{x0}" y1="{Y_(0):.1f}" x2="{x1}" y2="{Y_(0):.1f}" stroke="{INK}" stroke-width="2"/>')
    for a, r in zip(CAL["time_min"], FIT["residuals"] / OX_SIGMA_NM):
        s.append(f'<line x1="{X_(a):.1f}" y1="{Y_(0):.1f}" x2="{X_(a):.1f}" y2="{Y_(r):.1f}" stroke="{MAROON}" stroke-width="2"/>')
        s.append(f'<circle cx="{X_(a):.1f}" cy="{Y_(r):.1f}" r="6" fill="{MAROON}"/>')
    s.append("</svg>")
    return "".join(s)


def bands_zoom_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 96, w - 20, h_ - 70, 20
    t = np.linspace(255, 285, 120)
    pb = predict(FIT, t)
    lo, hi = 138, 150
    X_ = lambda v: x0 + (v - 255) / 30 * (x1 - x0)
    Y_ = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Near 270 minutes: a narrow confidence band inside a wider prediction band; the three measured wafers of the new run fall inside the prediction band">']
    axes(s, X_, Y_, x0, x1, y0, y1, [255, 260, 265, 270, 275, 280, 285], [138, 140, 142, 144, 146, 148, 150], "oxidation time (min)", "oxide thickness (nm)")
    s.append(band(t, pb["pred_lo"], pb["pred_hi"], X_, Y_, MAROON, 0.15))
    s.append(band(t, pb["conf_lo"], pb["conf_hi"], X_, Y_, MAROON, 0.4))
    s.append(poly(t, pb["x"], X_, Y_, MAROON, 2.5))
    for k, v in enumerate(NEW["thickness_nm"]):
        s.append(f'<rect x="{X_(OX_NEW_MIN) - 6 + (k - 1) * 14:.1f}" y="{Y_(v) - 6:.1f}" width="12" height="12" fill="{INK}"/>')
    legend(s, x0 + 18, y1 + 24, [("0.4", MAROON, "95% confidence band (the mean)"), ("0.15", MAROON, "95% prediction band (one wafer)"),
                                 ("sq", INK, "new run, 3 wafers")])
    s.append("</svg>")
    return "".join(s)


def extrap_fig():
    w, h_ = 1130, 380
    x0, x1, y0, y1 = 100, w - 30, h_ - 70, 20
    t = np.linspace(0, 1440, 300)
    pb = predict(FIT, t)
    X_ = lambda v: x0 + v / 1440 * (x1 - x0)
    Y_ = lambda v: y0 - v / 420 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Bands from 0 to 24 hours: narrow inside the calibrated range of 30 to 480 minutes, widening beyond it">']
    s.append(f'<rect x="{X_(30):.1f}" y="{y1}" width="{X_(480) - X_(30):.1f}" height="{y0 - y1}" fill="{TINT}"/>')
    s.append(txt((X_(30) + X_(480)) / 2, y1 + 22, "calibrated range", 19, MUTED))
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 240, 480, 720, 960, 1200, 1440], [0, 100, 200, 300, 400], "oxidation time (min)", "oxide thickness (nm)")
    # draw the bands 10 times wider, so that they can be seen at this scale
    k = 10.0
    s.append(band(t, pb["x"] - k * 1.96 * pb["se_mean"], pb["x"] + k * 1.96 * pb["se_mean"], X_, Y_, MAROON, 0.3))
    s.append(poly(t, pb["x"], X_, Y_, MAROON, 3))
    for a, b in zip(CAL["time_min"], CAL["thickness_nm"]):
        s.append(f'<circle cx="{X_(a):.1f}" cy="{Y_(b):.1f}" r="5" fill="{INK}"/>')
    s.append(txt(X_(1440) - 4, Y_(P1440["x"][0]) + 70, f"24 h: {P1440['x'][0]:.0f} ± {1.96 * P1440['se_mean'][0]:.1f} nm", 20, INK, "end", 600))
    s.append(txt(X_(270), Y_(P270["x"][0]) - 34, f"270 min: ± {1.96 * P270['se_mean'][0]:.2f} nm", 20, INK, "middle", 600))
    s.append(txt(x1, y0 - 16, "95% confidence band, drawn 10 times wider", 18, MUTED, "end"))
    s.append("</svg>")
    return "".join(s)


def thin_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    t = np.linspace(0, 50, 120)
    pb = predict(FIT_THIN, t)
    X_ = lambda v: x0 + v / 50 * (x1 - x0)
    Y_ = lambda v: y0 - v / 40 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Thin-oxide runs on bare wafers: every measurement lies several nanometres above the prediction band">']
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40], "oxidation time (min)", "oxide thickness (nm)")
    s.append(band(t, pb["pred_lo"], pb["pred_hi"], X_, Y_, MAROON, 0.2))
    s.append(poly(t, pb["x"], X_, Y_, MAROON, 3))
    for a, b, z in zip(THIN["time_min"], THIN["thickness_nm"], Z_THIN):
        s.append(f'<rect x="{X_(a) - 7:.1f}" y="{Y_(b) - 7:.1f}" width="14" height="14" fill="{BAD}"/>')
        s.append(txt(X_(a) - 12, Y_(b) - 4, f"z = {z:.1f}", 17, BAD, "end", 600))
    legend(s, x0 + 235, y0 - 62, [("line", MAROON, "Deal-Grove, x_i = 1.5 nm"), ("0.2", MAROON, "95% prediction band"), ("sq", BAD, "measured, bare wafers")])
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ slides
FOOT = "Module 3 · Physics model: thermal oxidation"
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


slide(f"""  <div class="kicker">Digital Twins for Semiconductor Manufacturing · Module 3</div>
  <h1>Physics model: thermal oxidation</h1>
  <p style="font-size:30px;max-width:860px">Calibrate the Deal-Grove model on the furnace, give its uncertainty, and say where it applies.</p>
  <div style="margin-top:18px">{title_fig()}</div>""", "slide title-slide")

slide("""  <h2>By the end of this module you can</h2>
  <ul>
    <li>explain the <strong>Deal-Grove model</strong> x² + A·x = B·(t + τ): the linear and parabolic rate constants, and τ for a starting oxide;</li>
    <li>describe how B and B/A depend on <strong>temperature</strong> (Arrhenius) and on the ambient, dry or wet;</li>
    <li>fit the model to thickness measurements with the <strong>uncertainty on the thickness</strong>, and read the parameter uncertainties;</li>
    <li>tell a <strong>confidence band</strong> for the predicted mean from a <strong>prediction band</strong> for a new measurement;</li>
    <li>state the model's <strong>range of validity</strong>, including the thin-oxide regime, and the risk of extrapolation.</li>
  </ul>
  <p class="small" style="margin-top:16px">This is course outcome 3. The lab calibrates the model for your own simulated furnace.</p>""")

slide(f"""  <h2>How dry oxide grows</h2>
  <div style="margin-top:6px">{flux_fig()}</div>
  <p class="small" style="margin-top:12px">Deal and Grove (1965) set the three fluxes equal in steady state. The oxide grows at the silicon interface, so each nanometre of oxide consumes about 0.44 nm of silicon.</p>""")

slide(f"""  <h2>The Deal-Grove model</h2>
  <div class="eq" style="font-size:32px">x² + A·x = B·(t + τ) &nbsp;⇒&nbsp; x(t) = (A/2)·(√(1 + 4B(t + τ)/A²) − 1)</div>
  <table style="font-size:22px;width:100%">
    <tr><th>Symbol</th><th>Name</th><th>Textbook value, dry O₂, 1000 °C, (100) Si</th></tr>
    <tr><td>B</td><td>parabolic rate constant</td><td>{B_TB:.1f} nm²/min</td></tr>
    <tr><td>B/A</td><td>linear rate constant</td><td>{BA_TB:.3f} nm/min</td></tr>
    <tr><td>A</td><td>B divided by B/A</td><td>{A_TB:.0f} nm</td></tr>
    <tr><td>τ</td><td>time shift for a starting oxide x_i: τ = (x_i² + A·x_i)/B</td><td>{TAU25:.1f} min for x_i = 25 nm</td></tr>
  </table>
  <p class="small" style="margin-top:12px">Deal and Grove, J. Appl. Phys. 36, 3770 (1965): B = 772 µm²/h · e^(−1.23 eV/kT); B/A = 6.23×10⁶ µm/h · e^(−2.0 eV/kT) for (111), divided by 1.68 for (100).</p>""")

slide(f"""  <h2>Two limits</h2>
  {two(limits_fig(), [f"Thin oxide (x ≪ A ≈ {A_TB:.0f} nm): the reaction limits the rate, and x ≈ (B/A)·t.", "Thick oxide (x ≫ A): diffusion through the oxide limits it, and x ≈ √(B·t).", "The full solution moves smoothly from one to the other.", f"A bare wafer at 1000 °C grows about {float(reference_thickness(120.0, B_TB, BA_TB, 0.0)):.0f} nm in 2 hours (textbook constants)."])}""")

slide(f"""  <h2>A starting oxide: the time shift τ</h2>
  {two(tau_fig(), ["A wafer that already has x_i of oxide behaves as if it had been oxidizing for τ minutes already.", f"τ = (x_i² + A·x_i)/B = {TAU25:.1f} min for 25 nm.", "The same idea answers a two-step recipe: find the equivalent time of the oxide grown so far, then add the new time.", "τ is computed from the known x_i, not fitted."])}""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>How thick after 2 hours?</h2>
  <p>A wafer starts with 25 nm of oxide and is oxidized for 120 minutes in dry O₂ at 1000 °C, with the textbook constants: B = {B_TB:.1f} nm²/min, B/A = {BA_TB:.3f} nm/min.</p>
  <p><strong>Predict:</strong> more or less than 25 + 120 × (B/A) = {25 + 120 * BA_TB:.0f} nm?</p>
{reveal(f"Less: x = {X120:.1f} nm. With τ = {TAU25:.1f} min, x = (A/2)(√(1 + 4B(120 + τ)/A²) − 1). The linear estimate ignores that diffusion through the growing oxide already slows the rate at 25 to 90 nm, because A is only {A_TB:.0f} nm.")}""")

slide(f"""  <h2>Temperature and ambient</h2>
  {two(temp_fig(), ["B and B/A follow Arrhenius laws, C·e^(−E/kT): B with 1.23 eV, B/A with 2.0 eV (dry).", f"From 1000 to 1100 °C, B grows by a factor of {RATIO_1100:.2f}.", f"Wet (H₂O) oxidation is far faster: in 2 hours at 1000 °C, about {X2H_WET[1000]:.0f} nm from a 25 nm start, against {X2H[1000]:.0f} nm dry.", "The lab uses one temperature only: 1000 °C, dry."], size=22)}""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>How much faster at 1100 °C?</h2>
  <p>B = C₁·e^(−E₁/kT) with E₁ = 1.23 eV and k = 8.617×10⁻⁵ eV/K.</p>
  <p><strong>Predict:</strong> by what factor does B grow from 1000 °C (1273 K) to 1100 °C (1373 K)?</p>
{reveal(f"B(1100)/B(1000) = exp[(1.23/k)(1/1273.15 − 1/1373.15)] = {RATIO_1100:.2f}. A 100-degree change more than doubles the parabolic rate, which is why a furnace's temperature calibration matters so much to its oxide thickness.")}""")

slide(f"""  <h2>Calibration data from the furnace</h2>
  {two(cal_fig(), ["12 runs, 30 to 480 minutes, dry O₂ at 1000 °C, all with a measured 25.0 nm starting oxide.", "One ellipsometer measurement per run; the metrology group states σ = 0.5 nm.", "The textbook curve is close, but not right for this furnace.", "Simulated data, example key 2026."])}""")

slide(f"""  <h2>A straight line, for starting values only</h2>
  <div class="eq" style="font-size:30px">t = (x² − x_i²)·(1/B) + (x − x_i)·(1/(B/A))</div>
  <ul>
    <li>Linear in 1/B and 1/(B/A): ordinary least squares on t gives starting values. Three runs: B = {B3:.1f} nm²/min, B/A = {BA3:.4f} nm/min.</li>
    <li>But least squares assumes the noise is in t. Here t is set exactly by the recipe and the noise is in x, which sits on the right, squared.</li>
    <li>That is an <strong>errors-in-variables</strong> problem: the straight-line fit gives no honest uncertainty. Fit x(t) instead.</li>
  </ul>""")

slide(f"""  <h2>Fit x(t), with the noise on the thickness</h2>
  <div class="cols">
    <div style="flex:0 0 560px"><div style="font-family:Consolas,monospace;font-size:20px;background:{TINT};border-left:6px solid {INK};padding:14px 18px;line-height:1.5">curve_fit(model, t, x, p0,<br>&nbsp;&nbsp;sigma=0.5 nm for every x,<br>&nbsp;&nbsp;absolute_sigma=True)</div>
      <p style="font-size:22px;margin-top:14px">Free parameters: B and B/A only. τ follows from x_i.</p></div>
    <div><table style="font-size:22px;width:100%">
      <tr><th>Example (key 2026)</th><th>fit</th><th>textbook</th></tr>
      <tr><td>B (nm²/min)</td><td>{FIT['B']:.1f} ± {FIT['se_B']:.1f}</td><td>{B_TB:.1f}</td></tr>
      <tr><td>B/A (nm/min)</td><td>{FIT['BA']:.4f} ± {FIT['se_BA']:.4f}</td><td>{BA_TB:.4f}</td></tr>
      <tr><td>χ² / degrees of freedom</td><td>{FIT['chi2']:.1f} / {FIT['dof']}</td><td></td></tr>
    </table>
    <p class="small" style="margin-top:12px">absolute_sigma=True: σ is known, so the uncertainties come from σ itself, not from how this fit happens to scatter.</p></div>
  </div>""")

slide(f"""  <h2>The two estimates are correlated</h2>
  {two(ellipse_fig(), [f"Correlation {FIT['corr']:.2f}: a larger B with a smaller B/A fits almost as well.", "Each standard error alone hides that: the joint uncertainty is a narrow, tilted ellipse.", "Predictions need the full covariance, not two separate error bars.", "curve_fit returns the covariance from a linear approximation at the optimum."], width=600)}""")

slide(f"""  <h2>Does the model describe the data?</h2>
  {two(resid_fig(), [f"χ² = {FIT['chi2']:.1f} for {FIT['dof']} degrees of freedom: about what noise of σ = 0.5 nm would give.", "Residuals within ±2σ, with no trend in time.", "So, inside 30 to 480 minutes, the model fits to within the measurement noise.", "That says nothing yet about other conditions."], width=600)}""")

slide(f"""  <h2>Confidence band and prediction band</h2>
  {two(bands_zoom_fig(), [f"Predicted at 270 min: {P270['x'][0]:.2f} nm.", f"Confidence band (the mean): ± {1.96 * P270['se_mean'][0]:.2f} nm, from the parameter covariance.", f"Prediction band (one new wafer): ± {1.96 * P270['se_pred'][0]:.2f} nm, with se_pred = √(se_mean² + σ²).", "The new run's three wafers fall inside the prediction band."], size=22)}""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>How wide is the prediction band?</h2>
  <p>A calibrated model gives se_mean = {SE_Q[0]:.2f} nm at some time, and one measurement has σ = {SE_Q[1]:.2f} nm.</p>
  <p><strong>Predict:</strong> the half-width of the 95% prediction band?</p>
{reveal(f"se_pred = √({SE_Q[0]:.2f}² + {SE_Q[1]:.2f}²) = {SE_Q_PRED:.3f} nm, so the half-width is 1.96 × {SE_Q_PRED:.3f} = {1.96 * SE_Q_PRED:.2f} nm. Calibrating with more runs shrinks se_mean, but never below the measurement's own σ.")}""")

slide(f"""  <h2>Extrapolation</h2>
  <div>{extrap_fig()}</div>
  <p class="small" style="margin-top:0">Beyond 480 minutes the band widens, and it still assumes that the model is right out there. Nothing in 30 to 480 minutes tests that.</p>""")

slide(f"""  <h2>A challenging condition: thin oxide</h2>
  {two(thin_fig(), ["Real dry oxidation is faster than Deal-Grove below about 20 nm (Massoud, Plummer and Irene, 1985).", "The course's simulator includes such extra early growth: this mismatch is built in on purpose.", f"Bare wafers, 10 to 40 min: every wafer is {min(Z_THIN):.0f} to {max(Z_THIN):.0f} prediction standard errors above the prediction.", "The bands could not warn: the calibration never tested thin oxide."], size=22)}""")

slide("""  <h2>State where the model applies</h2>
  <ul>
    <li><strong>This calibration:</strong> dry O₂, 1000 °C, a starting oxide near 25 nm, 30 to 480 minutes, (100) wafers.</li>
    <li><strong>Outside it:</strong> other temperatures, wet ambient, thin oxides, long runs. The twin should flag the prediction as unsupported, or use a model built for the condition.</li>
    <li><strong>Bands</strong> describe noise and parameter uncertainty, not a wrong model.</li>
    <li><strong>Claim only what the data support:</strong> on simulated data, a good fit shows the method recovers the course's simulator, not that a real furnace follows Deal-Grove.</li>
  </ul>""")

slide("""  <h2>Summary</h2>
  <ul>
    <li>x² + A·x = B·(t + τ): linear growth for thin oxide, parabolic for thick, τ for the oxide already there.</li>
    <li>B and B/A rise steeply with temperature; wet oxidation is much faster than dry.</li>
    <li>Fit x(t) with σ on the thickness and absolute_sigma=True; B and B/A are strongly correlated.</li>
    <li>Confidence band: the mean. Prediction band: one new measurement, wider by σ.</li>
    <li>The bands widen on extrapolation and say nothing about a wrong model: test a challenging condition.</li>
  </ul>
  <div class="eq" style="font-size:25px;margin-top:16px"><strong>Next:</strong> the Module 3 lab. You write the model, calibrate it on your own furnace and test it on thin oxide.</div>""")

slide("""  <h2>Credits and license</h2>
  <p>These slides were written for SemiAcademy's <em>Digital Twins for Semiconductor Manufacturing</em> course, developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). License: <strong>CC BY 4.0</strong>, https://creativecommons.org/licenses/by/4.0/.</p>
  <p style="font-size:20px">Sources: B. E. Deal and A. S. Grove, "General Relationship for the Thermal Oxidation of Silicon", J. Appl. Phys. 36, 3770–3778 (1965), rate constants as tabulated in S. Dunham's EE 528 notes, University of Washington; H. Z. Massoud, J. D. Plummer and E. A. Irene, J. Electrochem. Soc. 132(11), 2685–2700 (1985), on the thin-oxide regime; SciPy documentation for curve_fit. Cited, not reproduced.</p>
  <p class="small">Every data value on these slides is simulated with the course's furnace model (furnace.py 1.3.0, example key 2026). Source: github.com/CSU-Physics/digital-twins-notebooks (media/m3/m3-slides.html).</p>""")

# ------------------------------------------------------------------ assemble
head_html = (T / "deck_head.html").read_text()
head_html = head_html.replace("<title>Module 1 Slides: From Bits to Qubits</title>", "<title>Module 3 Slides: Physics Model of Thermal Oxidation</title>")
assert "Physics Model of Thermal Oxidation" in head_html
head_html = head_html.replace("</style>", "  code { font-family: Consolas, \"Courier New\", monospace; font-size:0.92em; }\n</style>", 1)
bar = (T / "deck_bar.html").read_text().replace("1 / 19", f"1 / {len(slides)}")
ctrl = (T / "deck_ctrl.js").read_text()
html = head_html + "\n" + "\n".join(slides) + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
for bad in ("—", "${", "`"):
    assert bad not in html, bad
(OUT / "m3-slides.html").write_text(html)
print("wrote", OUT / "m3-slides.html", len(slides), "slides,", len(html), "bytes")
print("tau25", round(TAU25, 2), "x120", round(X120, 2), "ratio1100", round(RATIO_1100, 3), "x2h dry", {k: round(v, 1) for k, v in X2H.items()}, "wet", {k: round(v, 1) for k, v in X2H_WET.items()})
print("fit", round(FIT["B"], 2), round(FIT["BA"], 4), "p270", round(P270["x"][0], 2), "z thin", np.round(Z_THIN, 1), "se_pred q", round(SE_Q_PRED, 4))
