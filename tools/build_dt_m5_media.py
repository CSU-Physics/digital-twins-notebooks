"""Build the Module 5 slide deck (media/m5/m5-slides.html) for Digital Twins for Semiconductor Manufacturing.

Every number and point on the slides is computed here from the course's furnace model (content/m5/furnace.py), the
lab's helpers (content/m5/r2r_tools.py) and the reference functions of the Module 5 check, so the slides agree with the
lab notebook. Run from anywhere: python3 tools/build_dt_m5_media.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m5"), str(ROOT / "checks")]
import furnace as F  # noqa: E402
from r2r_tools import ALARM_LIMIT_NM, ALARM_RUNS, Q_NM2, R_NM2, closed_loop, dg_thickness, qualification_offset, summary  # noqa: E402
from dt_m5_check import reference_alarm, reference_ewma, reference_kalman, reference_next_time  # noqa: E402

T = ROOT / "tools"
OUT = ROOT / "media" / "m5"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, INK, MUTED, LINE, TINT, OK, BAD, BLUE, GOLD = "#99004C", "#24313D", "#6A626B", "#E4DCE0", "#F8F4F6", "#2E7D4F", "#B3261E", "#2F5E8E", "#A86B00"

# ------------------------------------------------------------------ numbers used on the slides (example key 2026)
KEY = 2026
S = F.r2r_setup(KEY)
TR = F.r2r_truth(KEY)
D_REF = qualification_offset(S)
TWIN90 = float(dg_thickness(90.0, S["B"], S["BA"], S["xi_nm"]))
NT = lambda d: reference_next_time(d, S["target_nm"], S["B"], S["BA"], S["xi_nm"], S["t_limits"])
LOG_F = closed_loop(KEY, "fixed")
LOG_E = closed_loop(KEY, "ewma", next_time=reference_next_time, ewma_update=reference_ewma, lam=0.2)
LOG_K = closed_loop(KEY, "kalman", next_time=reference_next_time, kalman_update=reference_kalman)
LOG_A = closed_loop(KEY, "kalman", next_time=reference_next_time, kalman_update=reference_kalman, mode="advisory", review_every=3)
SUM = {"fixed 90 min": summary(LOG_F), "EWMA, λ = 0.2": summary(LOG_E), "Kalman": summary(LOG_K)}
SUM_A = summary(LOG_A)
LAMS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
LAM_MAE = {lam: summary(closed_loop(KEY, "ewma", next_time=reference_next_time, ewma_update=reference_ewma, lam=lam))["mae_nm"] for lam in LAMS}
BEST_LAM = min(LAM_MAE, key=LAM_MAE.get)
GAP = int(LOG_K.loc[LOG_K["measured_nm"].isna(), "run"].iloc[0])
ALARM_K = reference_alarm(LOG_K["d_hat_nm"] - D_REF, ALARM_LIMIT_NM, ALARM_RUNS)
FAIL_F = closed_loop(KEY, "fixed", scenario="failing")
FAIL_K = closed_loop(KEY, "kalman", next_time=reference_next_time, kalman_update=reference_kalman, scenario="failing")
ALARM_FK = reference_alarm(FAIL_K["d_hat_nm"] - D_REF, ALARM_LIMIT_NM, ALARM_RUNS)
HELD_FK = FAIL_K.loc[FAIL_K["held"], "run"].tolist()
SUM_FK = summary(FAIL_K)
# sensitivity: this furnace's true growth at 90 minutes for a true temperature 0 to 4 degrees below the qualification run
_B, _BA = F._rates_array(np.array([1000 + TR["tc_offset_C"] - b for b in range(5)]))
SENS = F._grow_many(np.full(5, 90.0), 1.5, _B * TR["fB"], _BA * TR["fBA"], TR["thin_C"], TR["thin_L"])
PER_C = float(np.mean(-np.diff(SENS)))
DTDX = 1.0 / (S["B"] / (2 * TWIN90 + S["B"] / S["BA"]))       # minutes per nm of the twin's growth at 90 min
# Kalman gain from P = R
P, GAINS = R_NM2, []
for run in range(30):
    Pm = P + Q_NM2
    GAINS.append(Pm / (Pm + R_NM2))
    P = (1 - GAINS[-1]) * Pm
P_ST = (Q_NM2 + np.sqrt(Q_NM2 ** 2 + 4 * Q_NM2 * R_NM2)) / 2
K_ST = P_ST / (P_ST + R_NM2)
# pause-and-predict values
PP_E = reference_ewma(4.90, 4.50, 0.2)
PP_K = reference_kalman(4.90, 0.035, 4.50, Q_NM2, R_NM2)
PP_T = reference_next_time(4.20, S["target_nm"], S["B"], S["BA"], S["xi_nm"], S["t_limits"])
assert round(PP_E, 2) == 4.82 and round(PP_K[0], 3) == 4.812
assert SUM["Kalman"]["mae_nm"] < SUM["EWMA, λ = 0.2"]["mae_nm"] < SUM["fixed 90 min"]["mae_nm"] / 4
assert ALARM_FK is not None and HELD_FK and ALARM_FK < HELD_FK[0] and SUM["Kalman"]["held_runs"] == 0


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


# ------------------------------------------------------------------ figures
def loop_fig(w=1130, h_=300, mid="l1"):
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 1130 300" role="img" aria-label="The run-to-run loop: the twin recommends an oxidation time, the furnace runs, the monitor wafer is measured, the estimator updates the offset, and the twin uses it for the next run">',
         arrow_def(mid, MAROON)]
    boxes = [(10, "twin", "Deal-Grove + offset d̂"), (295, "recipe rule", "next oxidation time"), (580, "furnace run", "the product wafers"),
             (865, "monitor wafer", "measured thickness")]
    for x, a, b in boxes:
        s.append(f'<rect x="{x}" y="40" width="255" height="100" rx="10" fill="#fff" stroke="{MAROON}" stroke-width="2.5"/>')
        s.append(txt(x + 127, 82, a, 22, INK, weight=600))
        s.append(txt(x + 127, 112, b, 18, MUTED))
    for x in (265, 550, 835):
        s.append(f'<line x1="{x + 2}" y1="90" x2="{x + 26}" y2="90" stroke="{MAROON}" stroke-width="4" marker-end="url(#{mid})"/>')
    s.append(f'<rect x="420" y="200" width="290" height="80" rx="10" fill="{TINT}" stroke="{MAROON}" stroke-width="2.5"/>')
    s.append(txt(565, 234, "estimator", 22, INK, weight=600))
    s.append(txt(565, 262, "EWMA or Kalman: update d̂", 18, MUTED))
    s.append(f'<path d="M992 142 L992 240 L716 240" fill="none" stroke="{MAROON}" stroke-width="4" marker-end="url(#{mid})"/>')
    s.append(f'<path d="M418 240 L137 240 L137 146" fill="none" stroke="{MAROON}" stroke-width="4" marker-end="url(#{mid})"/>')
    s.append(txt(860, 230, "y = measured − predicted", 17, MUTED))
    s.append(txt(270, 230, "before the next run", 17, MUTED))
    s.append("</svg>")
    return "".join(s)


def bias_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    runs = np.arange(0, 61)
    bias = np.array(TR["bias_C"]["drift"])
    true_T = 1000 + TR["tc_offset_C"] - bias
    X_ = lambda v: x0 + v / 60 * (x1 - x0)
    Y_ = lambda v: y0 - (v - 994) / 8 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Over 60 runs the thermocouple reading stays at 1000 degrees while the true temperature falls by about five degrees">']
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 10, 20, 30, 40, 50, 60], [994, 996, 998, 1000, 1002], "run", "temperature (°C)")
    s.append(poly(runs, np.full(runs.size, 1000.0), X_, Y_, MUTED, 3.5, "8 6"))
    s.append(poly(runs, true_T, X_, Y_, BAD, 3.5))
    legend(s, x0 + 24, y1 + 26, [("dash", MUTED, "thermocouple reading, held at the setpoint"), ("line", BAD, "true temperature: setpoint − bias")], 17)
    s.append(txt(X_(30), Y_(994.6), f"bias at run 60: {bias[-1]:.1f} °C", 17, BAD, "middle", 600))
    s.append("</svg>")
    return "".join(s)


def offset_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    X_ = lambda v: x0 + v / 60 * (x1 - x0)
    Y_ = lambda v: y0 - (v - 52) / 16 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Fixed recipe: the twin predicts the same thickness every run, the monitor wafers measure about six nanometres more at first and less and less as the furnace drifts">',
         arrow_def("o1", MAROON)]
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 10, 20, 30, 40, 50, 60], [52, 56, 60, 64, 68], "run", "thickness (nm)")
    s.append(poly([0, 60], [TWIN90, TWIN90], X_, Y_, BLUE, 3.5))
    s.append(poly([0, 60], [S["target_nm"]] * 2, X_, Y_, INK, 2, "6 5"))
    pts = [(0, S["qual_thickness_nm"])] + [(r, m) for r, m in zip(LOG_F["run"], LOG_F["measured_nm"]) if m == m]
    for a, b in pts:
        s.append(f'<circle cx="{X_(a):.1f}" cy="{Y_(b):.1f}" r="4" fill="{MAROON}"/>')
    s.append(f'<line x1="{X_(1.5):.1f}" y1="{Y_(TWIN90) - 4:.1f}" x2="{X_(1.5):.1f}" y2="{Y_(S["qual_thickness_nm"]) + 8:.1f}" stroke="{MAROON}" stroke-width="3" marker-start="url(#o1)" marker-end="url(#o1)"/>')
    s.append(txt(X_(3), (Y_(TWIN90) + Y_(S["qual_thickness_nm"])) / 2 + 6, f"d_ref = {D_REF:.2f} nm", 18, MAROON, "start", 600))
    legend(s, x0 + 150, y1 + 24, [("dot", MAROON, "monitor wafer, measured"), ("line", BLUE, f"twin's prediction for 90 min: {TWIN90:.2f} nm"), ("dash", INK, f"target {S['target_nm']} nm")], 16)
    s.append("</svg>")
    return "".join(s)


def runs_fig(logs, w=620, h_=390, ylo=57, yhi=63, label="Drift scenario"):
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    X_ = lambda v: x0 + v / 60 * (x1 - x0)
    Y_ = lambda v: y0 - (v - ylo) / (yhi - ylo) * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="{label}: achieved thickness of each run against the run number for each controller, with the target">']
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 10, 20, 30, 40, 50, 60], list(range(ylo, yhi + 1)), "run", "achieved thickness (nm)")
    s.append(poly([0, 60], [S["target_nm"]] * 2, X_, Y_, INK, 2, "6 5"))
    items = []
    for name, log, color in logs:
        s.append(poly(log["run"], log["achieved_nm"], X_, Y_, color, 2.5))
        items.append(("line", color, name))
    items.append(("dash", INK, "target"))
    legend(s, x0 + 24, y0 - 26 * len(items) - 4, items, 17)
    s.append("</svg>")
    return "".join(s)


def weights_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 30
    J = 15
    X_ = lambda j: x0 + 18 + j * (x1 - x0 - 36) / (J - 1)
    Y_ = lambda v: y0 - v / 0.5 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Weights of past observations in an EWMA: with lambda 0.5 they fall off within a few runs, with lambda 0.2 they spread over many runs">']
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 2, 4, 6, 8, 10, 12, 14], [0, 0.1, 0.2, 0.3, 0.4, 0.5], "runs back (0 = the newest observation)", "weight", yfmt="{:.1f}")
    for j in range(J):
        for lam, color, dx in ((0.5, MAROON, -7), (0.2, BLUE, 7)):
            v = lam * (1 - lam) ** j
            s.append(f'<rect x="{X_(j) + dx - 6:.1f}" y="{Y_(v):.1f}" width="12" height="{y0 - Y_(v):.1f}" fill="{color}"/>')
    legend(s, x1 - 220, y1 + 24, [("sq", MAROON, "λ = 0.5"), ("sq", BLUE, "λ = 0.2")], 18)
    s.append("</svg>")
    return "".join(s)


def gain_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    X_ = lambda v: x0 + (v - 1) / 29 * (x1 - x0)
    Y_ = lambda v: y0 - v / 0.6 * (y0 - y1)
    # gain with a measurement in every run, and with run 15 missing
    P, G2 = R_NM2, []
    for run in range(1, 31):
        Pm = P + Q_NM2
        if run == 15:
            G2.append(None)
            P = Pm
            continue
        K = Pm / (Pm + R_NM2)
        G2.append(K)
        P = (1 - K) * Pm
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Kalman gain run by run: it starts near 0.5 and settles near 0.22; after a run with no measurement it jumps up a little">']
    axes(s, X_, Y_, x0, x1, y0, y1, [1, 5, 10, 15, 20, 25, 30], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "run", "Kalman gain K", yfmt="{:.1f}")
    s.append(poly([1, 30], [K_ST, K_ST], X_, Y_, MUTED, 2, "6 5"))
    s.append(poly(range(1, 31), GAINS, X_, Y_, MAROON, 3.5))
    for r, g in enumerate(G2, start=1):
        if g is not None:
            s.append(f'<circle cx="{X_(r):.1f}" cy="{Y_(g):.1f}" r="4" fill="{BLUE}"/>')
    s.append(txt(X_(15), Y_(G2[15]) - 16, "after a missing run", 16, BLUE))
    legend(s, x0 + 150, y1 + 30, [("line", MAROON, "a measurement every run"), ("dot", BLUE, "run 15 not measured"), ("dash", MUTED, f"steady gain {K_ST:.2f}")], 17)
    s.append("</svg>")
    return "".join(s)


def lambda_fig():
    w, h_ = 620, 390
    x0, x1, y0, y1 = 92, w - 20, h_ - 70, 20
    import math
    X_ = lambda v: x0 + (math.log10(v) + 1.4) / 1.4 * (x1 - x0)
    Y_ = lambda v: y0 - v / 0.8 * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Mean absolute error of the EWMA controller against lambda: high for small lambda (lag), lowest near 0.3, rising again toward lambda 1 (noise)">']
    axes(s, X_, Y_, x0, x1, y0, y1, [0.05, 0.1, 0.2, 0.3, 0.5, 1.0], [0, 0.2, 0.4, 0.6, 0.8], "λ (log scale)", "mean absolute error (nm)", yfmt="{:.1f}")
    s.append(poly(LAMS, [LAM_MAE[l] for l in LAMS], X_, Y_, MAROON, 3.5))
    for l in LAMS:
        s.append(f'<circle cx="{X_(l):.1f}" cy="{Y_(LAM_MAE[l]):.1f}" r="5" fill="{MAROON}"/>')
    s.append(poly([0.05, 1.0], [SUM["Kalman"]["mae_nm"]] * 2, X_, Y_, BLUE, 2.5, "7 5"))
    s.append(txt(X_(0.06), Y_(LAM_MAE[0.05]) - 14, "lags the drift", 17, MUTED, "start"))
    s.append(txt(X_(1.0) - 4, Y_(LAM_MAE[1.0]) - 16, "follows the noise", 17, MUTED, "end"))
    legend(s, x0 + 200, y1 + 26, [("line", MAROON, "EWMA controller"), ("dash", BLUE, "Kalman controller")], 17)
    s.append("</svg>")
    return "".join(s)


def fail_fig():
    w, h_ = 1130, 380
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Failing thermocouple: the Kalman controller lengthens the oxidation time until it is held at 100 minutes; from then on the thickness falls below target. The alarm fires about ten runs before the first held run">']
    # left: oxidation time
    x0, x1, y0, y1 = 86, 520, h_ - 70, 20
    X_ = lambda v: x0 + v / 60 * (x1 - x0)
    Y_ = lambda v: y0 - (v - 80) / 25 * (y0 - y1)
    axes(s, X_, Y_, x0, x1, y0, y1, [0, 10, 20, 30, 40, 50, 60], [80, 85, 90, 95, 100, 105], "run", "oxidation time (min)")
    s.append(poly([0, 60], [100, 100], X_, Y_, BAD, 2.5, "7 5"))
    s.append(txt(X_(2), Y_(100) - 10, "recipe limit, 100 min", 17, BAD, "start"))
    s.append(poly(FAIL_K["run"], FAIL_K["ox_time_min"], X_, Y_, BLUE, 3.5))
    for xr, lab, col in ((ALARM_FK, f"alarm, run {ALARM_FK}", GOLD), (HELD_FK[0], f"held from run {HELD_FK[0]}", BAD)):
        s.append(f'<line x1="{X_(xr):.1f}" y1="{y0}" x2="{X_(xr):.1f}" y2="{y1 + 40}" stroke="{col}" stroke-width="2.5"/>')
    s.append(txt(X_(ALARM_FK) - 6, y1 + 34, f"alarm, run {ALARM_FK}", 17, GOLD, "end", 600))
    s.append(txt(X_(HELD_FK[0]) + 6, y0 - 12, f"held from run {HELD_FK[0]}", 17, BAD, "start", 600))
    # right: thickness
    u0, u1 = 660, w - 20
    U_ = lambda v: u0 + v / 60 * (u1 - u0)
    V_ = lambda v: y0 - (v - 53) / 10 * (y0 - y1)
    for v in range(54, 64, 2):
        s.append(f'<line x1="{u0}" y1="{V_(v):.1f}" x2="{u1}" y2="{V_(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(u0 - 10, V_(v) + 7, f"{v}", 19, MUTED, "end"))
    for v in (0, 10, 20, 30, 40, 50, 60):
        s.append(txt(U_(v), y0 + 28, f"{v}", 19, MUTED))
    s.append(f'<line x1="{u0}" y1="{y0}" x2="{u1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt((u0 + u1) / 2, y0 + 56, "run", 20, MUTED))
    s.append(txt(u0 - 56, (y0 + y1) / 2, "achieved thickness (nm)", 20, MUTED, extra=f' transform="rotate(-90 {u0 - 56} {(y0 + y1) / 2:.1f})"'))
    s.append(poly([0, 60], [S["target_nm"]] * 2, U_, V_, INK, 2, "6 5"))
    s.append(poly(FAIL_F["run"], FAIL_F["achieved_nm"], U_, V_, MUTED, 2.5))
    s.append(poly(FAIL_K["run"], FAIL_K["achieved_nm"], U_, V_, BLUE, 3))
    legend(s, u0 + 20, y0 - 84, [("line", BLUE, "Kalman controller"), ("line", MUTED, "fixed 90 min"), ("dash", INK, "target")], 17)
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ slides
FOOT = "Module 5 · Keeping the twin in sync"
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


slide(f"""  <div class="kicker">Digital Twins for Semiconductor Manufacturing · Module 5</div>
  <h1>Keeping the twin in sync</h1>
  <p style="font-size:30px;max-width:940px">Estimate how far the furnace has drifted from its twin, correct the next run, and know when to call for a calibration.</p>
  <div style="margin-top:10px">{loop_fig(1060, 281, "l0")}</div>""", "slide title-slide")

slide("""  <h2>By the end of this module you can</h2>
  <ul>
    <li>explain how a growing <strong>thermocouple bias</strong> thins the oxide while the furnace's own data look normal;</li>
    <li>define the <strong>offset d</strong> between the furnace and its twin, and estimate it with an <strong>EWMA</strong> and a one-variable <strong>Kalman filter</strong>;</li>
    <li>set the next oxidation time with the <strong>inverted Deal-Grove model</strong>, within the recipe limits;</li>
    <li>compare controllers on <strong>the same disturbances</strong> by their mean absolute error, largest deviation and runs held at a limit;</li>
    <li>tell <strong>advisory</strong> from <strong>automatic</strong> mode, and decide when to <strong>recalibrate</strong> or call <strong>maintenance</strong>.</li>
  </ul>
  <p class="small" style="margin-top:12px">Course outcome 5. The update equations are given, not derived; the lab runs every controller on your own furnace.</p>""")

slide(f"""  <h2>Why the twin falls out of sync</h2>
  {two(bias_fig(), ["A thermocouple's calibration changes with use at high temperature: its wires are contaminated and damaged, slowly and by an unknown amount (Machin, Tucker and Pearce, 2021).", "Here it reads high by a bias b. The controller holds the reading at 1000 °C, so the wafers see 1000 °C − b.", "Every trace says 1000 °C: the furnace's own data cannot show the drift.", "Only a measurement of the product, the oxide thickness, can."], size=22)}""")

slide(f"""  <h2>How much thickness a degree costs</h2>
  <table style="font-size:22px;width:100%">
    <tr><th>true temperature below the qualification run</th><th>0 °C</th><th>1 °C</th><th>2 °C</th><th>3 °C</th><th>4 °C</th></tr>
    <tr><td>thickness after 90 minutes (nm)</td>{"".join(f"<td>{v:.2f}</td>" for v in SENS)}</tr>
  </table>
  <ul style="margin-top:18px">
    <li>About <strong>{PER_C:.2f} nm less oxide per degree</strong> at 90 minutes and 1000 °C, from the Arrhenius temperature dependence of both rate constants (Module 3).</li>
    <li>The twin grows {1 / DTDX:.2f} nm per minute at 90 minutes, so each degree of bias costs about <strong>{PER_C * DTDX:.1f} minute</strong> of extra oxidation.</li>
    <li>A bias of 5 °C makes the oxide about {5 * PER_C:.1f} nm too thin: several times the run-to-run noise.</li>
  </ul>
  <p class="small" style="margin-top:12px">Simulated furnace, example key 2026; the bias changes the true temperature only.</p>""")

slide(f"""  <h2>The state the twin estimates: the offset d</h2>
  {two(offset_fig(), ["<strong>d = measured monitor-wafer thickness − the twin's prediction for that run's recipe.</strong>", f"At qualification, d_ref = {D_REF:.2f} nm: Deal-Grove leaves out the fast early growth on bare wafers (Module 3). A known, steady offset is no problem.", "As the bias grows, d falls: that change is the drift.", "One monitor wafer per run is the only observation, and some runs have none."], size=22)}""")

slide(f"""  <h2>A fixed recipe on a drifting furnace</h2>
  {two(runs_fig([("fixed 90 min", LOG_F, MUTED)], label="Fixed recipe"), [f"Every run at 90 minutes: the achieved thickness falls about {S['target_nm'] - LOG_F['achieved_nm'].iloc[-5:].mean():.0f} nm below the target over 60 runs.", f"Mean absolute error {SUM['fixed 90 min']['mae_nm']:.2f} nm; largest deviation {SUM['fixed 90 min']['max_dev_nm']:.2f} nm.", "The run-to-run noise, about 0.4 nm in one observation, hides the trend from run to run; the drift wins over many runs.", "Simulated data, example key 2026."], size=22)}""")

slide(f"""  <h2>The run-to-run loop</h2>
  <div style="margin-top:6px">{loop_fig()}</div>
  <ul style="font-size:22px">
    <li>Run-to-run control combines statistical process control with feedback between runs (Sachs, Hu and Ingolfsson, 1995); it is a standard part of fab process control.</li>
    <li>In a fab it runs in the factory's process control systems; SEMI E133 standardizes how such systems, run-to-run control and fault detection among them, share data.</li>
  </ul>""")

slide(f"""  <h2>EWMA: a weighted memory of the offset</h2>
  <div class="eq" style="font-size:30px">d̂_new = λ · y + (1 − λ) · d̂_old &nbsp;&nbsp; <span style="font-size:22px">(no measurement: d̂_new = d̂_old)</span></div>
  {two(weights_fig(), ["y is the offset observed in the newest run: measured − predicted.", "Unrolled, the estimate weights the observations λ, λ(1 − λ), λ(1 − λ)², ... going back.", "Large λ: follows changes fast and passes on noise. Small λ: smooth, but lags a drift.", "λ = 1 trusts only the last run."], width=620, size=22)}""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>One EWMA update</h2>
  <p>The estimate is d̂ = 4.90 nm. The next run's monitor wafer gives an offset y = 4.50 nm. λ = 0.2.</p>
  <p><strong>Predict:</strong> the new estimate. Then: what if that run had no measurement?</p>
{reveal(f"d̂ = 0.2 × 4.50 + 0.8 × 4.90 = 0.90 + 3.92 = {PP_E:.2f} nm: one fifth of the way from 4.90 toward 4.50. With no measurement there is nothing to blend in, and the estimate stays at 4.90 nm.")}""")

slide(f"""  <h2>Worked example: the EWMA pulls the thickness back</h2>
  {two(runs_fig([("fixed 90 min", LOG_F, MUTED), ("EWMA, λ = 0.2", LOG_E, MAROON)], label="Fixed recipe and EWMA controller"), ["Same furnace, same bias, same noise: only the recipe decisions differ.", f"The EWMA controller lengthens the time from {LOG_E['ox_time_min'].iloc[0]:.1f} to {LOG_E['ox_time_min'].iloc[-1]:.1f} minutes as the furnace cools.", f"Mean absolute error {SUM['EWMA, λ = 0.2']['mae_nm']:.2f} nm instead of {SUM['fixed 90 min']['mae_nm']:.2f} nm.", "What remains is mostly each run's own noise, which no controller can predict."], size=22)}""")

slide(f"""  <h2>The recipe rule: invert Deal-Grove</h2>
  <div class="eq" style="font-size:28px">x = target − d̂ &nbsp;&nbsp;&nbsp; t = (x² − xᵢ² + A·(x − xᵢ)) / B, &nbsp; A = B / (B/A)</div>
  <ul style="font-size:23px">
    <li>The furnace grows the twin's prediction <strong>plus d</strong>, so the twin aims at the target <strong>minus d̂</strong>.</li>
    <li>From x² + A·x = B·(t + τ), with τ = (xᵢ² + A·xᵢ)/B: the same Deal-Grove model as Module 3, solved for t.</li>
    <li><strong>Clip</strong> t to the recipe limits, 80 to 100 minutes. A clipped run is <strong>held at a limit</strong>: it cannot reach the target.</li>
    <li>Example furnace: B = {S['B']} nm²/min, B/A = {S['BA']} nm/min, target {S['target_nm']} nm. With d̂ = d_ref = {D_REF:.2f} nm: t = {NT(D_REF):.2f} min.</li>
  </ul>""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>Set the next run</h2>
  <p>Same furnace: B = {S['B']} nm²/min, B/A = {S['BA']} nm/min, xᵢ = 1.5 nm, target {S['target_nm']} nm, limits 80 to 100 minutes. After some drift, d̂ = 4.20 nm.</p>
  <p><strong>Predict:</strong> the oxidation time. Longer or shorter than {NT(D_REF):.2f} minutes?</p>
{reveal(f"x = {S['target_nm']} − 4.20 = {S['target_nm'] - 4.20:.2f} nm; A = {S['B']} / {S['BA']} = {S['B'] / S['BA']:.1f} nm; t = ({S['target_nm'] - 4.2:.2f}² − 1.5² + {S['B'] / S['BA']:.1f} × {S['target_nm'] - 4.2 - 1.5:.2f}) / {S['B']} = {PP_T:.2f} min. Longer: the furnace now grows less, so the twin must aim higher. It is inside the limits.")}""")

slide(f"""  <h2>A one-variable Kalman filter</h2>
  <div class="eq" style="font-size:25px">model: d_k = d_(k−1) + w (variance Q) &nbsp;&nbsp; y_k = d_k + v (variance R)</div>
  <div class="eq" style="font-size:25px">P⁻ = P + Q &nbsp;&nbsp; K = P⁻ / (P⁻ + R) &nbsp;&nbsp; d̂ = d̂ + K · (y − d̂) &nbsp;&nbsp; P = (1 − K) · P⁻</div>
  <ul style="font-size:22px">
    <li>P is the variance of the estimate. Q: how much the offset may move per run. R: the noise of one observation.</li>
    <li>R = {R_NM2} nm² from the qualification study (temperature noise and the monitor wafer's measurement); Q = {Q_NM2} nm², a little larger than the random part of the drift.</li>
    <li>No measurement: only P⁻ = P + Q. The estimate stays; its uncertainty grows.</li>
    <li>Kalman (1960); for one variable these four lines are the whole filter.</li>
  </ul>""")

slide(f"""  <h2>The gain settles, and an EWMA appears</h2>
  {two(gain_fig(), [f"From P = R, the gain starts at {GAINS[0]:.2f}: the filter learns fast at first.", f"It settles at {K_ST:.2f} within about ten runs: from then on the filter is an EWMA with λ = {K_ST:.2f}.", "Muth (1960): exponential smoothing is the best forecast for a random walk observed with noise.", "After a missing run, P has grown, so the next measurement gets more weight."], size=22)}""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>One Kalman update</h2>
  <p>d̂ = 4.90 nm with P = 0.035 nm². The next observation is y = 4.50 nm. Q = {Q_NM2} nm², R = {R_NM2} nm².</p>
  <p><strong>Predict:</strong> the gain, the new estimate and its variance.</p>
{reveal(f"P⁻ = 0.035 + 0.01 = 0.045; K = 0.045 / (0.045 + 0.16) = {0.045 / 0.205:.3f}; d̂ = 4.90 + {0.045 / 0.205:.3f} × (4.50 − 4.90) = {PP_K[0]:.3f} nm; P = (1 − {0.045 / 0.205:.3f}) × 0.045 = {PP_K[1]:.4f} nm². The gain is close to the EWMA's 0.2, and so is the result.")}""")

slide(f"""  <h2>Three controllers, the same disturbances</h2>
  <table style="font-size:23px;width:100%">
    <tr><th>controller</th><th>mean absolute error (nm)</th><th>largest deviation (nm)</th><th>runs held at a limit</th></tr>
    {"".join(f"<tr><td>{n}</td><td>{v['mae_nm']:.3f}</td><td>{v['max_dev_nm']:.3f}</td><td>{v['held_runs']}</td></tr>" for n, v in SUM.items())}
  </table>
  <ul style="font-size:22px;margin-top:14px">
    <li>Same bias trajectory, same noise in each run, the same missing measurement (run {GAP}); the furnace simulator is rerun under each controller's own decisions.</li>
    <li>Both controllers come about {SUM['fixed 90 min']['mae_nm'] / SUM['Kalman']['mae_nm']:.0f} times closer to the target than the fixed recipe. The Kalman filter is a little better: it settles faster, and its steady gain, {K_ST:.2f}, is a little above 0.2.</li>
    <li>Scored on the achieved thickness, which only a simulation knows; a fab sees the monitor wafer.</li>
  </ul>""")

slide(f"""  <h2>Choosing λ: noise against lag</h2>
  {two(lambda_fig(), [f"Small λ averages over many runs and trails the drift: λ = 0.05 gives {LAM_MAE[0.05]:.2f} nm.", f"λ = 1 copies each run's noise into the next recipe: {LAM_MAE[1.0]:.2f} nm.", f"In between, near λ = {BEST_LAM:g}, the error is lowest: {LAM_MAE[BEST_LAM]:.2f} nm.", "The best λ depends on how fast the offset moves compared with the noise: what Q and R state."], size=22)}""")

slide(f"""  <h2>Gaps, advisory mode and automatic mode</h2>
  <ul>
    <li><strong>No measurement</strong> (run {GAP} here): both estimators keep d̂, so the next recipe is unchanged; the Kalman filter also widens P. A twin should report the extra uncertainty.</li>
    <li><strong>Advisory mode:</strong> the controller recommends and an engineer accepts each change. Reviewed once every three runs, the Kalman controller's error rises only from {SUM['Kalman']['mae_nm']:.3f} to {SUM_A['mae_nm']:.3f} nm here, because the drift is slow.</li>
    <li><strong>Automatic mode:</strong> every recommendation runs at once. No delay and no workload, but nobody looks at each decision: the controller needs <strong>limits</strong> and <strong>alarms</strong>.</li>
    <li>A new controller usually starts in advisory mode, until its recommendations have earned trust.</li>
  </ul>""")

slide(f"""  <h2>When to recalibrate, when to call maintenance</h2>
  <div>{fail_fig()}</div>
  <p class="small" style="margin-top:0">Recalibrate when |d̂ − d_ref| &gt; {ALARM_LIMIT_NM} nm for {ALARM_RUNS} runs in a row: the drift is about 4 °C. In the drift scenario the alarm fires at run {ALARM_K} while the thickness is still on target. A failing thermocouple (above) drives the time to its limit: held for {SUM_FK['held_runs']} runs, off target by up to {SUM_FK['max_dev_nm']:.1f} nm. The alarm came {HELD_FK[0] - ALARM_FK} runs earlier: stop and call maintenance.</p>""")

slide("""  <h2>Summary</h2>
  <ul>
    <li>A growing thermocouple bias thins the oxide while every trace looks normal; only the measured thickness shows it.</li>
    <li>The twin tracks one number, the offset d, with an EWMA or a Kalman filter.</li>
    <li>The recipe rule inverts Deal-Grove to aim at the target minus d̂, within the recipe limits.</li>
    <li>Compare controllers on the same disturbances: mean error, largest deviation, runs held.</li>
    <li>A controller hides the drift it corrects: watch d̂, recalibrate in time, call maintenance when the recipe runs out of range.</li>
  </ul>
  <div class="eq" style="font-size:25px;margin-top:10px"><strong>Next:</strong> the Module 5 lab: two estimators, the recipe rule and the alarm, in closed loop.</div>""")

slide("""  <h2>Credits and license</h2>
  <p>These slides were written for SemiAcademy's <em>Digital Twins for Semiconductor Manufacturing</em> course, developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). License: <strong>CC BY 4.0</strong>, https://creativecommons.org/licenses/by/4.0/.</p>
  <p style="font-size:19px">Sources: E. Sachs, A. Hu and A. Ingolfsson, "Run by run process control: combining SPC and feedback control", IEEE Transactions on Semiconductor Manufacturing 8(1), 26-43 (1995); R. E. Kalman, "A new approach to linear filtering and prediction problems", Journal of Basic Engineering 82(1), 35-45 (1960); J. F. Muth, "Optimal properties of exponentially weighted forecasts", Journal of the American Statistical Association 55(290), 299-306 (1960); SEMI E133, Specification for Automated Process Control Systems Interface; J. Machin, D. Tucker and J. Pearce, "A comprehensive survey of reported thermocouple drift rates since 1972", International Journal of Thermophysics 42, 139 (2021). Cited, not reproduced.</p>
  <p class="small">Every data value on these slides is simulated with the course's furnace model (furnace.py 1.5.0, example key 2026). Source: github.com/CSU-Physics/digital-twins-notebooks (media/m5/m5-slides.html).</p>""")

# ------------------------------------------------------------------ assemble
head_html = (T / "deck_head.html").read_text()
head_html = head_html.replace("<title>Module 1 Slides: From Bits to Qubits</title>", "<title>Module 5 Slides: Keeping the Twin in Sync</title>")
assert "Keeping the Twin in Sync" in head_html
head_html = head_html.replace("</style>", "  code { font-family: Consolas, \"Courier New\", monospace; font-size:0.92em; }\n</style>", 1)
bar = (T / "deck_bar.html").read_text().replace("1 / 19", f"1 / {len(slides)}")
ctrl = (T / "deck_ctrl.js").read_text()
html = head_html + "\n" + "\n".join(slides) + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
for bad in ("—", "–", "${", "`"):
    assert bad not in html, bad
(OUT / "m5-slides.html").write_text(html)
print("wrote", OUT / "m5-slides.html", len(slides), "slides,", len(html), "bytes")
print("summary", {k: {a: round(b, 3) for a, b in v.items()} for k, v in SUM.items()}, "advisory", round(SUM_A["mae_nm"], 3))
print("lambda", {k: round(v, 3) for k, v in LAM_MAE.items()}, "best", BEST_LAM, "gap", GAP, "alarm drift", ALARM_K, "failing", ALARM_FK, HELD_FK[0], SUM_FK)
print("per C", round(PER_C, 3), "min per nm", round(DTDX, 3), "gains", np.round(GAINS[:3], 3), "steady", round(K_ST, 4), "PP", PP_E, PP_K, round(PP_T, 3))
