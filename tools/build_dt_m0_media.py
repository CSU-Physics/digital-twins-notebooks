"""Build the Module 0 slide deck (media/m0/m0-slides.html) for Digital Twins for Semiconductor Manufacturing.

Every number and curve on the slides is computed here from the course's furnace model (content/m0/furnace.py)
and the reference functions of the Module 0 check, so the slides agree with the lab notebook.
Run from anywhere: python3 tools/build_dt_m0_media.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m0"), str(ROOT / "checks")]
import furnace  # noqa: E402
from furnace import HOLD_START_S, furnace_log  # noqa: E402
from dt_m0_check import hold_readings, reference_moving_average  # noqa: E402

T = ROOT / "tools"
OUT = ROOT / "media" / "m0"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, INK, MUTED, LINE, TINT = "#99004C", "#24313D", "#6A626B", "#E4DCE0", "#F8F4F6"

# ------------------------------------------------------------------ numbers used on the slides
ex = furnace_log(2026)
z = ex["zone_temp_C"].to_numpy()
t = ex["time_s"].to_numpy()
miss_t = t[np.isnan(z)].astype(int).tolist()
hold_rows = ex.loc[ex["time_s"] >= HOLD_START_S, "zone_temp_C"]
h = hold_readings(ex)
ma = reference_moving_average(h, 5)
R = np.array([999.6, 1000.4, 1000.1, 999.8, 1000.3, 999.9, 1000.2, 999.7])
N = dict(
    rows=len(ex), nmiss=int(np.isnan(z).sum()), hold_rows=len(hold_rows), hold_n=h.size,
    hold_mean=h.mean(), hold_sd=h.std(ddof=1), ma_n=ma.size, ma_sd=ma.std(ddof=1),
    zero_mean=np.nan_to_num(hold_rows.to_numpy()).mean(), zero_sd=np.nan_to_num(hold_rows.to_numpy()).std(ddof=1),
    ff_mean=hold_rows.ffill().mean(), ff_sd=hold_rows.ffill().std(),
    r_mean=R.mean(), r_s=R.std(ddof=1), r_p=R.std(), bug0=np.convolve(h, np.ones(5), "valid")[0],
)
N["ratio"] = N["hold_sd"] / N["ma_sd"]
assert N["nmiss"] == 4 and N["hold_rows"] == 61 and N["hold_n"] == 60 and N["ma_n"] == 56
assert f"{N['r_s']:.3f}" == "0.293" and f"{N['r_p']:.3f}" == "0.274" and abs(N["r_mean"] - 1000) < 1e-9


def true_T(tt):
    return furnace.SETPOINT_C - (furnace.SETPOINT_C - furnace.LOAD_C) * np.exp(-np.asarray(tt, float) / furnace.TAU_S)


LAG_T = 100
lag_err = true_T(LAG_T) - true_T(np.arange(LAG_T - 40, LAG_T + 1, 10)).mean()
slope = (furnace.SETPOINT_C - furnace.LOAD_C) / furnace.TAU_S * np.exp(-LAG_T / furnace.TAU_S)


# ------------------------------------------------------------------ SVG helpers
def txt(x, y, s, size=20, fill=INK, anchor="middle", weight=400, extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{extra}>{s}</text>'


def path_with_gaps(xs, ys, X, Y):
    d, pen = [], False
    for a, b in zip(xs, ys):
        if np.isnan(b):
            pen = False
            continue
        d.append(f"{'L' if pen else 'M'}{X(a):.1f} {Y(b):.1f}")
        pen = True
    return " ".join(d)


def run_plot(w, h_, small=False):
    """The example run: zone temperature against time in minutes, with gaps where readings are missing."""
    x0, x1 = (20, w - 10) if small else (110, w - 20)
    y0, y1 = (h_ - 20, 15) if small else (h_ - 70, 20)
    lo, hi = 790, 1010
    X = lambda m: x0 + (m / 30) * (x1 - x0)
    Y = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Zone temperature of the example furnace run against time: a warm-up from 800 to 1000 degrees C, then a hold near 1000">']
    if not small:
        for v in (800, 850, 900, 950, 1000):
            s.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{LINE}"/>')
            s.append(txt(x0 - 10, Y(v) + 7, v, 20, MUTED, "end"))
        for m in (0, 5, 10, 15, 20, 25, 30):
            s.append(txt(X(m), y0 + 30, m, 20, MUTED))
        s.append(txt((x0 + x1) / 2, y0 + 62, "time since push (minutes)", 21, MUTED))
        s.append(txt(26, (y0 + y1) / 2, "zone temperature (°C)", 21, MUTED, extra=f' transform="rotate(-90 26 {(y0 + y1) / 2:.1f})"'))
        s.append(f'<rect x="{X(20):.1f}" y="{y1}" width="{X(30) - X(20):.1f}" height="{y0 - y1}" fill="{TINT}"/>')
        s.append(txt((X(20) + X(30)) / 2, Y(930), "hold period", 22, MAROON, weight=600))
        s.append(txt((X(20) + X(30)) / 2, Y(930) + 28, "from 20 minutes on", 19, MUTED))
        s.append(txt(X(7.5), Y(880), "warm-up", 22, MAROON, weight=600))
    s.append(f'<line x1="{x0}" y1="{Y(1000):.1f}" x2="{x1}" y2="{Y(1000):.1f}" stroke="{MUTED}" stroke-dasharray="7 6" stroke-width="2"/>')
    if not small:
        s.append(txt(X(0.3), Y(1000) - 10, "setpoint 1000 °C", 19, MUTED, "start"))
        s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(f'<path d="{path_with_gaps(t / 60, z, X, Y)}" fill="none" stroke="{INK if small else MAROON}" stroke-width="{3 if small else 2.5}"/>')
    if not small:
        for mt in miss_t:
            s.append(f'<line x1="{X(mt / 60):.1f}" y1="{y0}" x2="{X(mt / 60):.1f}" y2="{y0 - 12}" stroke="{MAROON}" stroke-width="3"/>')
        s.append(txt(x1, y0 - 18, "ticks: missing readings", 18, MAROON, "end"))
    s.append("</svg>")
    return "".join(s)


def strip_plot():
    """The eight readings on a line, with the mean and a band of one sample SD either side."""
    w, h_ = 500, 250
    x0, x1, lo, hi = 30, w - 30, 999.4, 1000.6
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    yb = 150
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="The eight readings as dots, the mean at 1000.0 and a band one standard deviation wide on each side">']
    s.append(f'<rect x="{X(N["r_mean"] - N["r_s"]):.1f}" y="40" width="{X(N["r_mean"] + N["r_s"]) - X(N["r_mean"] - N["r_s"]):.1f}" height="{yb - 40}" fill="#F1E3EA"/>')
    s.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    for v in (999.5, 1000.0, 1000.5):
        s.append(f'<line x1="{X(v):.1f}" y1="{yb}" x2="{X(v):.1f}" y2="{yb + 8}" stroke="{INK}" stroke-width="2"/>')
        s.append(txt(X(v), yb + 34, f"{v:.1f}", 20, MUTED))
    s.append(f'<line x1="{X(N["r_mean"]):.1f}" y1="34" x2="{X(N["r_mean"]):.1f}" y2="{yb}" stroke="{MAROON}" stroke-width="3"/>')
    for k, v in enumerate(R):
        s.append(f'<circle cx="{X(v):.1f}" cy="{yb - 26 - 22 * (k % 3)}" r="9" fill="{INK}"/>')
    s.append(txt(X(N["r_mean"]), 26, "mean 1000.0", 20, MAROON, weight=600))
    s.append(txt(X(N["r_mean"] + N["r_s"]) + 6, 62, "± s", 20, MAROON, "start"))
    s.append(txt(w / 2, yb + 72, "reading (°C)", 20, MUTED))
    s.append("</svg>")
    return "".join(s)


def hold_plot():
    """The example run's hold-period readings and their 5-point moving average, against reading number."""
    w, h_ = 640, 400
    x0, x1, y0, y1, lo, hi = 92, w - 14, h_ - 66, 18, 998.6, 1001.6
    X = lambda k: x0 + k / 59 * (x1 - x0)
    Y = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Hold-period readings scattered around 999.6 degrees C and their 5-point moving average, a much smoother line">']
    for v in (999.0, 999.5, 1000.0, 1000.5, 1001.0):
        s.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 8, Y(v) + 7, f"{v:.1f}", 19, MUTED, "end"))
    for k in (0, 10, 20, 30, 40, 50):
        s.append(txt(X(k), y0 + 28, k, 19, MUTED))
    s.append(txt((x0 + x1) / 2, y0 + 56, "reading number in the hold period", 20, MUTED))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    for k, v in enumerate(h):
        s.append(f'<circle cx="{X(k):.1f}" cy="{Y(v):.1f}" r="4.5" fill="{INK}" fill-opacity="0.75"/>')
    pts = " ".join(f"{X(k + 2):.1f},{Y(v):.1f}" for k, v in enumerate(ma))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{MAROON}" stroke-width="4"/>')
    s.append(f'<circle cx="{x0 + 14}" cy="{y1 + 12}" r="5" fill="{INK}"/>')
    s.append(txt(x0 + 26, y1 + 19, "readings", 20, INK, "start"))
    s.append(f'<line x1="{x0 + 150}" y1="{y1 + 12}" x2="{x0 + 176}" y2="{y1 + 12}" stroke="{MAROON}" stroke-width="4"/>')
    s.append(txt(x0 + 184, y1 + 19, "5-point moving average", 20, MAROON, "start"))
    s.append("</svg>")
    return "".join(s)


def box(x, y, w, h_, title, sub, on=True):
    stroke, fill = (MAROON, "#FBF3F7") if on else (LINE, "#FFFFFF")
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h_}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
    out += txt(x + w / 2, y + 38, title, 23, INK, weight=600)
    for k, line in enumerate(sub):
        out += txt(x + w / 2, y + 68 + 26 * k, line, 19, MUTED)
    return out


def arrow(x1, y, x2, mid):
    return f'<line x1="{x1}" y1="{y}" x2="{x2 - 4}" y2="{y}" stroke="{MUTED}" stroke-width="3" marker-end="url(#{mid})"/>'


def arrow_def(mid):
    """Each SVG gets its own marker id: a marker in a hidden slide does not draw in another slide."""
    return f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{MUTED}"/></marker></defs>'


def flow_twin():
    w, h_ = 1130, 260
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Readings arrive, are cleaned, summarized, compared with a model, and lead to a decision">', arrow_def("ah6")]
    items = [("Readings arrive", ["a tool sends", "a stream of values"], True),
             ("Clean", ["missing values,", "units, time stamps"], True),
             ("Summarize", ["mean, SD,", "smoothing, plots"], True),
             ("Compare", ["with a model of", "the process"], False),
             ("Decide", ["a person or a", "controller acts"], False)]
    bw, gap = 190, 40
    for k, (a, b, on) in enumerate(items):
        x = 6 + k * (bw + gap)
        s.append(box(x, 40, bw, 130, a, b, on))
        if k < 4:
            s.append(arrow(x + bw + 4, 105, x + bw + gap, "ah6"))
    s.append(f'<line x1="6" y1="200" x2="{6 + 3 * bw + 2 * gap}" y2="200" stroke="{MAROON}" stroke-width="3"/>')
    s.append(txt(6 + (3 * bw + 2 * gap) / 2, 232, "Module 0: by hand, on one furnace log", 21, MAROON, weight=600))
    s.append(f'<line x1="{6 + 3 * (bw + gap)}" y1="200" x2="{w - 6}" y2="200" stroke="{MUTED}" stroke-width="3"/>')
    s.append(txt(6 + 3 * (bw + gap) + (2 * bw + gap) / 2, 232, "Modules 1 to 7", 21, MUTED, weight=600))
    s.append("</svg>")
    return "".join(s)


def flow_key():
    w, h_ = 1130, 200
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Canvas gives a KEY, the notebook uses it, the check cell tests the functions, and only then prints the verification value">', arrow_def("ah17")]
    items = [("Canvas, Question 1", ["shows your KEY,", "a number 100 to 999"]),
             ("Notebook, Step 6", ["set KEY = your number", "and run the cell"]),
             ("Check cell, Step 7", ["tests your functions", "on hidden cases"]),
             ("Passed?", ["it prints your", "verification value"])]
    bw, gap = 238, 46
    for k, (a, b) in enumerate(items):
        x = 6 + k * (bw + gap)
        s.append(box(x, 30, bw, 140, a, b, k in (0, 3)))
        if k < 3:
            s.append(arrow(x + bw + 4, 100, x + bw + gap, "ah17"))
    s.append("</svg>")
    return "".join(s)


def furnace_svg():
    """A horizontal tube furnace: quartz tube, heater zones, a boat of wafers and a thermocouple."""
    w, h_ = 560, 330
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Sketch of a horizontal tube furnace with three heater zones, a quartz tube, a boat of wafers and a thermocouple">']
    s.append(f'<rect x="40" y="60" width="480" height="170" rx="16" fill="{TINT}" stroke="{INK}" stroke-width="3"/>')
    for k, lab in enumerate(("zone 1", "zone 2", "zone 3")):
        x = 60 + k * 150
        s.append(f'<rect x="{x}" y="70" width="140" height="26" rx="4" fill="#F1E3EA" stroke="{MAROON}" stroke-width="2"/>')
        s.append(f'<rect x="{x}" y="194" width="140" height="26" rx="4" fill="#F1E3EA" stroke="{MAROON}" stroke-width="2"/>')
        s.append(txt(x + 70, 90, lab, 18, MAROON))
    s.append(f'<rect x="20" y="108" width="520" height="74" rx="37" fill="#FFFFFF" stroke="{MUTED}" stroke-width="3"/>')
    s.append(f'<rect x="170" y="166" width="220" height="10" fill="{INK}"/>')
    for k in range(13):
        x = 182 + k * 16
        s.append(f'<line x1="{x}" y1="120" x2="{x}" y2="166" stroke="{INK}" stroke-width="5"/>')
    s.append(f'<line x1="540" y1="140" x2="300" y2="140" stroke="{MAROON}" stroke-width="4"/>')
    s.append(f'<circle cx="300" cy="140" r="7" fill="{MAROON}"/>')
    s.append(txt(280, 270, "quartz tube with a boat of silicon wafers", 20, INK))
    s.append(txt(280, 300, "heater zones above and below", 20, MUTED))
    s.append(txt(480, 40, "thermocouple", 20, MAROON, "middle", 600))
    s.append(f'<line x1="480" y1="48" x2="470" y2="134" stroke="{MAROON}" stroke-width="2"/>')
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ the slides
FOOT = "Module 0 · Start here"
slides = []


def slide(body, cls="slide"):
    n = len(slides) + 1
    foot = ('<div class="foot"><span>SemiAcademy · ISEP-CSU</span><span>CC BY 4.0</span></div>' if n == 1
            else f'<div class="foot"><span>{FOOT}</span><span>{n}</span></div>')
    slides.append(f'<!-- {n} -->\n<section class="{cls}">\n{body}\n  {foot}\n</section>\n')


def reveal(ans):
    return f'  <div class="reveal"><button type="button">Show the answer</button>\n    <div class="ans">{ans}</div></div>'


slide(f"""  <div class="kicker">Digital Twins for Semiconductor Manufacturing · Module 0</div>
  <h1>Start here: Python on a furnace log</h1>
  <p style="font-size:30px;max-width:820px">How the course works, and the notebook, NumPy and pandas skills that every later lab uses.</p>
  <div style="margin-top:14px">{run_plot(620, 230, small=True)}</div>
  <p class="small" style="margin-top:4px">A simulated furnace run: the zone warms from 800 °C and holds near 1000 °C.</p>""", "slide title-slide")

slide("""  <h2>By the end of this module you can</h2>
  <ul>
    <li>run a <strong>browser notebook</strong>, keep a copy of your work and recover from an error;</li>
    <li>summarize readings with a mean and a <strong>sample standard deviation</strong>, leaving missing values out;</li>
    <li>load a furnace log into a <strong>pandas</strong> table, select rows and plot the run with units;</li>
    <li>smooth readings with a <strong>moving average</strong>, and find and fix a bug in a function.</li>
  </ul>
  <p class="small" style="margin-top:24px">These are the entry skills for Modules 1 to 7. The readiness check shows which ones you already have; the optional prep pack covers any you miss.</p>""")

rows = [("0", "Orientation, readiness check and Python warm-up", "1.75"), ("1", "What a digital twin is", "2.5"),
        ("2", "The fab as a data system", "3"), ("3", "Physics model: thermal oxidation", "3"),
        ("4", "Virtual metrology for oxide thickness", "3"), ("5", "Keeping the twin in sync", "3"),
        ("6", "Toolset twin: queues and maintenance", "3"), ("7", "Capstone: integrate, validate and document", "3.5")]
assert abs(sum(float(r[2]) for r in rows) - 22.75) < 1e-9
tr = "".join(f'<tr><td style="text-align:center">{a}</td><td>{b}</td><td style="text-align:right">{c}</td></tr>' for a, b, c in rows)
slide(f"""  <h2>Eight modules, one furnace twin</h2>
  <table style="font-size:21px;width:100%">
    <tr><th style="width:110px;text-align:center">Module</th><th>Topic</th><th style="width:120px;text-align:right">Hours</th></tr>
    {tr}
    <tr><td></td><td><strong>Total</strong></td><td style="text-align:right"><strong>22.75</strong></td></tr>
  </table>
  <p class="small" style="margin-top:12px">Modules 1 to 5 build the twin of one furnace layer by layer; Module 6 places the furnace in a three-tool line; Module 7 joins the pieces.</p>""")

slide("""  <h2>How the course and the badge work</h2>
  <div class="cols">
    <div class="card"><h3>Each module</h3><p>Slides, short sections with Watch and Try it panels, a browser lab, a lab check (10 points) and, from Module 1 on, a quiz (20 points).</p></div>
    <div class="card"><h3>Unlocking</h3><p>The Module 0 lab check at 8/10 opens Module 1. In Modules 1 to 6, the lab check at 8/10 and the quiz at 16/20 open the next module.</p></div>
    <div class="card"><h3>The badge</h3><p>At least 207 of 230 points, every minimum met, and the post-course survey submitted.</p></div>
  </div>
  <p class="small">Every graded item is marked automatically in Canvas. Attempts are unlimited and your highest score counts.</p>""")

slide(f"""  <h2>One furnace for the whole course</h2>
  <div class="cols">
    <div style="flex:0 0 560px">{furnace_svg()}</div>
    <div>
      <p><strong>Thermal oxidation</strong> grows a silicon dioxide layer on silicon wafers at about 800 to 1200 °C.</p>
      <p>A <strong>thermocouple</strong> reports each zone's temperature every few seconds.</p>
      <p>The course furnace is <strong>simulated</strong>: you get your own data, and every result can be checked.</p>
    </div>
  </div>
  <p class="small">Module 0 uses one zone-temperature log. Later modules add oxide growth, metrology, drift and a three-tool line.</p>""")

slide(f"""  <h2>What a twin needs from data</h2>
  <div style="margin-top:10px">{flow_twin()}</div>
  <p style="margin-top:26px">A twin is only as good as the data that reach it. Module 0 practises the first three steps by hand; the rest of the course adds the model and the decision.</p>""")

slide("""  <h2>Notebooks in your browser</h2>
  <div class="cols">
    <div>
      <ul>
        <li><strong>JupyterLite</strong> runs Python inside your browser: nothing to install, no account.</li>
        <li>The first start can take up to a minute.</li>
        <li>Run the cells <strong>in order</strong> with Shift + Enter.</li>
        <li>Each notebook shows its version near the top.</li>
      </ul>
    </div>
    <div class="card"><h3>If something goes wrong</h3>
      <p style="margin-bottom:10px">A cell fails: read the <strong>last line</strong> of the error first.</p>
      <p style="margin-bottom:10px">Strange results: Kernel &gt; Restart Kernel and Run All Cells.</p>
      <p>Keep a copy: File &gt; Download. Clearing your browser's site data deletes work saved in the browser.</p></div>
  </div>
  <p class="small">The Set Up Your Tools page in Canvas covers browsers, saving your work and an optional Google Colab copy of each lab.</p>""")

slide(f"""  <h2>NumPy arrays and the standard deviation</h2>
  <div class="cols">
    <div style="flex:0 0 540px">
<pre class="code">readings = np.array([999.6, 1000.4, 1000.1,
    999.8, 1000.3, 999.9, 1000.2, 999.7])
readings[0]           # 999.6
readings + 273.15     # every element
readings.mean()       # {N['r_mean']:.1f}
readings.std(ddof=1)  # {N['r_s']:.3f}  sample
readings.std()        # {N['r_p']:.3f}  population</pre>
    </div>
    <div>{strip_plot()}
      <div class="eq" style="font-size:28px;margin-top:6px">s = √( Σ (x<sub>i</sub> − x̄)² / (n − 1) )</div>
    </div>
  </div>
  <p class="small">Python counts positions from 0. The sample SD divides by n − 1; NumPy's default divides by n.</p>""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>Sample or population: does it matter?</h2>
  <p>For the 8 readings, the sample SD (divide by n − 1) is {N['r_s']:.3f} °C and the population SD (divide by n) is {N['r_p']:.3f} °C: a gap of {100 * (N['r_s'] / N['r_p'] - 1):.0f}%.</p>
  <p><strong>Predict:</strong> with 60 readings from the same hold, does the gap grow, shrink or stay the same?</p>
{reveal(f"It shrinks. The ratio of the two is √(n / (n − 1)): {np.sqrt(8 / 7):.3f} for n = 8, but {np.sqrt(60 / 59):.3f} for n = 60, under 1%. The choice matters most for small samples. The course uses the sample SD (ddof=1) for readings.")}""")

head = ex.head(3)
trs = "".join(f"<tr><td>{r.time_s}</td><td>{r.run_id}</td><td>{r.setpoint_C:.1f}</td><td>{r.zone_temp_C:.2f}</td></tr>" for r in head.itertuples())
trs += '<tr><td colspan="4" style="text-align:center;color:#6A626B">…</td></tr>'
for r in ex.iloc[36:38].itertuples():
    v = "NaN" if np.isnan(r.zone_temp_C) else f"{r.zone_temp_C:.2f}"
    sty = ' style="color:#99004C;font-weight:600"' if v == "NaN" else ""
    trs += f'<tr><td>{r.time_s}</td><td>{r.run_id}</td><td>{r.setpoint_C:.1f}</td><td{sty}>{v}</td></tr>'
slide(f"""  <h2>pandas: a furnace log as a table</h2>
  <div class="cols">
    <div style="flex:0 0 500px">
      <table style="font-size:21px">
        <tr><th>time_s</th><th>run_id</th><th>setpoint_C</th><th>zone_temp_C</th></tr>
        {trs}
      </table>
    </div>
    <div>
<pre class="code">log = furnace_log(2026)   # {N['rows']} rows
log["zone_temp_C"]        # one column
log[log["time_s"] &gt;= 1200]  # some rows
log["zone_temp_C"].isna().sum()  # {N['nmiss']}</pre>
    </div>
  </div>
  <p class="small">NaN means "not a number": the logger missed a reading. The example log misses the readings at {", ".join(str(v) for v in miss_t[:-1])} and {miss_t[-1]} s.</p>""")

slide(f"""  <h2>Missing values: leave them out, never fill with zero</h2>
  <table style="font-size:23px">
    <tr><th>Hold period of the example run ({N['hold_rows']} rows, 1 missing)</th><th style="text-align:right">Mean (°C)</th><th style="text-align:right">Sample SD (°C)</th></tr>
    <tr><td>Leave the missing reading out ({N['hold_n']} readings)</td><td style="text-align:right">{N['hold_mean']:.3f}</td><td style="text-align:right">{N['hold_sd']:.3f}</td></tr>
    <tr><td>Repeat the previous reading (forward fill)</td><td style="text-align:right">{N['ff_mean']:.3f}</td><td style="text-align:right">{N['ff_sd']:.3f}</td></tr>
    <tr><td style="color:#99004C">Fill it with 0</td><td style="text-align:right;color:#99004C">{N['zero_mean']:.2f}</td><td style="text-align:right;color:#99004C">{N['zero_sd']:.2f}</td></tr>
  </table>
  <p style="margin-top:18px">One zero among {N['hold_rows']} readings pulls the mean down by about 1000 / {N['hold_rows']} ≈ {1000 / N['hold_rows']:.0f} °C and makes the SD meaningless.</p>
  <p class="small">Good habits: count the missing values, say how you handled them, and never let a filled value pass for a measured one.</p>""")

slide(f"""  <h2>Plot the run, with units</h2>
  <div>{run_plot(1130, 470)}</div>
  <p class="small" style="margin-top:2px">Every plot in the course has axis labels with units and says when the data are simulated. Gaps in the line are missing readings.</p>""")

slide(f"""  <h2>Moving average: smoothing the hold</h2>
  <div class="cols">
    <div style="flex:0 0 640px">{hold_plot()}</div>
    <div>
      <div class="eq" style="font-size:24px;margin-top:0">y<sub>j</sub> = (x<sub>j</sub> + x<sub>j+1</sub> + … + x<sub>j+w−1</sub>) / w</div>
      <ul style="padding-left:28px">
        <li style="font-size:24px"><strong>n</strong> readings give <strong>n − w + 1</strong> averages: {N['hold_n']} → {N['ma_n']} for w = 5.</li>
        <li style="font-size:24px">Independent noise: the SD falls by about <strong>√w</strong>.</li>
        <li style="font-size:24px">The price: the average <strong>lags</strong> behind a changing signal.</li>
      </ul>
    </div>
  </div>""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>How much does a 5-point average smooth?</h2>
  <p>The example hold readings have a sample SD of {N['hold_sd']:.3f} °C.</p>
  <p><strong>Predict:</strong> the SD of their 5-point moving average.</p>
{reveal(f"About {N['hold_sd']:.3f} / √5 ≈ {N['hold_sd'] / np.sqrt(5):.2f} °C. The notebook gives {N['ma_sd']:.3f} °C, a ratio of {N['ratio']:.2f} rather than {np.sqrt(5):.2f}. An SD estimated from {N['ma_n']} overlapping averages has its own scatter, so expect close, not exact.")}""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>The price of smoothing</h2>
  <p>A twin running in real time has only past readings, so it uses a <strong>trailing</strong> average: at {LAG_T} s, the 5-point average uses the readings from {LAG_T - 40} to {LAG_T} s, centred 20 s in the past.</p>
  <p>At {LAG_T} s the zone is warming by about {slope:.1f} °C per second. <strong>Predict:</strong> how far below the current temperature does the average read?</p>
{reveal(f"About {slope:.1f} × 20 ≈ {slope * 20:.0f} °C. Without noise the exact gap is {lag_err:.1f} °C: the warm-up was steeper in the past. In the hold the level is flat, so the lag costs nothing there; Module 5 meets the same trade-off in the EWMA filter.")}""")

slide(f"""  <h2>Debugging: the planted bug</h2>
  <div class="cols">
    <div style="flex:0 0 600px">
<pre class="code">def moving_average(x, window):
    kernel = np.ones(window)
    return np.convolve(np.asarray(x, dtype=float),
                       kernel, mode="valid")

moving_average(hold_readings, 5)[0]
# {N['bug0']:.2f}</pre>
    </div>
    <div>
      <p style="font-size:25px">Look at the numbers: is {N['bug0']:,.0f} °C possible for a zone held at 1000 °C?</p>
      <p style="font-size:25px">Test a case you can do by hand: [1, 2, 3, 4, 5, 6] with w = 3 should give [2, 3, 4, 5].</p>
      <p style="font-size:25px">Change one thing, then rerun the cell and every cell that uses it.</p>
    </div>
  </div>
  <p class="small">In the lab you find and fix this bug. The check cell tests the repaired function on cases you cannot see.</p>""")

slide(f"""  <h2>Your KEY and the verification value</h2>
  <div>{flow_key()}</div>
  <ul style="margin-top:14px">
    <li>Question 1, the verification value, is worth <strong>4 of the 10 points</strong>; Questions 2 to 5 are worth 1.5 each.</li>
    <li>8 of 10 opens Module 1, so working code is required.</li>
    <li>A new attempt gives a new KEY: run Steps 6 and 7 again with it.</li>
  </ul>""")

slide("""  <h2>Summary</h2>
  <ul>
    <li>A twin starts with data: readings that are cleaned and summarized.</li>
    <li>Mean and <strong>sample</strong> SD (ddof=1), with missing values left out, never set to zero.</li>
    <li>pandas picks columns and rows; every plot carries units.</li>
    <li>A w-point moving average divides independent noise by about √w, and lags a changing signal.</li>
  </ul>
  <div class="eq" style="font-size:26px;margin-top:20px"><strong>Next:</strong> the Module 0 lab. You write <code>mean_and_sd()</code>, fix <code>moving_average()</code>, and analyse your own furnace run.</div>""")

slide("""  <h2>Credits and license</h2>
  <p>These slides were written for SemiAcademy's <em>Digital Twins for Semiconductor Manufacturing</em> course, developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU).</p>
  <p>License: <strong>Creative Commons Attribution 4.0 International (CC BY 4.0)</strong>, https://creativecommons.org/licenses/by/4.0/. You may copy, adapt and share them, with credit.</p>
  <p>Suggested credit: "Module 0 slides, Digital Twins for Semiconductor Manufacturing, SemiAcademy / ISEP-CSU, CC BY 4.0."</p>
  <p class="small">Every data value on these slides is simulated with the course's furnace model (furnace.py, example run 2026). Source: github.com/CSU-Physics/digital-twins-notebooks (media/m0/m0-slides.html). Print or save as PDF from your browser to get one slide per page.</p>""")

# ------------------------------------------------------------------ assemble
head_html = (T / "deck_head.html").read_text()
head_html = head_html.replace("<title>Module 1 Slides: From Bits to Qubits</title>", "<title>Module 0 Slides: Start Here, Python on a Furnace Log</title>")
assert "Module 0 Slides" in head_html
extra_css = ("  .code { font-family: Consolas, \"Courier New\", monospace; font-size:21px; line-height:1.4; background:var(--tint); "
             "border-left:6px solid var(--ink); padding:14px 18px; margin:0 0 12px; white-space:pre; overflow:hidden; }\n"
             "  code { font-family: Consolas, \"Courier New\", monospace; font-size:0.92em; }\n</style>")
head_html = head_html.replace("</style>", extra_css, 1)
bar = (T / "deck_bar.html").read_text().replace("1 / 19", f"1 / {len(slides)}")
ctrl = (T / "deck_ctrl.js").read_text()
html = head_html + "\n" + "\n".join(slides) + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
for bad in ("—", "${", "`"):
    assert bad not in html, bad
(OUT / "m0-slides.html").write_text(html)
print("wrote", OUT / "m0-slides.html", len(slides), "slides,", len(html), "bytes")
print({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in N.items()})
print("lag", round(float(lag_err), 2), "slope", round(float(slope), 3))
