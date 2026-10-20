"""Build the Module 1 slide deck (media/m1/m1-slides.html) for Digital Twins for Semiconductor Manufacturing.

Every number and curve on the slides is computed here from the course's furnace model (content/m1/furnace.py),
the lab helpers (content/m1/twin_tools.py) and the reference update of the Module 1 check, so the slides agree
with the lab notebook. Run from anywhere: python3 tools/build_dt_m1_media.py
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m1"), str(ROOT / "checks")]
from furnace import LOAD_C, SETPOINT_C, temperature_stream, stream_truth  # noqa: E402
from twin_tools import first_order, fit_tau, rms, run_twin  # noqa: E402
from dt_m1_check import reference_update  # noqa: E402

T = ROOT / "tools"
OUT = ROOT / "media" / "m1"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, INK, MUTED, LINE, TINT = "#99004C", "#24313D", "#6A626B", "#E4DCE0", "#F8F4F6"

# ------------------------------------------------------------------ numbers used on the slides
st = temperature_stream(2026)
truth = stream_truth(2026)
warm = st[st["time_s"] < 600]
TAU, TAU_SD = fit_tau(warm["time_s"], warm["zone_temp_C"])
model = run_twin(st, TAU, 0.5, None)
synced = run_twin(st, TAU, 0.5, reference_update)
late = model["time_s"] >= 600
N = dict(tau=TAU, sd=TAU_SD, tau_true=truth["tau_s"], t_dist=truth["disturbance_start_s"],
         ol_early=rms(model["error"][~late]), ol_late=rms(model["error"][late]),
         sy_early=rms(synced["error"][~late]), sy_late=rms(synced["error"][late]),
         ol_max=float(np.nanmax(np.abs(model["error"]))), sy_max=float(np.nanmax(np.abs(synced["error"]))))
EVERY = [(e, rms(run_twin(st, TAU, 0.5, reference_update, every=e)["error"][late])) for e in (1, 3, 6, 18)]
GAINS = [(g, rms(run_twin(st, TAU, g, reference_update)["error"][~late]), rms(run_twin(st, TAU, g, reference_update)["error"][late]))
         for g in (0.0, 0.2, 0.5, 0.8, 1.0)]
assert f"{TAU:.2f}" == "130.76" and f"{N['ol_late']:.2f}" == "5.61" and f"{N['sy_late']:.2f}" == "1.29"
T_RULE = 150 * math.log(20)        # time to come within 10 C of the setpoint from a 200 C gap, tau = 150 s


# ------------------------------------------------------------------ SVG helpers (as in the Module 0 deck)
def txt(x, y, s, size=20, fill=INK, anchor="middle", weight=400, extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{extra}>{s}</text>'


def box(x, y, w, h_, title, sub, on=True, title_size=23):
    stroke, fill = (MAROON, "#FBF3F7") if on else (LINE, "#FFFFFF")
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h_}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
    out += txt(x + w / 2, y + 36, title, title_size, INK, weight=600)
    for k, line in enumerate(sub):
        out += txt(x + w / 2, y + 64 + 25 * k, line, 19, MUTED)
    return out


def arrow_def(mid, color=MUTED):
    return (f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0 0L10 5L0 10z" fill="{color}"/></marker></defs>')


def line(x1, y1, x2, y2, mid, color=MUTED, dash="", width=3, both=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    start = f' marker-start="url(#{mid})"' if both else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d} marker-end="url(#{mid})"{start}/>'


def axes(s, X, Y, x0, x1, y0, y1, xt, yt, xl, yl, yfmt="{:.0f}"):
    for v in yt:
        s.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y(v) + 7, yfmt.format(v), 19, MUTED, "end"))
    for v in xt:
        s.append(txt(X(v), y0 + 28, f"{v:g}", 19, MUTED))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(txt((x0 + x1) / 2, y0 + 56, xl, 20, MUTED))
    s.append(txt(22, (y0 + y1) / 2, yl, 20, MUTED, extra=f' transform="rotate(-90 22 {(y0 + y1) / 2:.1f})"'))


def path_gaps(xs, ys, X, Y):
    d, pen = [], False
    for a, b in zip(xs, ys):
        if np.isnan(b):
            pen = False
            continue
        d.append(f"{'L' if pen else 'M'}{X(a):.1f} {Y(b):.1f}")
        pen = True
    return " ".join(d)


# ------------------------------------------------------------------ figures
def title_fig():
    w, h_ = 640, 220
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" aria-hidden="true">', arrow_def("ah1", MAROON)]
    s.append(box(10, 50, 200, 110, "Furnace", ["physical entity"], on=False))
    s.append(box(430, 50, 200, 110, "Twin", ["virtual entity"], on=True))
    s.append(line(220, 88, 420, 88, "ah1", MAROON))
    s.append(line(420, 128, 220, 128, "ah1", MUTED, dash="7 6"))
    s.append(txt(320, 76, "readings, every 10 s", 18, MAROON))
    s.append(txt(320, 154, "decisions (later modules)", 18, MUTED))
    s.append("</svg>")
    return "".join(s)


def parts_fig():
    w, h_ = 1130, 330
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="A physical entity and a virtual entity joined by a two-way connection: data in, decisions out">',
         arrow_def("ah4", MAROON), arrow_def("ah4b", MUTED)]
    s.append(box(10, 60, 300, 170, "Physical entity", ["the oxidation furnace,", "its wafers and sensors"], on=False))
    s.append(box(820, 60, 300, 170, "Virtual entity", ["models of the furnace", "and the data that drive them"], on=True))
    s.append(line(320, 110, 810, 110, "ah4", MAROON))
    s.append(txt(565, 96, "data: sensor readings, metrology, run history", 20, MAROON, weight=600))
    s.append(line(810, 185, 320, 185, "ah4b", MUTED))
    s.append(txt(565, 216, "information and decisions: alerts, recipe changes", 20, MUTED, weight=600))
    s.append(txt(565, 290, "The connection: what flows, in which direction, how often and how automatically", 22, INK, weight=600))
    s.append("</svg>")
    return "".join(s)


def taxonomy_fig():
    w, h_ = 1130, 360
    rows = [("Digital model", "manual", "manual", False), ("Digital shadow", "automatic", "manual", True), ("Digital twin", "automatic", "automatic", True)]
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Digital model: manual data flow both ways. Digital shadow: automatic from physical to digital, manual back. Digital twin: automatic both ways.">',
         arrow_def("t1", MAROON), arrow_def("t2", MUTED)]
    for k, (name, inn, out, _) in enumerate(rows):
        y = 20 + k * 115
        s.append(txt(10, y + 58, name, 24, MAROON if k else INK, "start", 600))
        s.append(f'<rect x="270" y="{y + 15}" width="170" height="72" rx="10" fill="#FFFFFF" stroke="{LINE}" stroke-width="2.5"/>')
        s.append(txt(355, y + 59, "physical", 21, INK))
        s.append(f'<rect x="900" y="{y + 15}" width="170" height="72" rx="10" fill="#FBF3F7" stroke="{MAROON}" stroke-width="2.5"/>')
        s.append(txt(985, y + 59, "digital", 21, INK))
        s.append(line(450, y + 38, 890, y + 38, "t1" if inn == "automatic" else "t2", MAROON if inn == "automatic" else MUTED,
                      "" if inn == "automatic" else "8 7"))
        s.append(txt(670, y + 28, f"{inn} data flow in", 19, MAROON if inn == "automatic" else MUTED))
        s.append(line(890, y + 70, 450, y + 70, "t1" if out == "automatic" else "t2", MAROON if out == "automatic" else MUTED,
                      "" if out == "automatic" else "8 7"))
        s.append(txt(670, y + 98, f"{out} data flow back", 19, MAROON if out == "automatic" else MUTED))
    s.append("</svg>")
    return "".join(s)


def iso_fig():
    w, h_ = 1130, 300
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="ISO 23247 reference model: observable manufacturing elements, data collection and device control, the digital twin core, and the user domain">',
         arrow_def("iso", MUTED)]
    items = [("Observable", ["manufacturing elements", "(furnace, wafers)"]),
             ("Data collection", ["and device control:", "sensors, tool interface"]),
             ("Digital twin core", ["models, sync, analytics,", "simulation"]),
             ("User", ["engineers, MES,", "other applications"])]
    bw, gap = 238, 50
    for k, (a, b) in enumerate(items):
        x = 6 + k * (bw + gap)
        s.append(box(x, 40, bw, 150, a, b, on=(k == 2), title_size=22))
        if k < 3:
            s.append(line(x + bw + 4, 100, x + bw + gap - 4, 100, "iso", MUTED, both=True))
    s.append(txt(565, 250, "ISO 23247 (2021): a framework for digital twins in manufacturing, in four parts", 21, INK, weight=600))
    s.append("</svg>")
    return "".join(s)


def first_order_fig():
    w, h_ = 620, 380
    x0, x1, y0, y1 = 92, w - 16, h_ - 70, 20
    X = lambda t: x0 + t / 900 * (x1 - x0)
    Y = lambda v: y0 - (v - 790) / (1010 - 790) * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="First-order warm-up from 800 toward 1000 degrees C with tau = 150 s: 63 percent at one tau, 95 percent at three tau">']
    axes(s, X, Y, x0, x1, y0, y1, [0, 150, 300, 450, 600, 750, 900], [800, 850, 900, 950, 1000], "time since push (s)", "zone temperature (°C)")
    tt = np.linspace(0, 900, 181)
    s.append(f'<polyline points="{" ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(tt, first_order(tt, 150.0)))}" fill="none" stroke="{MAROON}" stroke-width="4"/>')
    s.append(f'<line x1="{x0}" y1="{Y(1000):.1f}" x2="{x1}" y2="{Y(1000):.1f}" stroke="{MUTED}" stroke-dasharray="7 6" stroke-width="2"/>')
    for t, lab in ((150, "τ: 63%"), (450, "3τ: 95%")):
        v = float(first_order(t, 150.0))
        s.append(f'<line x1="{X(t):.1f}" y1="{y0}" x2="{X(t):.1f}" y2="{Y(v):.1f}" stroke="{INK}" stroke-dasharray="5 5" stroke-width="2"/>')
        s.append(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="7" fill="{INK}"/>')
        s.append(txt(X(t) + 12, Y(v) + 30, f"{lab}, {v:.1f} °C", 19, INK, "start"))
    s.append("</svg>")
    return "".join(s)


def fit_fig():
    w, h_ = 620, 380
    x0, x1, y0, y1 = 92, w - 16, h_ - 70, 20
    X = lambda t: x0 + t / 600 * (x1 - x0)
    Y = lambda v: y0 - (v - 790) / (1010 - 790) * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="The first ten minutes of the example stream with the fitted first-order curve">']
    axes(s, X, Y, x0, x1, y0, y1, [0, 100, 200, 300, 400, 500, 600], [800, 850, 900, 950, 1000], "time since push (s)", "zone temperature (°C)")
    for t, v in zip(warm["time_s"], warm["zone_temp_C"]):
        if not np.isnan(v):
            s.append(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="4.5" fill="{INK}" fill-opacity="0.8"/>')
    tt = np.linspace(0, 600, 121)
    s.append(f'<polyline points="{" ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(tt, first_order(tt, TAU)))}" fill="none" stroke="{MAROON}" stroke-width="3.5"/>')
    s.append(txt(x1, Y(880), f"fitted τ = {TAU:.2f} s", 21, MAROON, "end", 600))
    s.append(txt(x1, Y(880) + 28, f"standard error {TAU_SD:.2f} s", 19, MUTED, "end"))
    s.append("</svg>")
    return "".join(s)


def dip_fig(which="model"):
    w, h_ = 1130, 400
    x0, x1, y0, y1 = 96, w - 20, h_ - 70, 20
    X = lambda t: x0 + (t - 420) / 1380 * (x1 - x0)
    Y = lambda v: y0 - (v - 982) / (1004 - 982) * (y0 - y1)
    run = model if which == "model" else synced
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="Readings from 5 to 30 minutes with a dip after 15 minutes, and the {"model alone" if which == "model" else "synced twin"}">']
    axes(s, X, Y, x0, x1, y0, y1, [600, 900, 1200, 1500, 1800], [985, 990, 995, 1000], "time since push (s)", "zone temperature (°C)")
    sel = st["time_s"] >= 420
    for t, v in zip(st["time_s"][sel], st["zone_temp_C"][sel]):
        if not np.isnan(v):
            s.append(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="4.5" fill="{INK}" fill-opacity="0.75"/>')
    rs = run["time_s"] >= 420
    s.append(f'<path d="{path_gaps(run["time_s"][rs], run["predicted"][rs], X, Y)}" fill="none" stroke="{MAROON}" stroke-width="3.5"/>')
    s.append(f'<line x1="{X(600):.1f}" y1="{y1}" x2="{X(600):.1f}" y2="{y0}" stroke="{MUTED}" stroke-dasharray="6 6" stroke-width="2"/>')
    s.append(txt(X(600) - 10, y1 + 22, "fit window ends", 19, MUTED, "end"))
    lab = "model alone" if which == "model" else "synced twin (gain 0.5)"
    s.append(f'<circle cx="{x1 - 360}" cy="{y0 - 36}" r="5" fill="{INK}"/>')
    s.append(txt(x1 - 346, y0 - 29, "readings", 20, INK, "start"))
    s.append(f'<line x1="{x1 - 230}" y1="{y0 - 36}" x2="{x1 - 200}" y2="{y0 - 36}" stroke="{MAROON}" stroke-width="4"/>')
    s.append(txt(x1 - 192, y0 - 29, lab, 20, MAROON, "start"))
    s.append("</svg>")
    return "".join(s)


def cycle_fig():
    w, h_ = 560, 340
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="The twin cycle: predict the next reading with the model, compare with the reading, correct the state, repeat">',
         arrow_def("cy", MAROON)]
    s.append(box(170, 10, 220, 90, "1. Predict", ["with the model"], on=True))
    s.append(box(330, 200, 220, 90, "2. Compare", ["with the reading"], on=False))
    s.append(box(10, 200, 220, 90, "3. Correct", ["T + gain · error"], on=False))
    s.append(f'<path d="M395 70 C 470 80 470 150 450 195" fill="none" stroke="{MAROON}" stroke-width="3" marker-end="url(#cy)"/>')
    s.append(f'<path d="M325 250 L 240 250" fill="none" stroke="{MAROON}" stroke-width="3" marker-end="url(#cy)"/>')
    s.append(f'<path d="M110 195 C 90 150 100 80 165 70" fill="none" stroke="{MAROON}" stroke-width="3" marker-end="url(#cy)"/>')
    s.append(txt(280, 330, "every 10 s, as each reading arrives", 19, MUTED))
    s.append("</svg>")
    return "".join(s)


def every_fig():
    w, h_ = 620, 380
    x0, x1, y0, y1 = 92, w - 16, h_ - 70, 30
    vals = [v for _, v in EVERY]
    top = math.ceil(max(vals) + 0.5)
    Y = lambda v: y0 - v / top * (y0 - y1)
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 {w} {h_}" role="img" aria-label="RMS error from 600 s on for updates every 10, 30, 60 and 180 s">']
    for v in range(0, top + 1):
        s.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{LINE}"/>')
        s.append(txt(x0 - 10, Y(v) + 7, v, 19, MUTED, "end"))
    bw = (x1 - x0) / len(EVERY)
    for k, (e, v) in enumerate(EVERY):
        cx = x0 + bw * (k + 0.5)
        s.append(f'<rect x="{cx - 48:.1f}" y="{Y(v):.1f}" width="96" height="{y0 - Y(v):.1f}" fill="{MAROON}" rx="3"/>')
        s.append(txt(cx, Y(v) - 10, f"{v:.2f}", 20, INK, weight=600))
        s.append(txt(cx, y0 + 28, f"{e * 10} s", 20, MUTED))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" stroke-width="2"/>')
    s.append(f'<line x1="{x0}" y1="{Y(N["ol_late"]):.1f}" x2="{x1}" y2="{Y(N["ol_late"]):.1f}" stroke="{INK}" stroke-dasharray="7 6" stroke-width="2"/>')
    s.append(txt(x1, Y(N["ol_late"]) - 10, f"model alone {N['ol_late']:.2f}", 19, INK, "end"))
    s.append(txt((x0 + x1) / 2, y0 + 56, "time between updates", 20, MUTED))
    s.append(txt(22, (y0 + y1) / 2, "RMS error from 600 s (°C)", 20, MUTED, extra=f' transform="rotate(-90 22 {(y0 + y1) / 2:.1f})"'))
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ slides
FOOT = "Module 1 · What a digital twin is"
slides = []


def slide(body, cls="slide"):
    n = len(slides) + 1
    foot = ('<div class="foot"><span>SemiAcademy · ISEP-CSU</span><span>CC BY 4.0</span></div>' if n == 1
            else f'<div class="foot"><span>{FOOT}</span><span>{n}</span></div>')
    slides.append(f'<!-- {n} -->\n<section class="{cls}">\n{body}\n  {foot}\n</section>\n')


def reveal(ans):
    return f'  <div class="reveal"><button type="button">Show the answer</button>\n    <div class="ans">{ans}</div></div>'


slide(f"""  <div class="kicker">Digital Twins for Semiconductor Manufacturing · Module 1</div>
  <h1>What a digital twin is</h1>
  <p style="font-size:30px;max-width:840px">Definitions, levels of coupling, and a first twin of the furnace that keeps in step with its readings.</p>
  <div style="margin-top:20px">{title_fig()}</div>""", "slide title-slide")

slide("""  <h2>By the end of this module you can</h2>
  <ul>
    <li>define a digital twin with the <strong>Digital Twin Consortium</strong> and <strong>NIST</strong> definitions;</li>
    <li>place a system on the <strong>model, shadow and twin</strong> classification;</li>
    <li>tell three <strong>levels of coupling</strong> apart: updating from observations, supporting a decision, acting automatically;</li>
    <li>explain what <strong>fidelity</strong> and <strong>update frequency</strong> mean for a twin;</li>
    <li>fit the time constant of a first-order model and keep the model in step with readings.</li>
  </ul>
  <p class="small" style="margin-top:20px">This is course outcome 1. The lab builds the first layer of the course's furnace twin.</p>""")

slide("""  <h2>Two definitions</h2>
  <div class="cols">
    <div class="card"><h3>Digital Twin Consortium</h3><p>A virtual representation of real-world entities and processes, driven by data, with &ldquo;synchronized interaction at a specified frequency and fidelity.&rdquo;</p>
      <p class="small" style="margin-top:12px">digitaltwinconsortium.org, definition first published in 2020</p></div>
    <div class="card"><h3>NIST IR 8356 (2025)</h3><p>&ldquo;A digital twin is the virtual (i.e., digital) representation of a physical or perceived real-world entity, concept, or notion.&rdquo;</p>
      <p class="small" style="margin-top:12px">Voas, Mell, Laplante and Piroumian, <em>Security and Trust Considerations for Digital Twin Technology</em></p></div>
  </div>
  <p class="small">Both start from a virtual representation. The Consortium adds what makes it a twin rather than a model: synchronization, at a frequency and fidelity chosen for the use.</p>""")

slide(f"""  <h2>Three parts of every twin</h2>
  <div style="margin-top:6px">{parts_fig()}</div>
  <p class="small" style="margin-top:12px">The Consortium allows the connection to be one-way or two-way, and notes that the most common mechanisms still keep a person in the loop.</p>""")

slide(f"""  <h2>Model, shadow or twin: classified by data flow</h2>
  <div>{taxonomy_fig()}</div>
  <p class="small">Kritzinger, Karner, Traar, Henjes and Sihn, IFAC-PapersOnLine 51(11), 2018. One common classification, by how data are integrated; other authors draw the lines differently.</p>""")

slide("""  <div class="kicker">Pause and predict</div>
  <h2>Model, shadow or twin?</h2>
  <p>An engineer keeps a spreadsheet model of the furnace. At the end of each shift she types in the shift's average zone temperatures, and if the predicted oxide thickness looks off, she changes the recipe by hand.</p>
  <p><strong>Predict:</strong> where does it sit in the classification?</p>
""" + reveal("A <strong>digital model</strong>. Data reach the spreadsheet only when a person types them in, and changes reach the furnace only by hand: the data flow is manual in both directions. It can still be useful; the classification says how it is connected, not how good it is."))

slide("""  <h2>Three levels of coupling</h2>
  <table style="font-size:22px;width:100%">
    <tr><th style="width:90px">Level</th><th>What the twin does</th><th>In a fab</th></tr>
    <tr><td>1</td><td>updates itself from observations</td><td>the furnace model follows each thermocouple reading</td></tr>
    <tr><td>2</td><td>its predictions support a person's decision</td><td>an engineer sees a thickness forecast and adjusts the recipe</td></tr>
    <tr><td>3</td><td>a controller acts on its predictions automatically</td><td>run-to-run control sets the next oxidation time within limits</td></tr>
  </table>
  <p class="small" style="margin-top:16px">Real systems mix levels, and most keep a person in the loop. This course uses the levels instead of one strict line between a shadow and a twin. The lab reaches level 1; Module 5 adds levels 2 and 3.</p>""")

slide("""  <h2>Fidelity and update frequency</h2>
  <div class="cols">
    <div class="card"><h3>Fidelity</h3><p>How much of the real system the twin represents, and how accurately: one zone temperature, or every zone, gas flow and wafer. The Consortium's view: enough for the use, and no more.</p></div>
    <div class="card"><h3>Update frequency</h3><p>How often the twin is synchronized: every reading, every run, every night. It must be fast enough for how quickly the real system changes and how large an error the use can accept.</p></div>
  </div>
  <p class="small">The Try it explorer and Step 6 of the lab show the cost of a slow update: the twin drifts with its model between updates.</p>""")

slide(f"""  <h2>A standard for manufacturing: ISO 23247</h2>
  <div style="margin-top:4px">{iso_fig()}</div>
  <p class="small">ISO 23247-1 to -4 (2021): overview and principles, reference architecture, digital representation, information exchange. A paid standard: described here in outline, not reproduced.</p>""")

slide(f"""  <h2>A first model of the furnace zone</h2>
  <div class="cols">
    <div style="flex:0 0 620px">{first_order_fig()}</div>
    <div>
      <div class="eq" style="font-size:26px;margin-top:0">T(t) = T<sub>set</sub> − (T<sub>set</sub> − T<sub>0</sub>) e<sup>−t/τ</sup></div>
      <ul style="padding-left:28px">
        <li style="font-size:24px">The gap to the setpoint shrinks by the same fraction every second.</li>
        <li style="font-size:24px">After <strong>τ</strong>: 63% of the way; after <strong>3τ</strong>: 95%.</li>
        <li style="font-size:24px">T<sub>0</sub> = 800 °C, T<sub>set</sub> = 1000 °C; τ = 150 s here.</li>
      </ul>
    </div>
  </div>""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>How long to get within 10 °C?</h2>
  <p>With τ = 150 s, the zone starts 200 °C below the setpoint. The gap shrinks as e<sup>−t/τ</sup>.</p>
  <p><strong>Predict:</strong> how long until the zone is within 10 °C of 1000 °C?</p>
{reveal(f"The gap must shrink by a factor of 200 / 10 = 20, so t = τ · ln 20 = 150 × {math.log(20):.2f} ≈ {T_RULE:.0f} s, about 7.5 minutes, or 3τ. A twin that knows τ can answer this before the run; that is one use of a model.")}""")

slide(f"""  <h2>Fitting τ to the first 10 minutes</h2>
  <div class="cols">
    <div style="flex:0 0 620px">{fit_fig()}</div>
    <div>
      <p style="font-size:25px"><code>fit_tau(t, y, T0, T_set)</code> finds the τ that brings the model closest to the readings (least squares, SciPy).</p>
      <p style="font-size:25px">Example stream: τ = {TAU:.2f} ± {TAU_SD:.2f} s.</p>
      <p style="font-size:25px">The simulation's real τ is {N['tau_true']:.2f} s: within one standard error. A real furnace never tells you this.</p>
    </div>
  </div>""")

slide(f"""  <h2>The model alone falls behind</h2>
  <div>{dip_fig("model")}</div>
  <p class="small" style="margin-top:2px">After about 15 minutes, something the model does not know about cools the zone. RMS error: {N['ol_early']:.2f} °C before 600 s, {N['ol_late']:.2f} °C from 600 s on, and up to {N['ol_max']:.1f} °C.</p>""")

slide(f"""  <h2>From model to twin: correct with each reading</h2>
  <div class="cols">
    <div style="flex:0 0 560px">{cycle_fig()}</div>
    <div>
      <div class="eq" style="font-size:25px;margin-top:0">T<sub>new</sub> = T<sub>pred</sub> + gain · (reading − T<sub>pred</sub>)</div>
      <ul style="padding-left:28px">
        <li style="font-size:24px">gain 0: ignore the reading; gain 1: jump to it.</li>
        <li style="font-size:24px">A missing reading (NaN) teaches nothing: keep T<sub>pred</sub>.</li>
        <li style="font-size:24px">Data flow one way, furnace to twin: a digital shadow, level 1.</li>
      </ul>
    </div>
  </div>""")

slide(f"""  <div class="kicker">Pause and predict</div>
  <h2>Does syncing always help?</h2>
  <p>The synced twin (gain 0.5) and the model alone use the same fitted τ = {TAU:.2f} s.</p>
  <p><strong>Predict:</strong> before 600 s, when the model is right, is the synced twin's error smaller or larger than the model's? And after the dip starts?</p>
{reveal(f"Before 600 s it is slightly <strong>larger</strong>: {N['sy_early']:.2f} against {N['ol_early']:.2f} °C, because each correction carries some noise from the reading into the state. From 600 s on it is much <strong>smaller</strong>: {N['sy_late']:.2f} against {N['ol_late']:.2f} °C, because the readings show the twin what the model cannot know.")}""")

slide(f"""  <h2>How often should the twin update?</h2>
  <div class="cols">
    <div style="flex:0 0 620px">{every_fig()}</div>
    <div>
      <p style="font-size:25px">Between updates, the twin runs on its model alone.</p>
      <p style="font-size:25px">Updating every 10 s keeps the error at {EVERY[0][1]:.2f} °C; every 3 minutes, {EVERY[-1][1]:.2f} °C, close to the model alone.</p>
      <p style="font-size:25px">The right frequency depends on how fast the furnace changes and on the error the use can accept.</p>
    </div>
  </div>""")

rows_g = "".join(f'<tr><td style="text-align:center">{g:.1f}</td><td style="text-align:right">{a:.2f}</td><td style="text-align:right">{b:.2f}</td></tr>' for g, a, b in GAINS)
slide(f"""  <h2>Choosing the gain: noise against tracking</h2>
  <div class="cols">
    <div style="flex:0 0 520px">
      <p style="font-size:22px;margin-bottom:8px;color:#6A626B">RMS error of the synced twin (°C)</p>
      <table style="font-size:23px;width:100%">
        <tr><th style="text-align:center">gain</th><th style="text-align:right">before 600 s</th><th style="text-align:right">from 600 s</th></tr>
        {rows_g}
      </table>
    </div>
    <div>
      <p style="font-size:25px">A small gain trusts the model and smooths the noise; a large gain trusts the readings and follows the dip.</p>
      <p style="font-size:25px">Gain 0 is the model alone. No single gain is best for both halves of the run.</p>
      <p style="font-size:25px">Module 5's Kalman filter sets the gain from the noise and the model's uncertainty.</p>
    </div>
  </div>""")

slide("""  <h2>Claim only what the data support</h2>
  <ul>
    <li>On simulated data, a good fit shows that the method recovers the course's <strong>simulator</strong>; it does not validate a model of a real furnace.</li>
    <li>With real data, check the residuals and test the model on data it was not fitted to.</li>
    <li>A first-order model fits the warm-up of this simulated zone; a real zone has several heaters, a controller and a load.</li>
    <li>Say what your twin is for, how it is connected, and how often it updates.</li>
  </ul>""")

slide("""  <h2>Summary</h2>
  <ul>
    <li>A digital twin is a virtual representation kept in step with a real entity at a chosen frequency and fidelity.</li>
    <li>Model, shadow, twin: manual or automatic data flow, in one direction or both.</li>
    <li>Levels of coupling: updates from observations, supports a decision, acts automatically.</li>
    <li>First-order furnace model: fit τ; correct with each reading using a gain.</li>
  </ul>
  <div class="eq" style="font-size:26px;margin-top:20px"><strong>Next:</strong> the Module 1 lab. You fit τ, write <code>update_prediction()</code>, and run the first twin of your own furnace stream.</div>""")

slide("""  <h2>Credits and license</h2>
  <p>These slides were written for SemiAcademy's <em>Digital Twins for Semiconductor Manufacturing</em> course, developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). License: <strong>CC BY 4.0</strong>, https://creativecommons.org/licenses/by/4.0/.</p>
  <p style="font-size:22px">Sources: Digital Twin Consortium, definition of a digital twin (short quotation, with credit); NIST IR 8356 (2025), a U.S. government work; W. Kritzinger et al., "Digital Twin in manufacturing: A categorical literature review and classification", IFAC-PapersOnLine 51(11), 1016–1022 (2018), cited; ISO 23247-1 to -4 (2021), described, not reproduced.</p>
  <p class="small">Every data value on these slides is simulated with the course's furnace model (furnace.py 1.1.0, example stream 2026). Source: github.com/CSU-Physics/digital-twins-notebooks (media/m1/m1-slides.html).</p>""")

# ------------------------------------------------------------------ assemble
head_html = (T / "deck_head.html").read_text()
head_html = head_html.replace("<title>Module 1 Slides: From Bits to Qubits</title>", "<title>Module 1 Slides: What a Digital Twin Is</title>")
assert "What a Digital Twin Is" in head_html
extra_css = ("  .code { font-family: Consolas, \"Courier New\", monospace; font-size:21px; line-height:1.4; background:var(--tint); "
             "border-left:6px solid var(--ink); padding:14px 18px; margin:0 0 12px; white-space:pre; overflow:hidden; }\n"
             "  code { font-family: Consolas, \"Courier New\", monospace; font-size:0.92em; }\n</style>")
head_html = head_html.replace("</style>", extra_css, 1)
bar = (T / "deck_bar.html").read_text().replace("1 / 19", f"1 / {len(slides)}")
ctrl = (T / "deck_ctrl.js").read_text()
html = head_html + "\n" + "\n".join(slides) + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
for bad in ("—", "${", "`"):
    assert bad not in html, bad
(OUT / "m1-slides.html").write_text(html)
print("wrote", OUT / "m1-slides.html", len(slides), "slides,", len(html), "bytes")
print({k: round(float(v), 3) for k, v in N.items()})
print("every", [(e, round(v, 3)) for e, v in EVERY])
print("gains", [(g, round(a, 3), round(b, 3)) for g, a, b in GAINS])
