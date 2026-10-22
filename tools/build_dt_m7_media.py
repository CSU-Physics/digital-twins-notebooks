"""Build the Module 7 slide deck (media/m7/m7-slides.html) for Digital Twins for Semiconductor Manufacturing.

Every number on the slides is computed here from the capstone components (content/m7) and the reference functions of the
Module 7 check, for the course example (KEY 2026), so the slides agree with the three notebooks.
Run from anywhere: python3 tools/build_dt_m7_media.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m7")]
import dt_m7_check as K  # noqa: E402
import process_twin as PT  # noqa: E402
import metrology_twin as MT  # noqa: E402
import toolset_twin as TT  # noqa: E402

T = ROOT / "tools"
OUT = ROOT / "media" / "m7"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, INK, MUTED, LINE, TINT, OK, BAD, BLUE, GOLD, GREEN = "#99004C", "#24313D", "#6A626B", "#E4DCE0", "#F8F4F6", "#2E7D4F", "#B3261E", "#2F5E8E", "#A86B00", "#2E7D4F"
EX = 2026

# ------------------------------------------------------------------ numbers used on the slides (the course example, KEY 2026)
R = {t: K.reference_results(t, EX) for t in ("process", "metrology", "toolset")}
V = {t: {k: K.reference_verdict(R[t][k][0], R[t][k][1], R[t]["tolerance"]) for k in ("heldout", "challenging")} for t in R}
UNIT = {"process": "nm", "metrology": "nm", "toolset": "h"}
assert V["process"] == {"heldout": "meets", "challenging": "misses"}
assert V["metrology"] == {"heldout": "meets", "challenging": "insufficient evidence"}
assert V["toolset"] == {"heldout": "meets", "challenging": "misses"}


def f3(x):
    return f"{x:.3f}"


def txt(x, y, s, size=20, fill=INK, anchor="middle", weight=400, extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{extra}>{s}</text>'


def arrow_def(mid, color=MUTED):
    return (f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0 0L10 5L0 10z" fill="{color}"/></marker></defs>')


def box(x, y, w, h, title, sub, col, title_size=23, sub_size=17):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#fff" stroke="{col}" stroke-width="3"/>',
         txt(x + w / 2, y + h / 2 - (6 if sub else -8), title, title_size, INK, weight=600)]
    if sub:
        s.append(txt(x + w / 2, y + h / 2 + 22, sub, sub_size, MUTED))
    return "".join(s)


def arrow(x1, y1, x2, y2, mid, color=MAROON, width=4):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" marker-end="url(#{mid})"/>'


# ------------------------------------------------------------------ figures
def pipeline_fig(w=1130, h_=250, mid="p1"):
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 1130 250" role="img" aria-label="Prepared components from Modules 3 to 6 are connected into a twin, which is verified, validated, and described by a data sheet">',
         arrow_def(mid, MAROON)]
    comps = [("Module 3", "Deal-Grove"), ("Module 4", "virtual metrology"), ("Module 5", "sync: Kalman"), ("Module 6", "SimPy line")]
    for k, (a, b) in enumerate(comps):
        y = 14 + 56 * k
        s.append(f'<rect x="10" y="{y}" width="210" height="46" rx="8" fill="{TINT}" stroke="{LINE}"/>')
        s.append(txt(115, y + 20, a, 17, MUTED))
        s.append(txt(115, y + 39, b, 18, INK, weight=600))
        s.append(arrow(222, y + 23, 268, 125, mid, MUTED, 2))
    s.append(box(272, 80, 190, 90, "your twin", "connected parts", MAROON))
    s.append(arrow(464, 125, 504, 125, mid))
    s.append(box(508, 80, 180, 90, "verify", "the code", BLUE))
    s.append(arrow(690, 125, 730, 125, mid))
    s.append(box(734, 80, 180, 90, "validate", "against data", GREEN))
    s.append(arrow(916, 125, 956, 125, mid))
    s.append(box(960, 80, 160, 90, "data sheet", "6 fields", GOLD))
    s.append(txt(820, 218, "held-out data · baseline · one challenging condition · a tolerance set first", 18, MUTED))
    s.append("</svg>")
    return "".join(s)


def vv_fig(w=620, h_=390, mid="v1"):
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 620 390" role="img" aria-label="Reality is simplified into a mathematical model, which is implemented as code. Verification compares the code with the model; validation compares the code\'s predictions with measurements of reality.">',
         arrow_def(mid, MAROON), arrow_def(mid + "g", GREEN), arrow_def(mid + "b", BLUE)]
    s.append(box(20, 20, 210, 78, "reality", "the furnace, the line", INK))
    s.append(box(390, 20, 210, 78, "model", "equations, rules", INK))
    s.append(box(390, 232, 210, 78, "code", "the twin you run", INK))
    s.append(arrow(232, 59, 386, 59, mid))
    s.append(txt(309, 48, "simplify", 17, MUTED))
    s.append(arrow(575, 100, 575, 228, mid))
    s.append(txt(568, 170, "implement", 16, MUTED, "end"))
    s.append(f'<line x1="440" y1="226" x2="440" y2="104" stroke="{BLUE}" stroke-width="3" stroke-dasharray="7 5" marker-end="url(#{mid}b)" marker-start="url(#{mid}b)"/>')
    s.append(txt(430, 150, "verification", 20, BLUE, "end", 600))
    s.append(txt(430, 174, "does the code solve", 16, BLUE, "end"))
    s.append(txt(430, 194, "the model correctly?", 16, BLUE, "end"))
    s.append(f'<path d="M386 280 L 125 280 L 125 104" fill="none" stroke="{GREEN}" stroke-width="3" marker-end="url(#{mid}g)" marker-start="url(#{mid}g)"/>')
    s.append(txt(20, 334, "validation: are the code's predictions close enough", 18, GREEN, "start", 600))
    s.append(txt(20, 358, "to measurements of reality, for the use you intend?", 18, GREEN, "start", 600))
    s.append("</svg>")
    return "".join(s)


def prequential_fig(w=1130, h_=230, mid="q1"):
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 1130 230" role="img" aria-label="For each run: the twin predicts, then the run is measured, the error is recorded, and only then the measurement updates the twin for the next run">',
         arrow_def(mid, MAROON)]
    steps = [("1. predict", "before the run", MAROON), ("2. run", "the tool or line", INK), ("3. measure", "the wafer or week", BLUE),
             ("4. score", "measured − predicted", GOLD), ("5. update", "for the next run", GREEN)]
    for k, (a, b, col) in enumerate(steps):
        x = 10 + 224 * k
        s.append(box(x, 40, 200, 96, a, b, col, 24, 16))
        if k < 4:
            s.append(arrow(x + 202, 88, x + 222, 88, mid))
    s.append(f'<path d="M1010 140 C 1010 205 120 205 110 142" fill="none" stroke="{MUTED}" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#{mid})"/>')
    s.append(txt(565, 222, "next run: the error of every run was made before its measurement was used, so it is an honest test", 18, MUTED))
    s.append("</svg>")
    return "".join(s)


def verdict_fig(w=1130, h_=300):
    tol = 0.5
    X = lambda v: 150 + v / 1.2 * 640
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 1130 300" role="img" aria-label="Three 95 % intervals against a tolerance of 0.5: one wholly below (meets), one containing it (insufficient evidence), one wholly above (misses)">']
    s.append(f'<rect x="{X(0)}" y="20" width="{X(tol) - X(0)}" height="230" fill="{OK}" opacity="0.08"/>')
    s.append(f'<line x1="{X(tol)}" y1="14" x2="{X(tol)}" y2="252" stroke="{INK}" stroke-width="3" stroke-dasharray="8 6"/>')
    s.append(txt(X(tol), 278, "tolerance 0.5", 20, INK, weight=600))
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
        s.append(txt(X(v), 296, f"{v:g}", 17, MUTED))
    cases = [(0.30, 0.10, "meets", OK, "MAE + half-width ≤ tolerance"), (0.45, 0.10, "insufficient evidence", GOLD, "the interval contains the tolerance"),
             (0.90, 0.20, "misses", BAD, "MAE − half-width > tolerance")]
    for k, (m, h, name, col, why) in enumerate(cases):
        y = 60 + 72 * k
        s.append(f'<line x1="{X(m - h)}" y1="{y}" x2="{X(m + h)}" y2="{y}" stroke="{col}" stroke-width="6"/>')
        for e in (m - h, m + h):
            s.append(f'<line x1="{X(e)}" y1="{y - 12}" x2="{X(e)}" y2="{y + 12}" stroke="{col}" stroke-width="4"/>')
        s.append(f'<circle cx="{X(m)}" cy="{y}" r="8" fill="{col}"/>')
        s.append(txt(20, y + 7, f"{m:.2f} ± {h:.2f}", 19, INK, "start"))
        s.append(txt(850, y + 2, name, 23, col, "start", 600))
        s.append(txt(850, y + 26, why, 17, MUTED, "start"))
    s.append("</svg>")
    return "".join(s)


def ratio_fig(w=620, h_=390):
    """Each track's held-out and challenging MAE, with its interval, as a multiple of its tolerance (log scale)."""
    import math
    rows = [("process", "held-out"), ("process", "challenging"), ("metrology", "held-out"), ("metrology", "challenging"), ("toolset", "held-out"), ("toolset", "challenging")]
    lo_v, hi_v = 0.3, 12.0
    X = lambda v: 190 + (math.log10(max(v, lo_v)) - math.log10(lo_v)) / (math.log10(hi_v) - math.log10(lo_v)) * 300
    s = [f'<svg width="{w}" height="{h_}" viewBox="0 0 620 390" role="img" aria-label="Mean absolute error divided by the tolerance, with 95 % intervals, for each track\'s held-out and challenging tests on the course example, on a log scale">']
    s.append(f'<rect x="{X(lo_v)}" y="12" width="{X(1) - X(lo_v)}" height="318" fill="{OK}" opacity="0.10"/>')
    s.append(f'<line x1="{X(1)}" y1="8" x2="{X(1)}" y2="332" stroke="{INK}" stroke-width="2.5" stroke-dasharray="7 5"/>')
    for v in (0.3, 0.5, 1, 2, 5, 10):
        s.append(txt(X(v), 352, f"{v:g}", 17, MUTED))
    s.append(txt(340, 380, "MAE / tolerance, with its 95 % interval", 18, MUTED))
    for k, (t, which) in enumerate(rows):
        key = "heldout" if which == "held-out" else "challenging"
        m, h, _ = R[t][key]
        tol = R[t]["tolerance"]
        v = V[t][key]
        col = OK if v == "meets" else BAD if v == "misses" else GOLD
        y = 34 + 52 * k
        s.append(txt(180, y + 6, f"{t}, {which}", 16, INK, "end"))
        s.append(f'<line x1="{X((m - h) / tol)}" y1="{y}" x2="{X((m + h) / tol)}" y2="{y}" stroke="{col}" stroke-width="5"/>')
        s.append(f'<circle cx="{X(m / tol)}" cy="{y}" r="7" fill="{col}"/>')
        s.append(txt(612, y + 6, "insufficient" if v == "insufficient evidence" else v, 16, col, "end", 600))
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ slides
FOOT = "Module 7 · Capstone: integrate, validate and document"
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


P, M_, TS = R["process"], R["metrology"], R["toolset"]

slide(f"""  <div class="kicker">Digital Twins for Semiconductor Manufacturing · Module 7</div>
  <h1>Capstone: integrate, validate and document</h1>
  <p style="font-size:30px;max-width:980px">Join prepared parts into one twin, show that the code does what the model says, test the twin on data it has not seen, and write down what it is fit for.</p>
  <div style="margin-top:6px">{pipeline_fig(1060, 235, "p0")}</div>""", "slide title-slide")

slide("""  <h2>By the end of this module you can</h2>
  <ul>
    <li>tell <strong>verification</strong> (does the code solve the model?) from <strong>validation</strong> (is the twin accurate enough for its use?);</li>
    <li>connect prepared components into one <strong>twin prototype</strong> and verify the connection;</li>
    <li>test the twin on <strong>held-out</strong> data and on one <strong>challenging condition</strong>, against a <strong>tolerance</strong> and a <strong>baseline</strong>;</li>
    <li>read a test as <strong>meets</strong>, <strong>misses</strong> or <strong>insufficient evidence</strong> from the error and its interval;</li>
    <li>complete a <strong>twin data sheet</strong>: intended use, data, results, where not to use it, one risk, and the action when confidence is low.</li>
  </ul>
  <p class="small" style="margin-top:12px">Course outcome 7. You choose one of three tracks: a process twin, a metrology twin or a toolset twin.</p>""")

slide(f"""  <h2>Three tracks, one method</h2>
  <div class="cols" style="margin-top:6px">
    <div class="card"><h3>Process twin</h3><p>Module 3's calibrated Deal-Grove model, kept in sync by Module 5's Kalman filter. It predicts each run's monitor wafer and recommends the oxidation time.</p><p class="small" style="margin-top:10px">tolerance: MAE ≤ {P['tolerance']:g} nm</p></div>
    <div class="card"><h3>Metrology twin</h3><p>Module 4's hybrid virtual-metrology model, kept in sync with one sampled wafer per run. It predicts every product wafer from the furnace traces.</p><p class="small" style="margin-top:10px">tolerance: MAE ≤ {M_['tolerance']:g} nm</p></div>
    <div class="card"><h3>Toolset twin</h3><p>Module 6's SimPy line with the drifting furnace of Module 5, calibrated from the fab's lot log. It forecasts the weekly mean cycle time.</p><p class="small" style="margin-top:10px">tolerance: weekly MAE ≤ {TS['tolerance']:g} h</p></div>
  </div>
  <p class="small" style="margin-top:14px">Each track takes about 2 hours, ends with the same six-field data sheet, and has its own project quiz. You need one.</p>""")

slide(f"""  <h2>Verification and validation</h2>
  {two(vv_fig(), ['<strong>Verification</strong>: does the code compute what the model says? Compare it with answers you can work out exactly.',
                  '<strong>Validation</strong>: how close are the twin\'s predictions to measurements, judged by the use you intend?',
                  'Verification first: a validation result means little if the code is wrong.',
                  'Definitions of the National Research Council (2012), used by the National Academies (2024) for digital twins.'], size=22)}""")

slide("""  <div class="kicker">Pause and predict</div>
  <h2>Verification or validation?</h2>
  <ol style="font-size:25px">
    <li>In Module 6 you checked that L = λW holds on the simulation's output.</li>
    <li>In Module 4 you scored three models on a time-ordered test set of later runs.</li>
    <li>The twin aims at the target with the recipe rule; its own prediction for that time is then exactly the target.</li>
    <li>In Module 3 you predicted the thin-oxide runs and found them thicker than the model said.</li>
  </ol>
""" + reveal("1 and 3 are <strong>verification</strong>: they check the code against an exact property of the model (Little's law, the inverse of the recipe rule). 2 and 4 are <strong>validation</strong>: they compare predictions with measured data, and 4 shows where the model does not apply."))

slide("""  <h2>You have done both already</h2>
  <table style="font-size:22px">
    <tr><th>module</th><th>verification</th><th>validation</th></tr>
    <tr><td>3, Deal-Grove</td><td>your x(t) against exact values of the model</td><td>a new run inside the range; the thin-oxide runs outside it</td></tr>
    <tr><td>4, virtual metrology</td><td>preprocessing inside each CV fold</td><td>time-ordered test set; a shifted recipe</td></tr>
    <tr><td>5, keeping in sync</td><td>Kalman gains against the steady-state value</td><td>controllers on the same disturbances; the failing thermocouple</td></tr>
    <tr><td>6, toolset</td><td>Little's law on the output</td><td>(the model line only: no fab data yet)</td></tr>
  </table>
  <p class="small" style="margin-top:16px">The capstone puts the pieces together and adds what each module did only in part: a test plan written first, a verdict with an interval, and a record of the result.</p>""")

slide("""  <h2>Write the test plan before the test</h2>
  <div class="cols">
    <div class="card"><h3>Tolerance</h3><p>From the intended use, set before any test runs. Changing it after seeing the result is not validation.</p></div>
    <div class="card"><h3>Held-out data</h3><p>Data the twin never saw while it was built or tuned, in time order.</p></div>
  </div>
  <div class="cols" style="margin-top:16px">
    <div class="card"><h3>Baseline</h3><p>The simplest method that could work. Not beating it is a valid finding.</p></div>
    <div class="card"><h3>One challenging condition</h3><p>Harder than normal operation, built in on purpose and named in the notebook.</p></div>
  </div>""")

slide(f"""  <h2>Predict first, then use the measurement</h2>
  <div>{prequential_fig()}</div>
  <p style="font-size:24px">A twin that stays in sync keeps learning from its measurements. It can still be tested honestly: score each prediction before its measurement updates the twin. Every error is then out of sample.</p>""")

slide(f"""  <h2>The error, and how sure you are of it</h2>
  <div class="eq">MAE = mean |measured − predicted| &nbsp;&nbsp; half-width = t<sub>0.975, n−1</sub> · s / √n</div>
  <ul style="font-size:24px">
    <li>One error per <strong>independent test unit</strong>: a run, or a week. The five wafers of a run share the run's effects, so they count as one unit, not five.</li>
    <li>Process twin, example furnace: MAE {f3(P['heldout'][0])} nm, half-width {f3(P['heldout'][1])} nm, over {P['heldout'][2]} runs.</li>
    <li>More test units narrow the interval, about as 1/√n (Module 6).</li>
  </ul>""")

slide(f"""  <h2>Three verdicts</h2>
  <div>{verdict_fig()}</div>
  <p style="font-size:23px">"Insufficient evidence" is an honest result, not a failure: the test cannot tell, and more data would. A "meets" that the data do not support is the result to avoid.</p>""")

slide("""  <div class="kicker">Pause and predict</div>
  <h2>Which verdict?</h2>
  <p>A twin's tolerance is 1.0 h. On 10 test weeks its MAE is 1.4 h with a half-width of 0.6 h. On 24 other weeks it is 0.6 h ± 0.2 h.</p>
""" + reveal("10 weeks: 0.8 to 2.0 h contains 1.0, so <strong>insufficient evidence</strong>: the twin may or may not meet the tolerance there, and the data sheet must not claim it does. 24 weeks: 0.4 to 0.8 h lies below 1.0, so <strong>meets</strong>."))

slide(f"""  <h2>Twin against baseline, course example</h2>
  <table style="font-size:23px">
    <tr><th>track</th><th>baseline</th><th>baseline MAE</th><th>twin MAE</th></tr>
    <tr><td>process</td><td>Deal-Grove alone, no sync</td><td>{f3(P['baseline'][0])} nm</td><td>{f3(P['heldout'][0])} nm</td></tr>
    <tr><td>metrology</td><td>mean of the last 5 sampled wafers</td><td>{f3(M_['baseline'][0])} nm</td><td>{f3(M_['heldout'][0])} nm</td></tr>
    <tr><td>toolset</td><td>mean weekly cycle time of weeks 1 to 6</td><td>{f3(TS['baseline'][0])} h</td><td>{f3(TS['heldout'][0])} h</td></tr>
  </table>
  <p style="font-size:23px;margin-top:16px">In all three the twin wins clearly. In your own work it may not: a twin that does not beat a simple method is a finding worth reporting, and the simple method may be enough.</p>""")

slide(f"""  <h2>The challenging condition decides where not to use the twin</h2>
  {two(ratio_fig(), ['<strong>Process:</strong> the failing thermocouple; the filter, tuned for slow drift, lags behind.',
                      '<strong>Metrology:</strong> after a thermocouple replacement the traces change meaning; the interval is wide, so <em>insufficient evidence</em>.',
                      '<strong>Toolset:</strong> lithography breakdowns that the twin does not model; it is wrong by hours.'], size=21)}""")

slide("""  <h2>The twin data sheet</h2>
  <table style="font-size:22px">
    <tr><th>field</th><th>what it records</th></tr>
    <tr><td>1. intended use</td><td>the asset, its recipe or line, and the decision the twin supports</td></tr>
    <tr><td>2. data and updates</td><td>what the twin reads and how often it is updated</td></tr>
    <tr><td>3. validation results</td><td>held-out, challenging and baseline MAE; the two verdicts</td></tr>
    <tr><td>4. do not use for</td><td>conditions outside what was tested, and tests that did not meet the tolerance</td></tr>
    <tr><td>5. one risk</td><td>a data-integrity or security risk, with a mitigation that reduces it</td></tr>
    <tr><td>6. low confidence</td><td>what the user does when the twin cannot be trusted</td></tr>
  </table>
  <p class="small" style="margin-top:12px">Modeled on datasheets for datasets (Gebru et al., 2021) and model cards (Mitchell et al., 2019).</p>""")

slide("""  <h2>Field 4: where the twin must not be used</h2>
  <ul style="font-size:25px">
    <li>Every condition <strong>outside</strong> the range the twin was calibrated and tested in: nothing supports a claim there.</li>
    <li>Every tested condition whose verdict was <strong>misses</strong> or <strong>insufficient evidence</strong>.</li>
    <li><strong>Not</strong> a condition your tests included and that met the tolerance: the results support it, and listing it would hide what the twin can do.</li>
  </ul>
  <div class="eq" style="font-size:24px">Claim only what the data support. On simulated data, a result shows how the twin recovers the course's simulator, not how it would work on a real tool.</div>""")

slide("""  <h2>Field 5: one risk and its mitigation</h2>
  <p style="font-size:24px">A twin is only as good as the data it reads, and a twin that sends decisions back is a path into the tool (NIST IR 8356).</p>
  <table style="font-size:21px">
    <tr><th>risk</th><th>a mitigation that reduces it</th></tr>
    <tr><td>a measurement attributed to the wrong run</td><td>match run ID and timestamp with the tool log; reject what does not match</td></tr>
    <tr><td>an unauthorized change sent through the sync channel</td><td>advisory mode with authenticated, logged approval of every change</td></tr>
    <tr><td>a predicted value recorded as if measured</td><td>label every value as predicted or measured</td></tr>
  </table>
  <p class="small" style="margin-top:12px">A mitigation that does not reduce the risk, such as widening the tolerance or retraining more often, is not a mitigation.</p>""")

slide("""  <h2>Field 6: when confidence is low</h2>
  <div class="cols">
    <div class="card"><h3>Stop relying on it</h3><p>Hold the last approved setting, or switch to advisory mode.</p></div>
    <div class="card"><h3>Measure instead</h3><p>Physical metrology, or the measured cycle times, until the cause is understood.</p></div>
    <div class="card"><h3>Bring it back on evidence</h3><p>Use the twin again only when new data show it meets the tolerance.</p></div>
  </div>
  <p style="font-size:23px;margin-top:18px">Not: widen the tolerance, refit on the test data and call it validated, or switch to another method without recording it.</p>""")

slide("""  <h2>Validation is never finished</h2>
  <ul style="font-size:25px">
    <li>The furnace drifts, thermocouples are replaced, recipes and products change: the twin you validated last month describes a slightly different tool.</li>
    <li>The National Academies' 2024 report on digital twins concludes that their verification, validation and uncertainty quantification must be a <strong>continual process</strong>.</li>
    <li>A data sheet therefore carries a date, and the sync layer, the alarms of Module 5 and new held-out data keep testing the twin while it is used.</li>
  </ul>""")

slide("""  <h2>Careers and the SMART USA Institute (optional)</h2>
  <ul style="font-size:24px">
    <li><strong>SMART USA</strong> (Semiconductor Manufacturing and Advanced Research with Twins) is the CHIPS Manufacturing USA institute for digital twins, led by the Semiconductor Research Corporation with Purdue as lead academic institution.</li>
    <li>It has $285 million in federal funding over five years and aims to train more than 110,000 workers and students in digital twin technology.</li>
    <li>Twin skills from this course are used by process, equipment, yield, metrology and industrial engineers, and by the data engineers who connect tools to models.</li>
  </ul>
  <p class="small">Sources: NIST, CHIPS award page for SMART USA; Purdue University news release (November 2024).</p>""")

slide("""  <h2>Summary</h2>
  <ul style="font-size:24px">
    <li>A twin is assembled from parts; the connection is code, and code needs <strong>verification</strong>.</li>
    <li><strong>Validation</strong> follows a plan written first: tolerance, held-out data, baseline, one challenging condition.</li>
    <li>Score each prediction before its measurement is used, with one error per independent test unit.</li>
    <li>The verdict compares the whole interval with the tolerance: <strong>meets</strong>, <strong>misses</strong> or <strong>insufficient evidence</strong>.</li>
    <li>The <strong>data sheet</strong> records what the twin is for, how it did, where it must not be used, one risk and the low-confidence action.</li>
  </ul>
  <div class="eq" style="font-size:21px;margin-top:10px"><strong>Next:</strong> choose your track, build and test the twin, fill in the data sheet, and take that track's project quiz.</div>""")

slide("""  <h2>Credits and license</h2>
  <p>These slides were written for SemiAcademy's <em>Digital Twins for Semiconductor Manufacturing</em> course, developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). License: <strong>CC BY 4.0</strong>, https://creativecommons.org/licenses/by/4.0/.</p>
  <p style="font-size:18px">Sources: National Academies of Sciences, Engineering, and Medicine, <em>Foundational Research Gaps and Future Directions for Digital Twins</em>, National Academies Press (2024), doi 10.17226/26894; National Research Council, <em>Assessing the Reliability of Complex Models</em>, National Academies Press (2012); T. Gebru et al., "Datasheets for datasets", Communications of the ACM 64(12), 86-92 (2021); M. Mitchell et al., "Model cards for model reporting", FAT* 2019, doi 10.1145/3287560.3287596; J. Voas, P. Mell, P. Laplante and V. Piroumian, NIST IR 8356, <em>Security and Trust Considerations for Digital Twin Technology</em> (2025), doi 10.6028/NIST.IR.8356; A. M. Law, <em>Simulation Modeling and Analysis</em>, 5th ed., McGraw-Hill (2015); NIST, CHIPS Manufacturing USA award to SMART USA. Cited, not reproduced.</p>
  <p class="small">Every data value on these slides is simulated with the course's models (content/m7, example KEY 2026). Source: github.com/CSU-Physics/digital-twins-notebooks (media/m7/m7-slides.html).</p>""")

# ------------------------------------------------------------------ assemble
head_html = (T / "deck_head.html").read_text()
head_html = head_html.replace("<title>Module 1 Slides: From Bits to Qubits</title>", "<title>Module 7 Slides: Capstone</title>")
assert "Capstone" in head_html
head_html = head_html.replace("</style>", "  code { font-family: Consolas, \"Courier New\", monospace; font-size:0.92em; }\n  .slide ol { margin:0 0 10px 0; padding-left:38px; }\n  .slide ol li { font-size:25px; line-height:1.45; margin:0 0 12px; }\n</style>", 1)
bar = (T / "deck_bar.html").read_text().replace("1 / 19", f"1 / {len(slides)}")
ctrl = (T / "deck_ctrl.js").read_text()
html = head_html + "\n" + "\n".join(slides) + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
for bad in ("—", "–", "${", "`"):
    assert bad not in html, bad
(OUT / "m7-slides.html").write_text(html)
print("wrote", OUT / "m7-slides.html", len(slides), "slides,", len(html), "bytes")
for t in R:
    print(t, {k: tuple(round(x, 3) for x in R[t][k][:2]) + (R[t][k][2],) for k in ("heldout", "challenging", "baseline")}, V[t],
          "value", K.personal_value(t, EX))
