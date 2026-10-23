"""Build the Module 3 browser notebook (Digital Twins for Semiconductor Manufacturing), its Colab copy, and copy its files.

content/m3 holds furnace.py (1.3.0), oxide_tools.py (supplied fitting helpers) and dt_m3_check.py (copied from checks/).
The check cell only imports the check file, so the course's reference solutions are not on screen."""
import shutil
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "m3"
NAME = "DT-M3-lab-oxidation-model.ipynb"
VERSION = "2026-10-25"
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}

shutil.copyfile(ROOT / "checks" / "dt_m3_check.py", OUT / "dt_m3_check.py")


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 3 lab: a physics model of thermal oxidation

**Digital Twins for Semiconductor Manufacturing · Module 3 · about 75 minutes** · notebook version {VERSION}

The furnace twin needs to know how much oxide a run will grow. In this lab you calibrate the **Deal-Grove model** of dry thermal oxidation to measurements from the course's simulated furnace, and you find out how far to trust it:

1. write `dg_tau()` and `dg_thickness()`, the two formulas of the model;
2. get starting values from a straight-line form of the model, and see why it is not used for the uncertainty;
3. fit the model's two rate constants to 12 measurements, with the measurement uncertainty placed on the thickness;
4. predict a new run, with a **confidence band** for the predicted mean and a **prediction band** for a new measurement, and compare with the run's measured wafers;
5. see the bands widen when you extrapolate;
6. test the model on a **thin-oxide run**, where it is deliberately wrong;
7. calibrate the model for **your own** furnace, set by your KEY, and get your verification value.

All the data are **simulated**. The **Module 3 Lab Check** in Canvas asks about your results, so keep this lab open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

The first time, Python can take up to a minute to start, and SciPy, which the fitting helper uses, adds a few seconds. The cell prints the versions of the libraries and of the two course files next to this notebook: `furnace.py`, the furnace simulator, and `oxide_tools.py`, the fitting helpers for this lab."""),
code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import furnace, oxide_tools
from furnace import oxidation_runs, oxidation_truth, textbook_rates, OX_XI_NM, OX_SIGMA_NM, OX_NEW_MIN, THIN_XI_NM
from oxide_tools import linear_form_fit, fit_deal_grove, predict

print("Notebook version {VERSION}")
print("NumPy", np.__version__, "| pandas", pd.__version__, "| SciPy", scipy.__version__,
      "| furnace.py", furnace.__version__, "| oxide_tools.py", oxide_tools.__version__)"""),
md("""## Step 1: the Deal-Grove model, and your task `dg_tau()`

In dry oxidation, oxygen diffuses through the oxide that is already there and reacts with the silicon underneath, so the oxide grows from the bottom. Deal and Grove (1965) showed that the thickness x after a time t follows

x² + A·x = B·(t + τ)

- **B** (nm²/min) is the **parabolic rate constant**: it sets the growth of thick oxides, where the oxygen's diffusion through the oxide limits the rate.
- **B/A** (nm/min) is the **linear rate constant**: it sets the growth of thin oxides, where the reaction at the silicon surface limits the rate. A = B / (B/A), in nm.
- **τ** (min) accounts for the oxide x_i that is already on the wafer: it is the time the model would need to grow x_i, τ = (x_i² + A·x_i) / B.

Both constants grow steeply with temperature, as Arrhenius laws. `textbook_rates(1000)` gives the values from Deal and Grove's table for dry O₂ at 1000 °C on (100) silicon, converted to nm and minutes.

**Your task:** write `dg_tau(xi, B, BA)`, which returns τ in minutes. Here `BA` stands for B/A."""),
code("""def dg_tau(xi, B, BA):
    \"\"\"tau (minutes): the time the Deal-Grove model needs to grow the starting oxide xi (nm). BA is B/A.\"\"\"
    # YOUR CODE HERE: compute A from B and BA, then tau
    raise NotImplementedError("Complete dg_tau() first.")"""),
code("""B_tb, BA_tb = textbook_rates(1000)
print(f"textbook values, dry O2, 1000 C, (100) silicon: B = {B_tb:.2f} nm^2/min, B/A = {BA_tb:.4f} nm/min, A = {B_tb / BA_tb:.1f} nm")
print(f"tau for a starting oxide of 25 nm: {dg_tau(25.0, B_tb, BA_tb):.1f} min")
print(f"tau for no starting oxide: {dg_tau(0.0, B_tb, BA_tb):.1f} min   (should be 0.0)")"""),
md("""Write down τ for 25 nm, to 1 decimal: the lab check asks for it."""),
md("""## Step 2: your task, `dg_thickness()`

Solving x² + A·x = B·(t + τ) for x gives the thickness after a time t:

x(t) = (A/2) · ( √(1 + 4·B·(t + τ)/A²) − 1 )

**Your task:** write `dg_thickness(t, B, BA, xi)`. It must work for a single time and for a NumPy array of times, so use `np.sqrt`, and use your `dg_tau()` for τ. At t = 0 it must return x_i."""),
code("""def dg_thickness(t, B, BA, xi):
    \"\"\"Deal-Grove oxide thickness (nm) after t minutes, starting from xi nm of oxide. Works for an array of times.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete dg_thickness() first.")"""),
code("""print(f"x(0) = {float(dg_thickness(0.0, B_tb, BA_tb, 25.0)):.2f} nm   (should be 25.00)")
print(f"x(120 min) = {float(dg_thickness(120.0, B_tb, BA_tb, 25.0)):.2f} nm, from a 25 nm start, with the textbook values")

t = np.linspace(0, 1440, 300)
x = dg_thickness(t, B_tb, BA_tb, 0.0)
A_tb = B_tb / BA_tb
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(t / 60, x, lw=2.5, label="Deal-Grove, bare wafer")
ax.plot(t / 60, BA_tb * t, "--", label="linear limit: (B/A) t")
ax.plot(t / 60, np.sqrt(B_tb * t), ":", lw=2, label="parabolic limit: sqrt(B t)")
ax.set_ylim(0, 450)
ax.set_xlabel("time (hours)")
ax.set_ylabel("oxide thickness (nm)")
ax.legend()
plt.show()"""),
md("""**What to notice.** While the oxide is much thinner than A (about 230 nm here), growth is nearly linear in time: the surface reaction limits it. Once the oxide is much thicker than A, growth slows toward √(B·t): diffusion through the oxide limits it. The real curve moves smoothly from one limit to the other."""),
md("""## Step 3: calibration data from the furnace

A model with textbook constants is only a starting point: every furnace differs a little, through its temperature calibration, gas flow and wafers. So the twin calibrates B and B/A on its own furnace.

`oxidation_runs(key)` returns three sets of measurements from the simulated furnace at 1000 °C in dry O₂:

- `"calibration"`: 12 runs from 30 to 480 minutes, each on a wafer with a starting oxide of **25.0 nm**, grown and measured before these runs;
- `"new"`: three wafers of one new run of 270 minutes, which you will predict before looking;
- `"thin"`: four short runs on bare wafers, used in Step 8.

Every thickness is one ellipsometer measurement. The metrology group states its standard deviation: **σ = 0.5 nm**, the same for every measurement. The cell loads the example furnace (key 2026)."""),
code("""runs = oxidation_runs(2026)
cal = runs["calibration"]
print(cal.to_string(index=False))
print(f"starting oxide {OX_XI_NM} nm, measurement standard deviation {OX_SIGMA_NM} nm")

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.errorbar(cal["time_min"], cal["thickness_nm"], yerr=OX_SIGMA_NM, fmt="o", ms=4, capsize=3, label="calibration runs")
ax.plot(t, dg_thickness(t, B_tb, BA_tb, OX_XI_NM), "--", label="textbook constants")
ax.set_xlim(0, 520); ax.set_ylim(0, 230)
ax.set_xlabel("oxidation time (min)")
ax.set_ylabel("oxide thickness (nm)")
ax.legend()
plt.show()"""),
md("""## Step 4: a straight-line form, for starting values only

Rearranged, the model is a straight line in two unknowns:

t = (x² − x_i²) · (1/B) + (x − x_i) · (1/(B/A))

so ordinary least squares on t gives 1/B and 1/(B/A). `linear_form_fit(t, x, xi)` does exactly that. The cell uses it on three runs, then on all 12.

This form has a catch. Least squares assumes that the quantity on the left, t, carries the noise and the ones on the right are exact. Here it is the other way round: the times are set exactly by the recipe, and the noise is in the measured thickness x, which sits on the right, squared. That is an **errors-in-variables** problem, and the straight-line fit gives no honest uncertainty. The course uses it only for starting values."""),
code("""three = cal.iloc[[0, 5, 11]]
B3, BA3 = linear_form_fit(three["time_min"], three["thickness_nm"], OX_XI_NM)
B12, BA12 = linear_form_fit(cal["time_min"], cal["thickness_nm"], OX_XI_NM)
print(f"straight-line form, 3 runs:  B = {B3:.1f} nm^2/min, B/A = {BA3:.4f} nm/min")
print(f"straight-line form, 12 runs: B = {B12:.1f} nm^2/min, B/A = {BA12:.4f} nm/min")"""),
md("""## Step 5: fit the model, with the uncertainty on the thickness

`fit_deal_grove(t, x, sigma, xi)` fits the thickness x(t) itself, so the noise is where it belongs. It uses SciPy's `curve_fit`, a weighted nonlinear least-squares fit, with:

- `sigma` = the stated standard deviation of each thickness, and `absolute_sigma=True`, because that value is known: the parameter uncertainties then come from σ itself, not from how well this particular fit happens to scatter;
- only **B** and **B/A** as free parameters. τ is not a third parameter: it follows from the known starting oxide.

It returns B and B/A with their standard errors and their correlation, the χ² of the fit and the residuals."""),
code("""fit = fit_deal_grove(cal["time_min"], cal["thickness_nm"], OX_SIGMA_NM, OX_XI_NM)
print(f"B   = {fit['B']:.2f} +/- {fit['se_B']:.2f} nm^2/min   (textbook {B_tb:.2f})")
print(f"B/A = {fit['BA']:.4f} +/- {fit['se_BA']:.4f} nm/min   (textbook {BA_tb:.4f})")
print(f"correlation of the two estimates: {fit['corr']:.2f}")
print(f"chi^2 = {fit['chi2']:.2f} for {fit['dof']} degrees of freedom")

fig, ax = plt.subplots(figsize=(8, 2.8))
ax.axhline(0, color="gray", lw=1)
ax.plot(cal["time_min"], fit["residuals"] / OX_SIGMA_NM, "o")
ax.set_xlabel("oxidation time (min)")
ax.set_ylabel("residual / sigma")
plt.show()"""),
md("""**What to notice.**

- In the example, this furnace's constants differ from the textbook by a few percent, several standard errors: calibrating was worth it.
- The two estimates are strongly **negatively correlated**: a slightly larger B with a slightly smaller B/A describes the data almost as well. Their uncertainties cannot be combined as if they were independent; the helper uses the full covariance.
- χ² close to its degrees of freedom, and residuals scattered within about ±2σ with no pattern, mean the model describes these 12 runs to within the stated measurement noise."""),
md("""## Step 6: predict a new run, with two bands

`predict(fit, t_new)` returns, for each time:

- `x`, the predicted thickness;
- `se_mean`, the standard error of the **predicted mean**: it comes from the uncertainty of B and B/A (their covariance and the gradient of x(t) with respect to them). The **confidence band** is x ± 1.96·se_mean;
- `se_pred`, the standard error for **one new measurement**: se_pred = √(se_mean² + σ²), because a new measurement adds its own noise. The **prediction band** is x ± 1.96·se_pred.

The new run lasted 270 minutes, inside the calibrated range (30 to 480 minutes). The cell predicts it, then compares with the three wafers measured from it."""),
code("""p = predict(fit, OX_NEW_MIN)
print(f"predicted thickness at {OX_NEW_MIN} min: {p['x'][0]:.2f} nm")
print(f"  95% confidence band (the mean):         {p['conf_lo'][0]:.2f} to {p['conf_hi'][0]:.2f} nm   (half-width {1.96 * p['se_mean'][0]:.2f})")
print(f"  95% prediction band (one measurement):  {p['pred_lo'][0]:.2f} to {p['pred_hi'][0]:.2f} nm   (half-width {1.96 * p['se_pred'][0]:.2f})")
new = runs["new"]
for x_meas in new["thickness_nm"]:
    inside = p["pred_lo"][0] <= x_meas <= p["pred_hi"][0]
    print(f"  measured wafer: {x_meas:.2f} nm, {'inside' if inside else 'OUTSIDE'} the prediction band")"""),
md("""Write down the half-width of the prediction band, to 2 decimals: the lab check asks for it.

**What to notice.** The confidence band is narrow: with 12 runs, the model's mean at 270 minutes is known to a few tenths of a nanometre. The prediction band is about three times wider, because any single new measurement adds the ellipsometer's 0.5 nm of noise, which no amount of calibration removes. To judge one measured wafer, use the prediction band.

**What the bands cover.** Both are approximate **pointwise** 95% intervals: each holds at one time, not for the whole curve at once. They assume that the Deal-Grove model is right for these runs, that the starting oxide is known and that σ is the only measurement noise. They do not cover everything that varies in a real furnace, such as differences from wafer to wafer or from run to run beyond the stated noise."""),
md("""## Step 7: the bands over time, and extrapolation

The cell draws both bands from 0 to 24 hours. The calibration runs cover 30 to 480 minutes."""),
code("""tt = np.linspace(0, 1440, 400)
pb = predict(fit, tt)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.fill_between(tt / 60, pb["pred_lo"], pb["pred_hi"], alpha=0.25, label="95% prediction band")
ax.fill_between(tt / 60, pb["conf_lo"], pb["conf_hi"], alpha=0.5, label="95% confidence band")
ax.plot(tt / 60, pb["x"], lw=1.5, label="calibrated model")
ax.plot(cal["time_min"] / 60, cal["thickness_nm"], "o", ms=4, label="calibration runs")
ax.axvspan(fit["t_min"] / 60, fit["t_max"] / 60, color="gray", alpha=0.08)
ax.set_xlabel("oxidation time (hours)")
ax.set_ylabel("oxide thickness (nm)")
ax.legend(loc="lower right")
plt.show()

for tm in [270, 480, 960, 1440]:
    q = predict(fit, tm)
    print(f"t = {tm:5d} min: x = {q['x'][0]:6.1f} nm, confidence half-width {1.96 * q['se_mean'][0]:.2f} nm, "
          f"{'inside' if q['inside'][0] else 'OUTSIDE'} the calibrated range")"""),
md("""**What to notice.** Inside the calibrated range the confidence band stays narrow; beyond 480 minutes it widens steadily, because the data no longer pin the curve down. And the band is the optimistic part: it assumes the model itself is still right out there. Nothing in 30 to 480 minutes tests that. **Extrapolation** is a claim the data do not support, however narrow the band looks."""),
md("""## Step 8: a challenging condition, built in on purpose

Real dry oxidation is faster than the Deal-Grove model predicts while the oxide is thinner than about 20 nm (the "thin-oxide regime", studied by Massoud, Plummer and Irene in 1985). The course's simulator includes such extra early growth. **This mismatch is deliberate**: it is the challenging condition of this lab.

The thin-oxide runs are short runs of 10 to 40 minutes on **bare** wafers, with only 1.5 nm of native oxide. The cell predicts them with your calibrated model, using x_i = 1.5 nm, and computes for each wafer z = (measured − predicted) / se_pred: how many prediction standard errors it is off. For a model that is right, |z| is rarely above 2."""),
code("""thin = runs["thin"]
fit_thin = dict(fit, xi=THIN_XI_NM)          # the same calibrated B and B/A, for a bare wafer
pt = predict(fit_thin, thin["time_min"])
z = (thin["thickness_nm"].to_numpy() - pt["x"]) / pt["se_pred"]
print(pd.DataFrame({"time_min": thin["time_min"], "measured_nm": thin["thickness_nm"], "predicted_nm": pt["x"].round(2),
                    "pred_band_nm": [f"{a:.2f} to {b:.2f}" for a, b in zip(pt["pred_lo"], pt["pred_hi"])], "z": z.round(1)}).to_string(index=False))

tz = np.linspace(0, 60, 200)
fig, ax = plt.subplots(figsize=(8, 3.5))
pz = predict(fit_thin, tz)
ax.fill_between(tz, pz["pred_lo"], pz["pred_hi"], alpha=0.25, label="95% prediction band")
ax.plot(tz, pz["x"], label="calibrated Deal-Grove, bare wafer")
ax.plot(thin["time_min"], thin["thickness_nm"], "s", color="C3", label="measured thin-oxide runs")
ax.set_xlabel("oxidation time (min)")
ax.set_ylabel("oxide thickness (nm)")
ax.legend()
plt.show()"""),
md("""**What to notice.** Every thin-oxide wafer is several nanometres thicker than predicted, many prediction standard errors outside the band. The bands were honest about noise and parameter uncertainty, but they cannot warn you about a model that is wrong for the condition. The calibration runs started at 25 nm, so they never tested the thin regime.

State the model's range of validity with it: **this calibration applies to dry oxidation at 1000 °C, from a starting oxide of about 25 nm, for 30 to 480 minutes.** Outside that, a twin should flag its prediction as unsupported, or use a model built for the condition."""),
md("""## Step 9: your own furnace

Open the **Module 3 Lab Check** in Canvas. Question 1 gives you a **KEY**, a whole number from 100 to 999. Type it below in place of `None`. Then calibrate the model on **your** furnace: one call to `fit_deal_grove()` on your calibration runs, as in Step 5, with `OX_SIGMA_NM` and `OX_XI_NM`."""),
code("""KEY = None      # <- your KEY from Question 1 of the lab check, for example KEY = 417

mine = oxidation_runs(KEY)
cal_mine = mine["calibration"]

# YOUR CODE HERE: the same call as in Step 5, on cal_mine
fit_mine = None"""),
code("""if fit_mine is None:
    print("Complete the call to fit_deal_grove in the cell above, then run both cells again.")
else:
    pm = predict(fit_mine, OX_NEW_MIN)
    print(f"B = {fit_mine['B']:.2f} +/- {fit_mine['se_B']:.2f} nm^2/min, B/A = {fit_mine['BA']:.4f} +/- {fit_mine['se_BA']:.4f} nm/min")
    print(f"chi^2 = {fit_mine['chi2']:.2f} for {fit_mine['dof']} degrees of freedom")
    print(f"predicted new run ({OX_NEW_MIN} min): {pm['x'][0]:.3f} nm, prediction band {pm['pred_lo'][0]:.2f} to {pm['pred_hi'][0]:.2f} nm")
    print("measured wafers of your new run:", ", ".join(f"{v:.2f}" for v in mine["new"]["thickness_nm"]))"""),
md("""## Step 10: check your work

The cell below tests your `dg_tau()` and `dg_thickness()` on cases you cannot see, and checks that `fit_mine` is the fit of your own 12 calibration runs with the stated starting oxide and σ. If all pass, it prints your **verification value**: a code made from your KEY and the thickness your calibrated model predicts for the new run of 270 minutes, in nm, computed with the course's reference functions. Only a passing check prints it, and it is **not** the number Step 9 printed. Type it into Question 1 of the lab check.

If it says **NOT PASSED**, read the message: it tells you what to fix."""),
code("""# The course's check code is in dt_m3_check.py, next to this notebook.
# It is kept out of the notebook so that the course's reference solutions are not on screen.
from dt_m3_check import check_dt_module3

passed, messages, value = check_dt_module3(dg_tau, dg_thickness, fit_mine, KEY)
for line in messages:
    print(line)
if passed:
    print()
    print(f"Verification value for KEY = {KEY}: {value:.3f}")
    print("Type this number into Question 1 of the Module 3 Lab Check.")"""),
md("""## What this means for the twin

The twin now has a physics model of the furnace with stated uncertainty: a predicted thickness, a band for the mean and a band for a new measurement, and a stated range in which they hold. For the example furnace, the new run's three wafers fell inside the prediction band. For your own furnace one of them may fall outside: a 95% band misses about one measurement in twenty, so a single wafer outside it does not, by itself, show that the model has failed. Outside the stated range, on thin oxides, the model failed by far more than its bands, and nothing in the calibration could have shown that. Module 4 adds a learned correction to this model; Module 5 keeps it in step with a furnace that drifts.

**Claim only what the data support.** On simulated data, a good fit shows that the method recovers the course's simulator. It does not show that a real furnace follows Deal-Grove.

To keep a copy of this notebook, choose **File > Download**."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"m3-{i:02d}"
text = nbf.writes(nb)
assert "—" not in text, "em dash in notebook text"
(OUT / NAME).write_text(text)
print("wrote", OUT / NAME, len(cells), "cells")

# ---------------------------------------------------------------- optional Google Colab copy
COLAB = ROOT / "colab" / "m3"
COLAB.mkdir(parents=True, exist_ok=True)
CNAME = "DT-M3-lab-oxidation-model-colab.ipynb"
RAW = "https://raw.githubusercontent.com/CSU-Physics/digital-twins-notebooks/main/content/m3/"
colab_cells = [cells[0],
               md("""## Colab only: get the course files

You are running the **Google Colab** copy of this lab. Colab needs three course files next to the notebook: `furnace.py`, the furnace simulator, `oxide_tools.py`, the fitting helpers, and `dt_m3_check.py`, the check code. Run the cell below once each time you open the notebook; it downloads them from the course repository. Everything else is the same as in the Canvas version, and your verification value is the same."""),
               code(f"""import urllib.request

BASE = "{RAW}"
for name in ["furnace.py", "oxide_tools.py", "dt_m3_check.py"]:
    urllib.request.urlretrieve(BASE + name, name)
    print("downloaded", name)""")] + cells[1:]
cnb = nbf.v4.new_notebook(cells=colab_cells, metadata={"colab": {"provenance": []},
                                                       "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                       "language_info": {"name": "python"}})
for i, c in enumerate(cnb.cells):
    c["id"] = f"m3c-{i:02d}"
ctext = nbf.writes(cnb)
assert "—" not in ctext
(COLAB / CNAME).write_text(ctext)
print("wrote", COLAB / CNAME, len(colab_cells), "cells")
