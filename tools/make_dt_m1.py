"""Build the Module 1 browser notebook (Digital Twins for Semiconductor Manufacturing), its Colab copy, and copy its files.

content/m1 holds furnace.py (1.1.0), twin_tools.py (supplied helpers) and dt_m1_check.py (copied from checks/).
The check cell only imports the check file, so the course's reference solutions are not on screen."""
import shutil
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "m1"
NAME = "DT-M1-lab-first-twin.ipynb"
VERSION = "2026-10-25"
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}

shutil.copyfile(ROOT / "checks" / "dt_m1_check.py", OUT / "dt_m1_check.py")


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 1 lab: a first twin of the furnace

**Digital Twins for Semiconductor Manufacturing · Module 1 · about 60 minutes** · notebook version {VERSION}

A model of a furnace becomes part of a twin when real readings keep it in step with the furnace. In this lab you build the first layer of the course's furnace twin:

1. use a **first-order model** of the zone temperature, which warms toward its setpoint with a time constant τ;
2. **fit τ** to the first 10 minutes of a stream of readings;
3. watch the model on its own fall behind when something happens that it does not know about;
4. write `update_prediction()`, which corrects the twin with each new reading, and compare the two;
5. see how the error grows when the twin is updated less often;
6. do the same for **your own** stream, set by your KEY, and get your verification value.

All the data are **simulated**. The **Module 1 Lab Check** in Canvas asks about your results, so keep this lab open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

The first time, Python can take up to a minute to start, and SciPy, which the fitting function uses, adds a few seconds. The cell prints the versions of the libraries and of the two course files next to this notebook: `furnace.py`, the furnace simulator, and `twin_tools.py`, the helpers for this lab."""),
code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import furnace, twin_tools
from furnace import temperature_stream, stream_truth, LOAD_C, SETPOINT_C
from twin_tools import first_order, fit_tau, run_twin, rms

print("Notebook version {VERSION}")
print("NumPy", np.__version__, "| pandas", pd.__version__, "| SciPy", scipy.__version__,
      "| furnace.py", furnace.__version__, "| twin_tools.py", twin_tools.__version__)"""),
md("""## Step 1: a first-order model of the zone

When the wafers are pushed in, the zone is at T0 = 800 °C. The controller drives it toward the setpoint T_set = 1000 °C, and the gap to the setpoint shrinks by the same fraction every second. That is a **first-order system**:

T(t) = T_set − (T_set − T0) · e^(−t/τ)

The **time constant** τ, in seconds, sets the speed. After one τ the zone has covered 63% of the way, after 3τ about 95%. `first_order(t, tau)` computes T(t). Run the cell for τ = 150 s."""),
code("""tau_example = 150.0
for t in [0, 150, 300, 450]:
    print(f"t = {t:3d} s: T = {first_order(t, tau_example):.2f} C")"""),
md("""Write down the temperature at 300 s: the lab check asks for it. Check the 63% rule yourself: at t = τ = 150 s the zone has covered (926.42 − 800) / 200 = 63% of the way."""),
md("""## Step 2: a stream of readings

`temperature_stream(key)` returns the simulated readings of one furnace run, one every 10 seconds for 30 minutes, in the same columns as the Module 0 log. The real τ of each run is different, and some time after 14 minutes **something happens that the model does not know about**. The cell loads the example run (key 2026) and plots it."""),
code("""stream = temperature_stream(2026)
print("rows:", len(stream), "| missing readings:", int(stream["zone_temp_C"].isna().sum()))

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(stream["time_s"] / 60, stream["zone_temp_C"], ".", ms=3)
ax.axhline(SETPOINT_C, color="gray", ls="--", lw=1)
ax.set_xlabel("time since push (minutes)")
ax.set_ylabel("zone temperature (°C)")
ax.set_title("Example stream (simulated)")
plt.show()"""),
md("""## Step 3: your task, fit τ

`fit_tau(t, y, T0, T_set)` finds the τ that makes the first-order model closest to the readings (a least-squares fit with SciPy's `curve_fit`; missing readings are left out). It returns two numbers: τ and its **standard error**, which says how precisely the readings pin τ down.

**Your task:** fit τ to the first 10 minutes of the stream, where nothing unusual happens yet. Replace the `None, None` line with one call to `fit_tau`, using the times and readings of `warm`, `T0=LOAD_C` and `T_set=SETPOINT_C`."""),
code("""warm = stream[stream["time_s"] < 600]          # the first 10 minutes: 60 rows

# YOUR CODE HERE: one call to fit_tau(...) that returns (tau, standard error)
tau_hat, tau_sd = None, None"""),
code("""if tau_hat is None:
    print("Complete the call to fit_tau above, then run both cells again.")
else:
    print(f"fitted tau = {tau_hat:.2f} s, standard error {tau_sd:.2f} s")
    print(f"the simulation's real tau = {stream_truth(2026)['tau_s']:.2f} s   (a real furnace never tells you this)")"""),
md("""Write down the fitted τ to 1 decimal: the lab check asks for it.

**Claim only what the data support.** For this example stream the fitted τ happens to lie within one standard error of the simulation's real τ. Other streams need not: a standard error describes the typical spread of the fit, so for about a third of the KEYs the fitted τ lands further away than one standard error, and that is no sign of a mistake. A good fit shows that the fitting works on the course's simulator, which is a first-order system with independent noise. It does not show that a real furnace is a first-order system: with real data, you would check the residuals and test the model on data it was not fitted to."""),
md("""## Step 4: the model on its own

With τ fitted, the model can predict the whole run without looking at another reading. In the language of Module 1, this is a **digital model**: nothing flows from the furnace into it after the fit. `run_twin(stream, tau, gain, update=None)` runs it and records, for every reading, the prediction made just before it arrived and the error, reading minus prediction.

The cell reports the **RMS error** (root-mean-square error): the square root of the mean of the squared errors, where each error is a reading minus the prediction made before that reading arrived (and before it corrected the twin). Missing readings are left out, and the result is in °C. For example, errors of +1, −2 and +2 °C give √((1 + 4 + 4)/3) = 1.73 °C."""),
code("""assert tau_hat is not None, "Complete Step 3 first: fit tau, then run this cell again."
model_only = run_twin(stream, tau_hat, gain=0.5, update=None)
late = model_only["time_s"] >= 600
print(f"model alone: RMS error before 600 s {rms(model_only['error'][~late]):.3f} C, from 600 s on {rms(model_only['error'][late]):.3f} C")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(stream["time_s"] / 60, stream["zone_temp_C"], ".", ms=3, label="readings")
ax.plot(model_only["time_s"] / 60, model_only["predicted"], lw=2, label="model alone")
ax.set_ylim(975, 1003)
ax.set_xlabel("time since push (minutes)")
ax.set_ylabel("zone temperature (°C)")
ax.legend()
plt.show()"""),
md("""**What to notice.** For the first 10 minutes the model's error is about the size of the thermocouple noise. Then the zone dips by several degrees and recovers. The model cannot see the dip, so its error grows to many times the noise, and nothing in the model will ever notice."""),
md("""## Step 5: your task, `update_prediction()`

A twin keeps its prediction in step with the furnace. Before each reading arrives, the twin predicts it with the model; when the reading arrives, the twin **corrects** its prediction:

T_new = T_pred + gain · (reading − T_pred)

The **gain**, between 0 and 1, is the weight of the new reading: gain 0 ignores it, gain 1 jumps to it. If the reading is missing (NaN), there is nothing to learn from it, so the prediction stays as it is.

**Your task:** write `update_prediction(T_pred, reading, gain)`. It returns one number. Hint: `np.isnan(reading)` is True when the reading is missing."""),
code("""def update_prediction(T_pred, reading, gain):
    \"\"\"Correct the twin's predicted temperature with a new reading; leave it unchanged if the reading is missing (NaN).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete update_prediction() first.")"""),
code("""print(update_prediction(900.0, 910.0, 0.5), "(should be 905.0)")
print(update_prediction(900.0, float("nan"), 0.5), "(should be 900.0)")

synced = run_twin(stream, tau_hat, gain=0.5, update=update_prediction)
if synced["predicted"].isna().any():
    print("The twin's prediction became nan: check what update_prediction() returns for a missing reading.")
print(f"model alone:  RMS error from 600 s on {rms(model_only['error'][late]):.3f} C")
print(f"synced twin:  RMS error from 600 s on {rms(synced['error'][late]):.3f} C   (gain 0.5, every reading)")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(model_only["time_s"] / 60, model_only["error"], lw=1, label="model alone")
ax.plot(synced["time_s"] / 60, synced["error"], lw=1, label="synced twin")
ax.axhline(0, color="gray", lw=1)
ax.set_xlabel("time since push (minutes)")
ax.set_ylabel("error: reading − prediction (°C)")
ax.legend()
plt.show()"""),
md("""**What to notice.** The synced twin's error stays close to the noise through the dip, because every reading pulls its state back toward the furnace. Before the dip, when the model is right, correcting with noisy readings makes the error slightly larger, not smaller. The data flow in one direction only: from the furnace to the twin. Nothing in this lab sends a decision back to the furnace."""),
md("""## Step 6: how often should the twin update?

A twin can only be as current as its last update. The cell updates the twin with every reading (every 10 s), then with every 3rd, 6th and 18th reading (every 30 s, 1 minute and 3 minutes), and prints the RMS error from 600 s on. Between updates the twin runs on the model alone."""),
code("""for every in [1, 3, 6, 18]:
    run = run_twin(stream, tau_hat, gain=0.5, update=update_prediction, every=every)
    print(f"update every {every * 10:4d} s: RMS error from 600 s on {rms(run['error'][late]):.3f} C")"""),
md("""**What to notice.** The error grows as the updates get further apart: between updates, the twin drifts with the model, and the model does not know about the dip. How often a twin must update depends on how fast the real system changes and how large an error its use can accept. The Digital Twin Consortium's definition asks for a "specified frequency" for exactly this reason."""),
md("""## Step 7: your own stream

Open the **Module 1 Lab Check** in Canvas. Question 1 gives you a **KEY**, a whole number from 100 to 999. Type it below in place of `None`. Then fit τ to the first 10 minutes of **your** stream: one call to `fit_tau`, as in Step 3, on `warm_mine`."""),
code("""KEY = None      # <- your KEY from Question 1 of the lab check, for example KEY = 417

mine = temperature_stream(KEY)
warm_mine = mine[mine["time_s"] < 600]

# YOUR CODE HERE: the same call as in Step 3, on warm_mine
tau_mine, sd_mine = None, None"""),
code("""if tau_mine is None:
    print("Complete the call to fit_tau in the cell above, then run both cells again.")
else:
    mine_synced = run_twin(mine, tau_mine, gain=0.5, update=update_prediction)
    late_mine = mine_synced["time_s"] >= 600
    print(f"run {mine['run_id'][0]}: fitted tau = {tau_mine:.2f} s (standard error {sd_mine:.2f} s)")
    print(f"synced twin: RMS error from 600 s on {rms(mine_synced['error'][late_mine]):.3f} C")"""),
md("""## Step 8: check your work

The cell below tests your `update_prediction()` on cases you cannot see, including missing readings, and checks that `tau_mine` is the fit of your own stream's first 10 minutes. If both pass, it prints your **verification value**: the RMS error, from 600 s on, of the twin that is synced with every reading (gain 0.5) on your stream, computed with the course's reference functions. Type it into Question 1 of the lab check.

If it says **NOT PASSED**, read the message: it tells you what to fix."""),
code("""# The course's check code is in dt_m1_check.py, next to this notebook.
# It is kept out of the notebook so that the course's reference solutions are not on screen.
from dt_m1_check import check_dt_module1

passed, messages, value = check_dt_module1(update_prediction, tau_mine, KEY)
for line in messages:
    print(line)
if passed:
    print()
    print(f"Verification value for KEY = {KEY}: {value:.3f}")
    print("Type this number into Question 1 of the Module 1 Lab Check.")"""),
md("""## What kind of twin did you build?

Your prototype takes data from the furnace automatically and corrects itself with every reading, but it sends nothing back: no person is shown a decision and no controller acts on it. In the classification of Kritzinger and co-authors that is a **digital shadow**, and in this course's terms it reaches the **first level of coupling**: the twin updates from observations. Module 5 adds the next levels, where the twin's predictions support an engineer's decision and then drive an automatic recipe update.

To keep a copy of this notebook, choose **File > Download**."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"m1-{i:02d}"
text = nbf.writes(nb)
assert "—" not in text, "em dash in notebook text"
(OUT / NAME).write_text(text)
print("wrote", OUT / NAME, len(cells), "cells")

# ---------------------------------------------------------------- optional Google Colab copy
COLAB = ROOT / "colab" / "m1"
COLAB.mkdir(parents=True, exist_ok=True)
CNAME = "DT-M1-lab-first-twin-colab.ipynb"
RAW = "https://raw.githubusercontent.com/CSU-Physics/digital-twins-notebooks/main/content/m1/"
colab_cells = [cells[0],
               md("""## Colab only: get the course files

You are running the **Google Colab** copy of this lab. Colab needs three course files next to the notebook: `furnace.py`, the furnace simulator, `twin_tools.py`, the helpers, and `dt_m1_check.py`, the check code. Run the cell below once each time you open the notebook; it downloads them from the course repository. Everything else is the same as in the Canvas version, and your verification value is the same."""),
               code(f"""import urllib.request

BASE = "{RAW}"
for name in ["furnace.py", "twin_tools.py", "dt_m1_check.py"]:
    urllib.request.urlretrieve(BASE + name, name)
    print("downloaded", name)""")] + cells[1:]
cnb = nbf.v4.new_notebook(cells=colab_cells, metadata={"colab": {"provenance": []},
                                                       "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                       "language_info": {"name": "python"}})
for i, c in enumerate(cnb.cells):
    c["id"] = f"m1c-{i:02d}"
ctext = nbf.writes(cnb)
assert "—" not in ctext
(COLAB / CNAME).write_text(ctext)
print("wrote", COLAB / CNAME, len(colab_cells), "cells")
