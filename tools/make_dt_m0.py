"""Build the Module 0 browser notebook (Digital Twins for Semiconductor Manufacturing) and copy its check file.

The check code stays in content/m0/dt_m0_check.py (served next to the notebook by JupyterLite); the check cell
only imports it, so the course's reference solutions are not on screen."""
import shutil
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "m0"
NAME = "DT-M0-lab-furnace-log.ipynb"
VERSION = "2026-10-20"
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}

shutil.copyfile(ROOT / "checks" / "dt_m0_check.py", OUT / "dt_m0_check.py")


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 0 lab: notebooks, NumPy and pandas on a furnace log

**Digital Twins for Semiconductor Manufacturing · Module 0 · about 40 minutes** · notebook version {VERSION}

A digital twin lives on data: readings that arrive from a tool, are cleaned, summarized and compared with a model. Every lab in this course works with one **simulated oxidation furnace**. In this warm-up you meet its first data set, a log of the zone temperature during one furnace run, and practise the Python you will use all course.

In this lab you:

1. work with a NumPy array of readings: indexing, arithmetic, mean and standard deviation;
2. load a furnace log into a pandas table, find the missing readings and plot the run;
3. write `mean_and_sd()`, which summarizes readings that include missing values;
4. find and fix a planted bug in `moving_average()`;
5. analyse **your own** furnace log, set by your KEY, and get your verification value.

All the data in this course are **simulated**: they stand in for a real furnace so that every result can be checked. The **Module 0 lab check** in Canvas asks about your results, so keep this lab open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser. It prints the versions of the libraries the course uses and of `furnace.py`, the course's furnace simulator, which sits next to this notebook."""),
code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import furnace
from furnace import furnace_log

print("Notebook version {VERSION}")
print("NumPy", np.__version__, "| pandas", pd.__version__, "| furnace.py", furnace.__version__)"""),
md("""## Step 1: an array of readings

A thermocouple in the furnace's centre zone gave these eight readings, in °C, once every 10 seconds while the zone was held at 1000 °C. A NumPy **array** holds them; Python counts positions from 0.

Run the cell, then read what each line printed."""),
code("""readings = np.array([999.6, 1000.4, 1000.1, 999.8, 1000.3, 999.9, 1000.2, 999.7])

print("number of readings:", readings.size)
print("first reading:", readings[0], "| last reading:", readings[-1])
print("readings 2 to 4:", readings[1:4])               # positions 1, 2 and 3
print("in kelvin:", readings + 273.15)                 # arithmetic acts on every element
print("deviation from 1000 C:", np.round(readings - 1000, 2))
print("mean:", readings.mean())
print("sample standard deviation (ddof=1):", round(readings.std(ddof=1), 3))
print("population standard deviation (ddof=0):", round(readings.std(), 3))"""),
md("""**Two standard deviations.** `np.std` divides by n by default (`ddof=0`, the population formula). When the readings are a **sample** of a process that could give more readings, the course uses the sample standard deviation, which divides by n − 1 (`ddof=1`). It is slightly larger; the difference matters most for small samples. Write down the sample standard deviation to 3 decimals: the lab check asks for it."""),
md("""## Step 2: a furnace log in a pandas table

`furnace_log(key)` returns the simulated log of one furnace run as a pandas **DataFrame**: a table with named columns and one row every 10 seconds for 30 minutes.

| Column | Meaning |
|---|---|
| `time_s` | seconds since the wafers were pushed into the furnace |
| `run_id` | the run's name |
| `setpoint_C` | the temperature the controller is holding, °C |
| `zone_temp_C` | the thermocouple reading, °C; **NaN** where the logger missed a reading |

The cell loads an example run (key 2026) that everyone shares. **NaN** ("not a number") marks a missing value; most NumPy and pandas functions need to be told what to do with it."""),
code("""example = furnace_log(2026)
print("rows and columns:", example.shape)
print(example.head())
print()
print("missing readings:", int(example["zone_temp_C"].isna().sum()))
print(example[example["zone_temp_C"].isna()])"""),
md("""Write down how many readings are missing in the example log: the lab check asks for it.

Two useful tools: `table["column"]` picks one column, and `table[condition]` keeps the rows where the condition is true. The cell below keeps the **hold period**, from 20 minutes (1,200 s) on, when the zone has reached its setpoint."""),
code("""hold = example[example["time_s"] >= 1200]
print("rows in the hold period:", len(hold))
print(hold["zone_temp_C"].describe().round(3))"""),
md("""## Step 3: plot the run

A plot shows at once what a table hides. Every plot in this course has **axis labels with units**. Run the cell: the zone warms from 800 °C towards the setpoint, then holds there with small fluctuations. Missing readings appear as gaps in the line."""),
code("""fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(example["time_s"] / 60, example["zone_temp_C"], lw=1, label="zone reading")
ax.axhline(1000, color="gray", ls="--", lw=1, label="setpoint")
ax.axvline(20, color="gray", ls=":", lw=1)
ax.set_xlabel("time since push (minutes)")
ax.set_ylabel("zone temperature (°C)")
ax.set_title("Example furnace run (simulated)")
ax.legend()
plt.show()"""),
md("""**What to notice.** The warm-up is smooth and fast at first, then slows as the zone approaches 1000 °C; Module 1 turns this shape into the first model of a twin. After 20 minutes the readings scatter by a few tenths of a degree around a level close to, but not exactly at, the setpoint."""),
md("""## Step 4: your task, `mean_and_sd()`

`readings.mean()` returns `nan` if any reading is missing. **Your task:** write `mean_and_sd(values)`. It takes an array (or a list, or a pandas column) of readings, leaves out the missing ones, and returns two numbers: the mean and the **sample** standard deviation.

Hints: `np.asarray(values, dtype=float)` turns the input into an array; `np.isnan(v)` is True where a value is missing, so `v[~np.isnan(v)]` keeps the rest. `np.nanmean` and `np.nanstd(..., ddof=1)` are another way."""),
code("""def mean_and_sd(values):
    \"\"\"Mean and sample standard deviation (ddof=1) of the readings, ignoring missing values (NaN).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete mean_and_sd() first.")"""),
md("""Test it on the hold period of the example run. The mean should be close to, but not exactly, 1000 °C."""),
code("""m, s = mean_and_sd(hold["zone_temp_C"])
print(f"hold period: mean {m:.3f} C, sample standard deviation {s:.3f} C")"""),
md("""## Step 5: find and fix a planted bug

A **moving average** replaces each reading by the average of a short run of neighbouring readings. It smooths out the scatter of the thermocouple, which is one way a twin separates the furnace's real behaviour from sensor noise. With a window of 5, the first smoothed value is the average of readings 0 to 4, the next of readings 1 to 5, and so on, so n readings give n − 4 smoothed values.

The function below has **a bug**. Run the two cells: the plot shows the smoothed hold period far from the readings. Find the mistake, fix it in the function, and run both cells again until the smoothed line runs through the middle of the readings."""),
code("""def moving_average(x, window):
    \"\"\"Average of each run of `window` neighbouring readings (only the full windows, so len(x) - window + 1 values).\"\"\"
    kernel = np.ones(window)
    return np.convolve(np.asarray(x, dtype=float), kernel, mode="valid")"""),
code("""h = hold["zone_temp_C"].dropna().to_numpy()      # the hold-period readings, missing ones dropped
smooth = moving_average(h, 5)
print("readings:", h.size, "| smoothed values:", smooth.size)
print("first readings:", np.round(h[:5], 2), "| first smoothed value:", round(float(smooth[0]), 3))

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(np.arange(h.size), h, ".", ms=4, label="readings")
ax.plot(np.arange(2, 2 + smooth.size), smooth, lw=2, label="5-point moving average")
ax.set_xlabel("reading number in the hold period")
ax.set_ylabel("zone temperature (°C)")
ax.legend()
plt.show()"""),
md("""When the line runs through the readings, compare the scatter before and after smoothing:"""),
code("""print(f"standard deviation of the readings:       {np.std(h, ddof=1):.3f} C")
print(f"standard deviation of the moving average:  {np.std(smooth, ddof=1):.3f} C")
print(f"ratio: {np.std(h, ddof=1) / np.std(smooth, ddof=1):.2f}   (for independent noise, expect about sqrt(5) = {np.sqrt(5):.2f})")"""),
md("""**What to notice.** Averaging 5 readings with independent noise divides the standard deviation by about √5 ≈ 2.2. Your ratio is close to that, but not exactly equal: a standard deviation estimated from about 60 readings has its own scatter. Smoothing has a price too. During the warm-up, a moving average lags behind the real temperature, which is why this lab smooths only the hold period. Module 5 meets the same trade-off in the EWMA filter."""),
md("""## Step 6: your own furnace run

Open the **Module 0 Lab Check** in Canvas. Question 1 gives you a **KEY**, a whole number from 100 to 999. Type it below in place of `None`, and run the cell. Your log is different from everyone else's: a different offset, noise level and set of missing readings."""),
code("""KEY = None      # <- your KEY from Question 1 of the lab check, for example KEY = 417

log = furnace_log(KEY)
mine = log[log["time_s"] >= 1200]["zone_temp_C"].dropna().to_numpy()
m, s = mean_and_sd(mine)
print(f"run {log['run_id'][0]}: missing readings {int(log['zone_temp_C'].isna().sum())}, hold-period readings {mine.size}")
print(f"hold period: mean {m:.3f} C, sample standard deviation {s:.3f} C")
print(f"5-point moving average: standard deviation {np.std(moving_average(mine, 5), ddof=1):.3f} C")"""),
md("""## Step 7: check your work

The cell below tests your `mean_and_sd()` and `moving_average()` on cases you cannot see. If both pass, it prints your **verification value**: the standard deviation of the 5-point moving average of your hold-period readings, computed with the course's reference functions. Type it into Question 1 of the lab check.

If it says **NOT PASSED**, read the message: it tells you what to fix. Fix your function, run its cell again, then run this cell again."""),
code("""# The course's check code is in dt_m0_check.py, next to this notebook.
# It is kept out of the notebook so that the course's reference solutions are not on screen.
# You may open the file to see how your functions are tested, but write your own functions first.
from dt_m0_check import check_dt_module0

passed, messages, value = check_dt_module0(mean_and_sd, moving_average, KEY)
for line in messages:
    print(line)
if passed:
    print()
    print(f"Verification value for KEY = {KEY}: {value:.3f}")
    print("Type this number into Question 1 of the Module 0 Lab Check.")"""),
md("""## What this has to do with a digital twin

You have just done, by hand, the first job of every twin: take the readings a tool sends, deal with missing values, and summarize what the tool is doing. In Module 1 you will fit a model to the warm-up curve and let it update itself as each new reading arrives, which is the first step from a model to a twin.

To keep a copy of this notebook, choose **File > Download**."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"m0-{i:02d}"
text = nbf.writes(nb)
assert "—" not in text, "em dash in notebook text"
(OUT / NAME).write_text(text)
print("wrote", OUT / NAME, len(cells), "cells")

# ---------------------------------------------------------------- optional Google Colab copy
# Same cells, plus one first code cell that downloads furnace.py and dt_m0_check.py from the repository,
# because Colab opens the notebook on its own, without the files that JupyterLite serves next to it.
COLAB = ROOT / "colab" / "m0"
COLAB.mkdir(parents=True, exist_ok=True)
CNAME = "DT-M0-lab-furnace-log-colab.ipynb"
RAW = "https://raw.githubusercontent.com/CSU-Physics/digital-twins-notebooks/main/content/m0/"
colab_cells = [cells[0],
               md("""## Colab only: get the course files

You are running the **Google Colab** copy of this lab. Colab needs two course files next to the notebook: `furnace.py`, the furnace simulator, and `dt_m0_check.py`, the check code. Run the cell below once each time you open the notebook; it downloads them from the course repository. Everything else is the same as in the Canvas version, and your verification value is the same."""),
               code(f"""import urllib.request

BASE = "{RAW}"
for name in ["furnace.py", "dt_m0_check.py"]:
    urllib.request.urlretrieve(BASE + name, name)
    print("downloaded", name)""")] + cells[1:]
cnb = nbf.v4.new_notebook(cells=colab_cells, metadata={"colab": {"provenance": []},
                                                       "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                       "language_info": {"name": "python"}})
for i, c in enumerate(cnb.cells):
    c["id"] = f"m0c-{i:02d}"
ctext = nbf.writes(cnb)
assert "—" not in ctext
(COLAB / CNAME).write_text(ctext)
print("wrote", COLAB / CNAME, len(colab_cells), "cells")
