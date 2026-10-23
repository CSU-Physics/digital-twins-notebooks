"""Build the Module 6 browser notebook (Digital Twins for Semiconductor Manufacturing), its Colab copy, and copy its files.

content/m6 holds line.py (the SimPy line model), line_tools.py (supplied helpers), dt_m6_check.py (copied from checks/) and the
SimPy 4.1.2 wheel (MIT license) that line.py imports. The check cell only imports the check file, so the course's reference
solutions are not on screen."""
import shutil
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "m6"
NAME = "DT-M6-lab-toolset-twin.ipynb"
VERSION = "2026-10-25"
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}

shutil.copyfile(ROOT / "checks" / "dt_m6_check.py", OUT / "dt_m6_check.py")


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 6 lab: a toolset twin of furnace, lithography and etch

**Digital Twins for Semiconductor Manufacturing · Module 6 · about 90 minutes** · notebook version {VERSION}

So far the twin has described one furnace: what it grows (Module 3), what a wafer measures (Module 4) and how it drifts (Module 5). A fab does not run one furnace. Lots of wafers queue for tools, and much of a lot's **cycle time** is waiting: nearly half, in this lab's line. In this lab the furnace joins a small line, **furnace, lithography and etch**, written as a discrete-event simulation in **SimPy**, and you use it to answer questions a tool-level twin cannot: where do lots wait, what does a change to one tool do to the whole line, and what does maintenance cost in cycle time.

You write five short functions:

1. `throughput()`, the lots that leave the line per hour;
2. `mean_cycle_time()`, the mean time a lot spends in the line;
3. `time_average_wip()`, the average number of lots in the line (**work in progress**, WIP);
4. `ci95()`, a 95 % confidence interval for a mean over **replications**;
5. `calibrate_now()`, the Module 5 drift alarm as a rule that a running simulation can call.

With them you check **Little's law** on the simulation's output, choose a **warm-up**, run **20 replications**, compare two **what-if changes** by their mean and spread, and compare a **preventive** with a **predictive** maintenance rule that uses the Module 5 drift estimate.

All the data are **simulated**. The **Module 6 Lab Check** in Canvas asks about your results, so keep this lab open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

The first time, Python can take up to a minute to start. The cell prints the versions of the libraries and of the three course files next to this notebook: `line.py`, the SimPy model of the line, `line_tools.py`, the helpers for this lab, and the SimPy wheel that `line.py` imports."""),
code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import line, line_tools
from line import line_config, BASE_LINE
from line_tools import (run_replications, stage_table, stage_summary, welch, plot_run, plot_welch,
                        replication_table, maintenance_table, HOURS, WARMUP_H)

print("Notebook version {VERSION}")
print("NumPy", np.__version__, "| pandas", pd.__version__, "| SimPy", line.simpy.__version__,
      "| line.py", line.__version__, "| line_tools.py", line_tools.__version__)
print(f"each replication simulates {{HOURS:g}} h; the first {{WARMUP_H:g}} h are the warm-up")"""),
md("""## Step 1: the line

The line has three **tool groups**. A lot (25 wafers) is released at a random time, waits for a free **furnace tube**, is oxidized, waits for the **lithography** tool, is exposed, waits for an **etch chamber**, is etched, and leaves. A lot that finds every tool of a group busy waits in a queue, first in, first out. Times are in **hours**.

| setting | meaning | value |
|---|---|---|
| `arrival_rate` | lots released per hour, at random times (exponential gaps) | 0.85 |
| `furnace_tools`, `furnace_h` | tubes, and the mean time a lot spends in one: load, the 90-minute oxidation of Module 5, unload | 2 tubes, 1.8 h |
| `litho_tools`, `litho_h` | lithography tools, and the mean exposure time of a lot | 1 tool, 1.0 h |
| `etch_tools`, `etch_h` | etch chambers, and the mean etch time of a lot | 2 chambers, 1.2 h |
| `furnace_cv`, `litho_cv`, `etch_cv` | how much the process time varies from lot to lot (standard deviation divided by the mean) | 0.30, 0.50, 0.40 |

Process times vary, as they do in a fab, and arrivals are random, so two runs of the same line differ. Each run is a **replication**, started from a **seed**: the same seed gives the same run, and a different seed gives a different run of the same line.

The core of the model is a short process for each lot. This simplified excerpt shows the idea; `line.py` has the rest:

```python
def lot(i):
    tube = yield tubes.get()                  # wait for a free furnace tube
    yield env.timeout(furnace_time[i])        # oxidize
    yield tubes.put(tube)                     # free the tube
    with litho.request() as req:              # wait for the lithography tool
        yield req
        yield env.timeout(litho_time[i])      # expose
    with etch.request() as req:               # wait for an etch chamber
        yield req
        yield env.timeout(etch_time[i])       # etch
```

`run_line(cfg, seed, hours)` runs the line for `hours` simulated hours, starting **empty**, and returns a log with one row per lot. The cell runs seed 1 for 1,000 hours."""),
code("""print(pd.Series(BASE_LINE).to_string())
print()
run1 = run_replications(None, [1])[0]
print(run1)
lots = run1.lots
print(lots[["lot", "t_in", "f_start", "l_start", "e_start", "t_out"]].head(6).round(2).to_string(index=False))
print("...")
print(lots[["lot", "t_in", "f_start", "l_start", "e_start", "t_out"]].tail(3).round(2).to_string(index=False))"""),
md("""**What to notice.** Each row is one lot: `t_in` is when it entered the line, `f_start`, `l_start` and `e_start` are when it got a tool in each group, and `t_out` is when it left. A lot that is still in the line when the run ends has **NaN** for the times it has not reached; the last lots show this. Lot 2 got a furnace tube at once (`f_start` equals `t_in`), and lot 3 waited about 0.8 hours for one. The time between `t_in` and `t_out` is the lot's **cycle time**."""),
md("""## Step 2: one replication over time

The top chart shows the number of lots in the line, the **work in progress**, over the 1,000 hours. The bottom chart shows the lots waiting at each tool group. The grey band is the first 200 hours, the warm-up, which Step 7 explains."""),
code("""plot_run(run1)
plt.show()"""),
md("""**What to notice.** The line starts empty, fills in the first hours, and then moves up and down around a level of about seven lots. The queue in front of **lithography** is the longest and the most variable; the etch queue is nearly always empty. One replication is one random run of the line, so everything you measure from it, a mean cycle time or an average WIP, has a spread that you can only estimate from several replications."""),
md("""## Step 3: your task, `throughput()`

Three numbers describe a line, and they must be measured **at the same boundary** and over the **same window**. Here the boundary is the whole line, from release to the end of etch, and the window runs from `t_start` to `t_end`. The first number is the **throughput**: the number of lots that leave the line in the window, divided by the window's length.

A lot belongs to the window if it leaves in it: `t_start <= t_out < t_end`. A lot that leaves exactly at `t_start` counts; a lot that leaves exactly at `t_end` belongs to the next window. A lot with no exit time (NaN) is still in the line and has not left. Comparisons with NaN are False, so a test like `t_out >= t_start` does not count it.

**Your task:** write `throughput(t_out, t_start, t_end)`. `t_out` is a NumPy array or a pandas Series of exit times in hours. Return the lots per hour."""),
code("""def throughput(t_out, t_start, t_end):
    \"\"\"Lots per hour that leave the line in the window t_start <= t_out < t_end; NaN in t_out means the lot is still in the line.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete throughput() first.")"""),
code("""print(throughput(np.array([5.0, 12.0, 15.0, 18.0, np.nan]), 10.0, 20.0), "  (should be 0.3: three lots leave between hours 10 and 20)")
print(throughput(lots["t_out"], WARMUP_H, HOURS), "  lots per hour for seed 1, after the warm-up")"""),
md("""## Step 4: your task, `mean_cycle_time()`

The second number is the **cycle time**: the time a lot spends in the line, `t_out - t_in`, which is the waiting and the processing in all three tool groups together. The lab measures it **when lots leave**: the mean cycle time is the mean of `t_out - t_in` over the lots that leave in the window. A lot still in the line has no cycle time yet, so it is not part of the mean.

**Your task:** write `mean_cycle_time(t_in, t_out, t_start, t_end)`. Select the lots with `t_start <= t_out < t_end` and return the mean of their `t_out - t_in`, in hours. When no lot leaves in the window there is no mean: return `float("nan")`."""),
code("""def mean_cycle_time(t_in, t_out, t_start, t_end):
    \"\"\"Mean of t_out - t_in (hours) over the lots that leave in the window t_start <= t_out < t_end; NaN if none do.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete mean_cycle_time() first.")"""),
code("""print(mean_cycle_time(np.array([0.0, 3.0, 8.0]), np.array([6.0, 10.0, np.nan]), 5.0, 12.0), "  (should be 6.5: cycle times 6 and 7; the third lot is still in the line)")
print(mean_cycle_time(lots["t_in"], lots["t_out"], WARMUP_H, HOURS), "  hours for seed 1, after the warm-up")"""),
md("""## Step 5: your task, `time_average_wip()`

The third number is the **work in progress**: the number of lots in the line, averaged over the window. The number changes every time a lot enters or leaves, so the average is over **time**: a lot that is in the line for 3 hours of a 10-hour window adds 3 lot-hours, and the average WIP is the total lot-hours divided by the window's length.

For each lot, the hours inside the window run from `max(t_in, t_start)` to `min(t_out, t_end)`, and are never below 0: a lot that left before the window, or entered after it, adds nothing. A lot with NaN as exit time is still in the line, so it stays until the end of the window; `np.where(np.isnan(t_out), np.inf, t_out)` turns NaN into "never".

**Your task:** write `time_average_wip(t_in, t_out, t_start, t_end)`. Return the sum of the lots' hours inside the window, divided by `t_end - t_start`."""),
code("""def time_average_wip(t_in, t_out, t_start, t_end):
    \"\"\"Average number of lots in the line over [t_start, t_end]: the hours every lot spends inside the window, divided by its length.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete time_average_wip() first.")"""),
code("""print(time_average_wip(np.array([0.0, 4.0, 9.0]), np.array([6.0, 12.0, np.nan]), 5.0, 10.0), "  (should be 1.4: 1 + 5 + 1 lot-hours in 5 hours)")
print(time_average_wip(lots["t_in"], lots["t_out"], WARMUP_H, HOURS), "  lots in the line on average for seed 1, after the warm-up")"""),
md("""## Step 6: Little's law on the simulation's output

**Little's law** says that for a line that is in a steady state, the three numbers are tied together:

**L = λ · W**

with **L** the average number of lots in the line, **λ** the throughput and **W** the mean cycle time. It needs no assumption about how lots arrive or how long they take, and it holds for any boundary, as long as L, λ and W all describe **the same one** over the same window. The reason is that the area under the WIP curve can be counted two ways: slice by slice of time (the average number times the window), or lot by lot (the sum of the hours each lot spends in the line, which is the number of lots times their mean time).

A simulation needs a check like this one, because it shows that the log and the measuring functions agree. The cell uses your three functions on seed 1, after the warm-up, and then repeats the check for a single tool group, **lithography**: the lots there are those waiting for or using the lithography tool, `stage_table()` measures L and W at that boundary, and the throughput is the lithography group's own: the lots that leave lithography in the window, per hour (`l_end` in the lot log). Over a finite window it need not equal the whole line's throughput."""),
code("""t_out, t_in = lots["t_out"], lots["t_in"]
lam = throughput(t_out, WARMUP_H, HOURS)
W = mean_cycle_time(t_in, t_out, WARMUP_H, HOURS)
L = time_average_wip(t_in, t_out, WARMUP_H, HOURS)
print(f"whole line:  L = {L:.3f} lots   lambda = {lam:.4f} lots/h   W = {W:.3f} h")
print(f"             lambda x W = {lam * W:.3f}   ratio L / (lambda x W) = {L / (lam * W):.3f}")

st = stage_table(run1)
print()
lam_litho = throughput(lots["l_end"], WARMUP_H, HOURS)      # the lithography group's own throughput
print(f"lithography: L = {st.loc['litho', 'in_group_lots']:.3f} lots   lambda = {lam_litho:.4f} lots/h   W = {st.loc['litho', 'in_group_h']:.3f} h   "
      f"lambda x W = {lam_litho * st.loc['litho', 'in_group_h']:.3f}")
print(f"mixing boundaries: lithography's W with the whole line's L: lambda x W = {lam_litho * st.loc['litho', 'in_group_h']:.3f} against L = {L:.3f}")"""),
md("""Write down the ratio L / (λ · W) for the whole line, to 3 decimals: the lab check asks for it.

**What to notice.** The ratio is close to 1 but not exactly 1, because the window has edges: lots that are in the line at `t_start` or at `t_end` take part in L and not fully in W. Over a longer window, or over many replications, the ratio gets closer to 1; with 20 replications in Step 8 it is within half a percent. At the lithography boundary the same law holds: L and λ · W agree closely, within a few thousandths of a lot here, and the small difference again comes from the edges of the window. If you take L from one boundary and W from another, as in the last line, the two numbers differ by a factor of two, and the law has not failed: it was applied to two different systems. This is the sense of "measured at the same boundary"."""),
md("""## Step 7: warm-up

The line starts **empty**, and the first lots meet no queue, so the first hours are not typical of the running line. The statistics of a **steady state** should leave them out. How many hours? **Welch's method** (Welch, 1983) uses the average of the replications: measure the average WIP in bins of 10 hours in every replication, average the replications bin by bin, and smooth the result with a moving average. The warm-up ends where the smoothed curve levels off.

The cell runs the 20 replications of the lab (seeds 1 to 20, 1,000 hours each) and draws the plot."""),
code("""runs = run_replications()
print(len(runs), "replications, seeds", runs[0].seed, "to", runs[-1].seed)
w = welch(runs, bin_h=10, w=6)
plot_welch(w, warmup=WARMUP_H)
plt.xlim(0, 400)
plt.show()
t_bins, raw, smooth = w
print(f"average lots in the line, first 10-hour bin: {raw[0]:.1f}; smoothed curve at 100 h: {smooth[9]:.1f}, at 200 h: {smooth[19]:.1f}, at 400 h: {smooth[39]:.1f}")"""),
md("""**What to notice.** The smoothed curve rises from about 4 lots in the first bin and levels off near 7 after roughly 120 hours. The lab discards the first **200 hours**, a safe margin beyond where the curve flattens: the window for the statistics is `WARMUP_H` to `HOURS`, hours 200 to 1,000.

The warm-up bias is small compared with the randomness between replications: with 20 replications, the mean cycle time over hours 0 to 1,000 differs from the mean over hours 200 to 1,000 by less than the noise of either. Discarding the warm-up costs a fifth of the run and removes a bias you could not otherwise see. The shorter the run, the larger the share of the run that the start-up takes, and the more the warm-up matters."""),
md("""## Step 8: your task, `ci95()`, and 20 replications

Each replication gives one value of a statistic, for example its mean cycle time. The replications are independent runs of the same line, so their values are independent draws, and their **mean** estimates the line's long-run value. How good is that estimate? A **95 % confidence interval** gives a range around the mean that, over repeated experiments, contains the true mean 95 % of the time:

**mean ± t · s / √n**

with **n** replications, **s** their sample standard deviation (with n − 1 in the divisor) and **t** the 0.975 quantile of the t distribution with n − 1 degrees of freedom, `stats.t.ppf(0.975, n - 1)`: for 20 replications t = 2.09. Using the t distribution, and not 1.96, accounts for estimating the standard deviation from a small number of values.

**Your task:** write `ci95(values)`. Return the **mean** and the **half-width** of the interval, as `mean, half_width`. The values are a list, an array or a Series."""),
code("""def ci95(values):
    \"\"\"(mean, half_width) of the 95 % confidence interval of the mean of independent values: t(0.975, n - 1) * s / sqrt(n).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete ci95() first.")"""),
code("""m_test, h_test = ci95([7.2, 8.1, 6.9, 7.7, 8.4])
print(round(float(m_test), 3), round(float(h_test), 3), "  (should be 7.66 and 0.768: the mean and the half-width)")

rows = []
for r in runs:
    t_in, t_out = r.lots["t_in"], r.lots["t_out"]
    rows.append({"seed": r.seed,
                 "throughput": throughput(t_out, WARMUP_H, HOURS),
                 "cycle_time": mean_cycle_time(t_in, t_out, WARMUP_H, HOURS),
                 "wip": time_average_wip(t_in, t_out, WARMUP_H, HOURS)})
reps = pd.DataFrame(rows)
reps["littles_ratio"] = reps["wip"] / (reps["throughput"] * reps["cycle_time"])
print(reps.head(5).round(3).to_string(index=False))
print()
for col, label in [("throughput", "throughput (lots/h)"), ("cycle_time", "cycle time (h)"), ("wip", "lots in the line"), ("littles_ratio", "Little's ratio")]:
    m, h = ci95(reps[col])
    print(f"{label:22s} mean {m:.3f}  95 % interval {m - h:.3f} to {m + h:.3f}  (half-width {h:.3f}; replications range {reps[col].min():.3f} to {reps[col].max():.3f})")
print()
for n in (5, 10, 20):
    print(f"cycle time from the first {n:2d} replications: {ci95(reps['cycle_time'][:n])[0]:.3f} h, half-width {ci95(reps['cycle_time'][:n])[1]:.3f} h")"""),
md("""Write down the **mean cycle time** over the 20 replications, to 2 decimals: the lab check asks for it.

**What to notice.** The 20 replications of the same line give cycle times from about 6.1 to 9.1 hours: a **single** replication says little. The mean of 20 is 7.56 hours with a half-width of 0.41, and this interval, not the single number, is what the simulation can claim. The half-width shrinks about as 1/√n: from 5 to 20 replications it falls from about 1.0 hours to 0.4. Little's ratio is 1.000 on average with half-width 0.005: the law holds for the replications as a whole, as it should.

The interval describes the **simulation's** uncertainty about the model line. It says nothing about how well the model matches a real fab, which only data from a real line can show."""),
md("""## Step 9: where does the time go?

`stage_summary()` splits the line into its three tool groups, averaged over the replications and measured after the warm-up. For each group, **utilization** is the share of the group's tool-hours spent processing, `queue_h` is the mean wait of a lot before it gets a tool, and `in_group_h` is the mean time in the group, wait plus processing."""),
code("""summary = stage_summary(runs)
print(summary.round(3).to_string())
print()
print(f"sum of the mean times in the three groups: {summary['in_group_h'].sum():.3f} h; mean cycle time {reps['cycle_time'].mean():.3f} h")
print(f"processing alone: {BASE_LINE['furnace_h'] + BASE_LINE['litho_h'] + BASE_LINE['etch_h']:.1f} h; waiting: {summary['queue_h'].sum():.2f} h")"""),
md("""**What to notice.** The times in the three groups add up to the cycle time, as they should. A lot spends about 4.0 hours being processed and the rest of its 7.6 hours, 3.6 hours, waiting. **Lithography** has the highest utilization, 0.84, and the longest wait, 2.3 hours: it is the **bottleneck**, the group that limits what the line can carry. The furnace, at 0.75 with two tubes, waits 1.3 hours. The etch chambers, at 0.50, almost never make a lot wait. A queue grows quickly as utilization approaches 1, so a group at 0.84 has a much longer queue than one at 0.50 and its small changes matter much more. This is why the next step compares a change to lithography with a change to etch."""),
md("""## Step 10: two what-if changes, compared by mean and spread

The twin can answer "what if" questions that would cost a fab real money to test. Two changes to the line, each a **10 % shorter process time**:

- **etch:** `etch_h` from 1.2 to 1.08 hours (a faster etch recipe);
- **lithography:** `litho_h` from 1.0 to 0.9 hours.

The simulation runs each changed line for the same 20 seeds as the baseline. Replications with the same seed use the same random numbers: the same lots arrive at the same relative times with the same unit process times. Differences between the two lines are then **paired**: for each seed, the baseline's mean cycle time minus the changed line's, and the 20 differences go through `ci95()`. Pairing removes the randomness that both lines share, so a small difference can stand out, which the unpaired comparison of the two separate intervals would hide."""),
code("""def cycle_times(rs):
    return np.array([mean_cycle_time(r.lots["t_in"], r.lots["t_out"], WARMUP_H, HOURS) for r in rs])

base_ct = cycle_times(runs)
m0, h0 = ci95(base_ct)
print(f"baseline: mean cycle time {m0:.3f} h, 95 % interval {m0 - h0:.3f} to {m0 + h0:.3f}\\n")

changes = {"etch 10 % faster": line_config(etch_h=1.08),
           "lithography 10 % faster": line_config(litho_h=0.9)}
for name, cfg in changes.items():
    new_ct = cycle_times(run_replications(cfg))
    m1, h1 = ci95(new_ct)
    diff = base_ct - new_ct                        # same seeds: one difference per replication
    md_, hd = ci95(diff)
    print(f"{name}:")
    print(f"  mean cycle time {m1:.3f} h, 95 % interval {m1 - h1:.3f} to {m1 + h1:.3f}  (the baseline's interval {'overlaps' if m1 + h1 > m0 - h0 else 'does not overlap'})")
    print(f"  paired reduction {md_:.3f} h, half-width {hd:.3f} h; the 20 differences range {diff.min():.3f} to {diff.max():.3f}; positive in {int((diff > 0).sum())} of 20")"""),
md("""Write down the paired mean **reduction** in cycle time for the **etch** change, in hours to 2 decimals: the lab check asks for it.

**What to notice.** The faster etch saves 0.14 hours, a small amount that matches the 0.12 hours of process time it removes. The two separate intervals overlap completely, but separate intervals are not the right test of a paired difference, and overlapping intervals do not by themselves show that there is no difference. The right test is the interval of the paired differences: they, all between 0.13 and 0.15 and positive in all 20 replications, show the saving clearly, but it is small. The faster **lithography** tool saves 1.35 hours with a half-width of 0.22, about ten times more for the same 10 % change, because it speeds up the group where lots wait. In this model, for these two changes, **improving the tool that is not the bottleneck barely changed the line's cycle time**; that is a finding of this comparison, not a rule for every line. A change like the lithography one can also be seen from the two separate intervals, which do not overlap.

The same lesson holds for the spread: quote the mean and the interval over replications, and say whether the two lines were compared on the same random numbers."""),
md("""## Step 11: your task, `calibrate_now()`, and maintenance

The furnace tubes carry the drift of Module 5. Every run, a tube's thermocouple bias grows by a small random amount, the thickness drifts by about half a nanometre per degree, and the **Module 5 Kalman filter**, one monitor wafer per run, keeps an estimate of that drift for each tube. The run-to-run controller compensates by lengthening the oxidation time, until it reaches its 100-minute limit, at a drift of about **5 nm**. Beyond that, lots come out **off target**. A thermocouple can also start to fail: once in about 150 runs, its bias then grows several times faster, as in the failing scenario of Module 5.

A **calibration** (a temperature profile run with a reference thermocouple) removes the bias, and takes the tube out of service for 8 hours. The line has three maintenance policies:

- **none:** never calibrate;
- **preventive:** calibrate a tube after every 50 runs, whatever its condition (`interval_runs`);
- **predictive:** calibrate a tube when its drift estimate has moved by more than **2.0 nm for 3 runs in a row**: the alarm of Module 5, applied to each tube's estimate while the line runs.

Two simplifications of this model matter for what follows. **First**, compensation does not change the furnace's processing time: the simulation draws each furnace step independently of the drift, so the extra oxidation minutes the controller adds are not added to the step. The comparison includes calibration downtime and the drift-based count of off-target lots, not the longer processing that compensation would take. **Second**, a calibration resets both the bias and a failing thermocouple, so it acts like a calibration plus a successful repair or replacement. The maintenance conclusions hold for the model under these assumptions.

The predictive policy calls **your** rule after every furnace run. It receives the tube's drift estimates since its last calibration, the latest last.

**Your task:** write `calibrate_now(recent, limit, k)`. Return `True` when there are at least `k` estimates and the **last k** are all beyond the limit in size, `abs(value) > limit`; otherwise return `False`. Compare each value with the limit by its size, because a drift can also be negative."""),
code("""def calibrate_now(recent, limit, k):
    \"\"\"True when recent has at least k values and the last k are all beyond the limit: abs(value) > limit.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete calibrate_now() first.")"""),
code("""print(calibrate_now([0.4, 2.2, 2.5, 2.6], 2.0, 3), "  (should be True)")
print(calibrate_now([2.5, 2.6, 1.9], 2.0, 3), "  (should be False: the last value is inside the limit)")
print(calibrate_now([2.5, 2.6], 2.0, 3), "  (should be False: only two values)")"""),
md("""The cell runs the 20 replications for each policy, on the same seeds, and `maintenance_table()` counts, after the warm-up, the **completed calibrations** (a calibration is logged when it ends, so one still under way when a replication ends is not counted), the **off-target lots** (lots run while a tube's drift was beyond 5 nm) and the mean cycle time. The table also shows how many runs a tube with a failing thermocouple made, at the median, between the start of the failure and its calibration; that median uses every finished calibration of the replication, warm-up included, and leaves out failures still unresolved at the end. When no replication has an off-target lot, the table says so instead of printing 0 ± 0."""),
code("""policies = {"none": line_config(),
            "preventive, every 50 runs": line_config(maintenance="preventive", interval_runs=50),
            "predictive, alarm at 2.0 nm for 3 runs": line_config(maintenance="predictive")}
rows = []
for name, cfg in policies.items():
    rs = run_replications(cfg, trigger=calibrate_now)
    mt = maintenance_table(rs)
    failing = pd.concat([r.calibrations for r in rs])["fault_runs"].dropna()     # runs a failing tube ran before its calibration
    c, ch = ci95(mt["calibrations"])
    o, oh = ci95(mt["off_target"])
    t, th = ci95(mt["cycle_time"])
    off_text = f"{o:.1f} ± {oh:.1f}" if mt["off_target"].max() > 0 else f"none in {len(mt)} replications"
    rows.append({"policy": name, "completed calibrations": f"{c:.1f} ± {ch:.1f}", "off-target lots": off_text,
                 "most in one replication": int(mt["off_target"].max()), "of lots run": f"{mt['lots_run'].mean():.0f}",
                 "failing tube ran (median runs)": f"{failing.median():.0f}" if len(failing) else "-",
                 "cycle time (h)": f"{t:.2f} ± {th:.2f}"})
print(pd.DataFrame(rows).to_string(index=False))"""),
md("""**What to notice.** Without calibrations, almost every lot is off target, since every tube drifts past 5 nm within the window. The **preventive** rule calibrates a tube every 50 runs, about 14 calibrations in all, and still leaves about 38 off-target lots per replication, with 82 in the worst one: a tube whose thermocouple starts to fail soon after a calibration can drift far beyond the limit before its next one. The **predictive** rule makes about 12 calibrations, fewer than the preventive rule, and **no** off-target lot was observed in any of the 20 replications. That does not establish zero risk: these replications simply contained no event from which to estimate a rate (zero of 20 replications gives a one-sided 95 % upper bound of about 14 % on the chance that a comparable replication has at least one off-target lot, not a bound on the share of lots). The rule calibrates a healthy tube when its estimate says so, and a failing tube after a median of 10 runs of failure, where the preventive rule leaves it running for 29. The estimate is Module 5's: one monitor wafer per run, filtered.

Maintenance is not free. Each calibration takes one of two tubes away for 8 hours, the other cannot carry the load alone, and the queue it builds takes hours to clear. The furnace is not the bottleneck, yet the mean cycle time rises from 7.6 hours to 9.0 with the predictive rule and 9.3 with the preventive one, an increase of 18 % to 23 %. The twin makes that cost visible, so it can be set against the cost of off-target lots; that trade is a decision for the fab, and it depends on prices that the simulation does not have."""),
md("""## Step 12: your own line

Open the **Module 6 Lab Check** in Canvas. Question 1 gives you a **KEY**, a whole number from 100 to 999. Type it below in place of `None` and run the cell. It uses **your KEY's 20 seeds** (KEY × 1,000 + 1 to KEY × 1,000 + 20), runs the baseline line and the line with a 10 % faster **lithography** tool on the same seeds, and reports the paired mean reduction in cycle time, with **your functions**."""),
code("""KEY = None      # <- your KEY from Question 1 of the lab check, for example KEY = 417

if KEY is None:
    print("Type your KEY in place of None, then run this cell again.")
else:
    seeds = [KEY * 1000 + i for i in range(1, 21)]
    base_runs = run_replications(None, seeds)
    fast_runs = run_replications(line_config(litho_h=0.9), seeds)
    base_my, fast_my = cycle_times(base_runs), cycle_times(fast_runs)
    m_base, h_base = ci95(base_my)
    m_fast, h_fast = ci95(fast_my)
    m_diff, h_diff = ci95(base_my - fast_my)
    print(f"KEY {KEY}: baseline mean cycle time {m_base:.3f} h (half-width {h_base:.3f}); litho 10 % faster {m_fast:.3f} h (half-width {h_fast:.3f})")
    print(f"paired reduction: {m_diff:.3f} h, half-width {h_diff:.3f} h")"""),
md("""## Step 13: check your work

The cell below tests your five functions on cases you cannot see. If all pass, it prints your **verification value**: a code made from your KEY and your paired mean reduction in cycle time (hours), computed with the course's reference functions. Only a passing check prints it, and it is **not** the last number Step 12 printed. Type it into Question 1 of the lab check.

If it says **NOT PASSED**, read the message: it tells you what to fix."""),
code("""# The course's check code is in dt_m6_check.py, next to this notebook.
# It is kept out of the notebook so that the course's reference solutions are not on screen.
from dt_m6_check import check_dt_module6

passed, messages, value = check_dt_module6(throughput, mean_cycle_time, time_average_wip, ci95, calibrate_now, KEY)
for line_text in messages:
    print(line_text)
if passed:
    print()
    print(f"Verification value for KEY = {KEY}: {value:.3f}")
    print("Type this number into Question 1 of the Module 6 Lab Check.")"""),
md("""## What this means for the twin

A twin of one tool answers "what will this tool do?" A twin of a toolset answers a different question: "what will the line do if something changes?" The line's cycle time, its work in progress and its throughput are tied together by Little's law, they are random, and a simulation gives them as a mean with an interval, from replications that start after a warm-up. The same model shows where a change matters, at the bottleneck, and what maintenance costs in cycle time. The Module 5 drift estimate turns maintenance from a calendar into a decision about each tube.

**What a three-tool model leaves out.** A real fab is much harder than this line:

- **Re-entrant flow.** A lot returns to lithography and etch for every layer, dozens of times, so the bottleneck depends on the mix of layers, and lots of different ages compete for the same tools.
- **Many products and priorities.** Different recipes, set-up changes between them, and rush lots that jump the queue.
- **Batching rules.** A real furnace processes many lots at once, and the rule for filling a batch changes the queues.
- **Other things that go wrong.** Random breakdowns, operators, transport and rework loops, none of which the model has.

Each of them can change which tool is the bottleneck. A model that leaves them out is useful for the questions it was built for, such as how queues respond to load and to one change, and it needs data from a real line before it is trusted for more.

**Claim only what the data support.** On simulated data, a result shows how the course's model line behaves. It does not show how a real line would respond to the same change.

To keep a copy of this notebook, choose **File > Download**."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"m6-{i:02d}"
text = nbf.writes(nb)
assert "—" not in text and "–" not in text, "em or en dash in notebook text"
(OUT / NAME).write_text(text)
print("wrote", OUT / NAME, len(cells), "cells")

# ---------------------------------------------------------------- optional Google Colab copy
COLAB = ROOT / "colab" / "m6"
COLAB.mkdir(parents=True, exist_ok=True)
CNAME = "DT-M6-lab-toolset-twin-colab.ipynb"
RAW = "https://raw.githubusercontent.com/CSU-Physics/digital-twins-notebooks/main/content/m6/"
colab_cells = [cells[0],
               md("""## Colab only: get SimPy and the course files

You are running the **Google Colab** copy of this lab. Colab needs **SimPy**, installed with pip, and three course files next to the notebook: `line.py`, the SimPy model of the line, `line_tools.py`, the helpers, and `dt_m6_check.py`, the check code. Run the cell below once each time you open the notebook; it installs SimPy and downloads the files from the course repository. Everything else is the same as in the Canvas version, and your verification value is the same."""),
               code(f"""%pip install -q simpy
import urllib.request

BASE = "{RAW}"
for name in ["line.py", "line_tools.py", "dt_m6_check.py"]:
    urllib.request.urlretrieve(BASE + name, name)
    print("downloaded", name)""")] + cells[1:]
cnb = nbf.v4.new_notebook(cells=colab_cells, metadata={"colab": {"provenance": []},
                                                       "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                       "language_info": {"name": "python"}})
for i, c in enumerate(cnb.cells):
    c["id"] = f"m6c-{i:02d}"
ctext = nbf.writes(cnb)
assert "—" not in ctext and "–" not in ctext
(COLAB / CNAME).write_text(ctext)
print("wrote", COLAB / CNAME, len(colab_cells), "cells")
