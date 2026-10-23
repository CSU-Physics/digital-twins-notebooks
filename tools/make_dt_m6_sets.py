"""Answer sets for Question 1 of the Module 6 lab check: 200 KEYs and their verification values.
Writes dt_m6_solutions.json at the repository root (gitignored: it holds every answer).

A KEY is skipped if its value is close to a rounding boundary, so that a different floating-point library cannot round it the
other way, or if the 10 % faster lithography tool did not shorten the mean cycle time in all 20 paired replications (it does
for every key tried; the test guards it)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m6"), str(ROOT / "checks")]
from dt_m6_check import personal_differences  # noqa: E402
import line_tools  # noqa: E402

rng = np.random.default_rng(20261026)
keys = [int(k) for k in rng.choice(np.arange(100, 1000), size=260, replace=False)]   # random order, so the 200 kept span 100 to 999
sets, skipped, mins = [], [], []
for k in keys:
    d = personal_differences(k)
    line_tools._CACHE.clear()
    raw = float(np.mean(d))
    if abs((raw * 1000) % 1 - 0.5) < 0.02 or min(d) <= 0:
        skipped.append(k)
        continue
    sets.append({"key": k, "value": round(raw, 3)})
    mins.append(min(d))
    if len(sets) == 200:
        break
assert len(sets) == 200
sets.sort(key=lambda d: d["key"])
from dt_m6_check import CODE_A, CODE_B, CODE_C  # noqa: E402
for s in sets:   # the verification value of Question 1: a code made from the result and the KEY (see dt_m6_check.py)
    s["code"] = ((CODE_A * int(round(s["value"] * 1000)) + CODE_B * s["key"] + CODE_C) % 9973) / 1000
assert len({s["code"] for s in sets}) == len(sets), "two KEYs share a verification code"
json.dump(sets, open(ROOT / "dt_m6_solutions.json", "w"))
vals = [s["value"] for s in sets]
print("200 sets; KEYs", sets[0]["key"], "to", sets[-1]["key"], "; values", min(vals), "to", max(vals), "; distinct", len(set(vals)),
      "; smallest single-replication reduction", round(min(mins), 3), "; skipped", len(skipped))
