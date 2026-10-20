"""Answer sets for Question 1 of the Module 1 lab check: 200 KEYs and their verification values.
Writes dt_m1_solutions.json at the repository root (gitignored: it holds every answer)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "m1"), str(ROOT / "checks")]
from dt_m1_check import GAIN, reference_tau, reference_update  # noqa: E402
from furnace import STREAM_FIT_END_S, temperature_stream  # noqa: E402
from twin_tools import rms, run_twin  # noqa: E402

rng = np.random.default_rng(20261021)
keys = [int(k) for k in rng.choice(np.arange(100, 1000), size=260, replace=False)]   # random order, so the 200 kept span 100 to 999
sets = []
for k in keys:
    twin = run_twin(temperature_stream(k), reference_tau(k), GAIN, reference_update)
    raw = rms(twin.loc[twin["time_s"] >= STREAM_FIT_END_S, "error"])
    if abs((raw * 1000) % 1 - 0.5) < 0.02:      # too close to a rounding boundary
        continue
    sets.append({"key": k, "value": round(raw, 3)})
    if len(sets) == 200:
        break
assert len(sets) == 200
sets.sort(key=lambda d: d["key"])
json.dump(sets, open(ROOT / "dt_m1_solutions.json", "w"))
vals = [s["value"] for s in sets]
print("200 sets; KEYs", sets[0]["key"], "to", sets[-1]["key"], "; values", min(vals), "to", max(vals), "; distinct", len(set(vals)))
