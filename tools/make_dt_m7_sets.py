"""Make the 200 stored answer sets of each Module 7 project quiz: Question 1, the verification value for a KEY.

For each track, KEYs from 100 to 999 are taken in a fixed shuffled order. A KEY is skipped when one of its three mean
absolute errors, in thousandths, lies within 0.01 of a rounding boundary, so that the value cannot differ between
computers. The sets go to dt_m7_<track>_solutions.json, which stays out of the public repository."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "content" / "m7"))
import dt_m7_check as K  # noqa: E402


def near_boundary(x):
    f = float(x) * 1000.0
    return abs((f - np.floor(f)) - 0.5) < 0.01


def make(track, n=200):
    order = np.random.default_rng([2099, {"process": 1, "metrology": 2, "toolset": 3}[track]]).permutation(np.arange(100, 1000))
    sets, skipped = [], []
    t0 = time.time()
    for key in order:
        key = int(key)
        r = K.reference_results(track, key)
        if any(near_boundary(r[t][0]) for t in ("heldout", "challenging", "baseline")):
            skipped.append(key)
            continue
        tol = r["tolerance"]
        sets.append({"key": key, "value": K.personal_value(track, key),
                     "heldout": round(r["heldout"][0], 3), "challenging": round(r["challenging"][0], 3), "baseline": round(r["baseline"][0], 3),
                     "heldout_verdict": K.reference_verdict(r["heldout"][0], r["heldout"][1], tol),
                     "challenging_verdict": K.reference_verdict(r["challenging"][0], r["challenging"][1], tol)})
        if len(sets) == n:
            break
    sets.sort(key=lambda s: s["key"])
    (ROOT / f"dt_m7_{track}_solutions.json").write_text(json.dumps(sets, indent=0))
    vals = [s["value"] for s in sets]
    hv = {v: sum(s["heldout_verdict"] == v for s in sets) for v in ("meets", "misses", "insufficient evidence")}
    cv = {v: sum(s["challenging_verdict"] == v for s in sets) for v in ("meets", "misses", "insufficient evidence")}
    print(track, len(sets), "sets; KEYs", sets[0]["key"], "to", sets[-1]["key"], "; values", min(vals), "to", max(vals),
          "; distinct", len(set(vals)), "; skipped", len(skipped), "of", len(sets) + len(skipped), "tried;",
          "held-out verdicts", hv, "; challenging verdicts", cv, f"; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    for track in sys.argv[1:] or ("process", "metrology", "toolset"):
        make(track)
