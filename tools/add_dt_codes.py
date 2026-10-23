"""Adds the verification code of each stored KEY to dt_m2_solutions.json ... dt_m6_solutions.json, without recomputing the results.
The code is (A * round(1000 * value) + B * KEY + C) mod 9973 / 1000, with the constants of each module's check file.
For a sample of KEYs it also recomputes the code with the check file's own verification_code().
Run one module per call (the modules share file names such as furnace.py): python3 tools/add_dt_codes.py 2"""
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for n in [int(a) for a in sys.argv[1:]] or (2,):
    sys.path[:0] = [str(ROOT / "content" / f"m{n}")]
    chk = importlib.import_module(f"dt_m{n}_check")
    path = ROOT / f"dt_m{n}_solutions.json"
    sets = json.load(open(path))
    for s in sets:
        s["code"] = ((chk.CODE_A * int(round(s["value"] * 1000)) + chk.CODE_B * s["key"] + chk.CODE_C) % 9973) / 1000
    codes = [s["code"] for s in sets]
    assert len(set(codes)) == len(sets), f"module {n}: two KEYs share a verification code"
    assert not any(abs(s["code"] - s["value"]) < 0.0005 for s in sets), f"module {n}: a code equals its own result"
    sample = sets[:: max(1, len(sets) // (3 if n == 6 else 8))]
    for s in sample:
        assert chk.verification_code(s["key"]) == s["code"], (n, s)
    json.dump(sets, open(path, "w"))
    same_as_value = sum(abs(s["code"] - s["value"]) < 0.0005 for s in sets)
    print(f"module {n}: {len(sets)} codes, all distinct, {min(codes):.3f} to {max(codes):.3f}; checked {len(sample)} with verification_code(); "
          f"codes equal to the result itself: {same_as_value}")
    sys.path.pop(0)
