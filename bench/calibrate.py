"""Fit the interval calibration factors k on HELD-OUT real captures (never on the reported benchmark in data/).

    python bench/calibrate.py                  # the DEV captures of data/ (data/SPLIT.json); never the TEST captures
    python bench/calibrate.py --data <dir>     # or another folder of held-out captures (same layout as data/)

Procedure: run the pipeline with k = 1 (FLOORSCAN_NO_CAL=1), collect (error, sigma) for every measurement with
ground truth, and choose per (tier, measurement kind) the smallest k >= 1 such that 95 % of |error| <= 1.96 * k * sigma
(the worst case when a kind has 3-7 samples, k = 1 below that).  Writes floorscan/calibration.json.
The "_depth_net" entry (dataprep/fit_depth_scale.py) is kept.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["FLOORSCAN_NO_CAL"] = "1"
os.environ.setdefault("FLOORSCAN_CACHE", os.path.join(ROOT, "bench", "cache"))

from bench import gates as G  # noqa: E402
from bench.metrics import evaluate  # noqa: E402
from bench.run_bench import gt_of, tier_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="folder of held-out captures with gt/index.json (default: the DEV captures in data/)")
    ap.add_argument("--from-results", help="metrics.json of a benchmark run made with k = 1: fit from its DEV entries, no re-run")
    a = ap.parse_args()
    cal = {}
    if a.from_results:
        for n, v in json.load(open(a.from_results))["scores"].items():
            if v.get("split") == "dev":
                for c in v["calib"]:
                    cal.setdefault(tier_of(n), {}).setdefault(c["kind"], []).append(c)
        label = f"the DEV captures of {os.path.relpath(a.from_results, ROOT)}"
        return write(cal, label)
    from floorscan.config import Config
    from floorscan.pipeline.run import run_capture
    if a.data:
        data = os.path.abspath(a.data)
        assert data != os.path.join(ROOT, "data"), "use the default (DEV split) for data/"
        names = sorted(d for d in os.listdir(data) if d != "gt" and os.path.exists(os.path.join(data, d, "capture.json")))
        label = os.path.relpath(data, ROOT)
    else:
        data = os.path.join(ROOT, "data")
        dev = json.load(open(os.path.join(data, "SPLIT.json")))["dev"]
        names = sorted(d for d in os.listdir(data) if d.split("_")[0] in {v for x in dev.values() for v in x["sessions"]}
                       and os.path.exists(os.path.join(data, d, "capture.json")))
        label = "the DEV captures of data/ (data/SPLIT.json)"
    cal = {}
    for n in names:
        tier = tier_of(n)
        gt = gt_of(n, data)
        out, *_ = run_capture(os.path.join(data, n), Config())
        m = evaluate(out, gt, *G.tol_open(tier))
        if gt.get("openings_reliable") is False:
            m["calib"] = [c for c in m["calib"] if not c["kind"].startswith("opening")]
        print(n, len(m["calib"]), "measurements", flush=True)
        for c in m["calib"]:
            cal.setdefault(tier, {}).setdefault(c["kind"], []).append(c)
    write(cal, label)


K_MAX = 3.0  # cap: beyond this intervals stop being informative; gross failures are reported as confident garbage


def write(cal, label):
    result = {"_note": f"fitted by bench/calibrate.py on {label}"}
    for tier, kinds in cal.items():
        result[tier] = {}
        for kind, v in kinds.items():
            z = [abs(c["err"]) / max(c["half"] / 1.96, 1e-6) for c in v]
            if len(v) >= 8:
                k = max(1.0, float(np.quantile(z, 0.95)) / 1.96)
            elif len(v) >= 3:
                k = max(1.0, float(np.max(z)) / 1.96)  # tiny sample: cover the worst case seen
            else:
                k = 1.0
            k = float(round(min(k * 1.05, K_MAX), 3))  # gross failures must not blow up every interval
            result[tier][kind] = dict(k=k, n=len(v), coverage_before=round(float(np.mean([c["inside"] for c in v])), 3))
    out_path = os.path.join(ROOT, "floorscan", "calibration.json")
    old = json.load(open(out_path)) if os.path.exists(out_path) else {}
    result.update({k: v for k, v in old.items() if k.startswith("_") and k != "_note"})  # keep the depth / height fits
    json.dump(result, open(out_path, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
