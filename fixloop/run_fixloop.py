"""Regenerate the fix-loop before / after runs on the REAL captures (case-study Part 4).

    python fixloop/run_fixloop.py [--captures 42444946_lidar 42444949_lidar]

BEFORE = the source snapshot taken when the fix was declared (fixloop/before_src/floorscan); AFTER = the same tree with ONLY
the fix applied (fixloop/after_src/floorscan = before_src + fix.diff).  Both run on the same raw captures through
bench/run_bench.py (FLOORSCAN_SRC picks the source tree), so everything is regenerable.
Writes results/fixloop/{before,after}/ (plans + metrics.json + report), summary.json and FIXLOOP_REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "fixloop")
OUT = os.path.join(ROOT, "results", "fixloop")


def run(which: str, caps):
    env = dict(os.environ, FLOORSCAN_SRC=os.path.join(FIX, f"{which}_src"))
    res = os.path.join(OUT, which)
    subprocess.check_call([sys.executable, os.path.join(ROOT, "bench", "run_bench.py"), "--results", res, "--force",
                           "--no-ablation", "--only", *caps], cwd=ROOT, env=env)
    return json.load(open(os.path.join(res, "metrics.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", nargs="*", default=["42444946_lidar", "42444949_lidar"])
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(FIX, "declaration.json")))
    before, after = run("before", a.captures), run("after", a.captures)

    def pooled(m):
        g = [v["gates"] for v in m["scores"].values()]
        return max(x["walls_max_abs_err"] for x in g), sum(x["walls_ok"] for x in g), sum(x["walls_n"] for x in g)

    (b, b_ok, n), (af, a_ok, _) = pooled(before), pooled(after)
    rows = []
    for c in a.captures:
        gb, ga = before["scores"][c]["gates"], after["scores"][c]["gates"]
        rows.append(f"| {c} | {1000 * gb['walls_max_abs_err']:.0f} -> **{1000 * ga['walls_max_abs_err']:.0f}** | "
                    f"{gb['walls_ok']}/{gb['walls_n']} -> {ga['walls_ok']}/{ga['walls_n']} | "
                    f"{gb['footprint_err_pct']:+.1f} % -> {ga['footprint_err_pct']:+.1f} % | "
                    f"{1000 * gb['ceilings_max_abs_err']:.0f} -> {1000 * ga['ceilings_max_abs_err']:.0f} |")
    rep_b = list(before.get("repeatability", {}).values())
    rep_a = list(after.get("repeatability", {}).values())
    rep_line = ""
    if rep_b and rep_a:
        mb = max(abs(w["diff"]) for w in rep_b[0]["walls"])
        ma = max(abs(w["diff"]) for w in rep_a[0]["walls"])
        rep_line = (f"\nRepeatability (largest per-wall difference between the two captures): {1000 * mb:.0f} mm -> {1000 * ma:.0f} mm "
                    f"({rep_b[0]['verdict']} -> {rep_a[0]['verdict']}).\n")
    summary = dict(gate=cfg["gate"], metric=cfg["metric"], before=b, after=af, predicted=cfg["predicted_after"],
                   required=cfg["required"], walls_ok_before=b_ok, walls_ok_after=a_ok, walls_n=n,
                   repeat_max_diff_before=mb if rep_line else None, repeat_max_diff_after=ma if rep_line else None)
    rep = ["# Fix loop on real data: before / after\n",
           f"Gate: **{cfg['gate']}**.  ",
           f"Worst wall-length error over both captures: **before {1000 * b:.0f} mm -> after {1000 * af:.0f} mm** "
           f"(predicted {1000 * cfg['predicted_after']:.0f} mm; required <= {1000 * cfg['required']:.0f} mm). "
           f"Walls in the gate: {b_ok}/{n} -> {a_ok}/{n}.\n",
           "| capture | worst wall error mm | walls in gate | footprint error | ceiling error mm |", "|---|---|---|---|---|"] + rows
    rep.append(rep_line)
    rep.append("Plans and metrics of each run: `before/`, `after/`. Root-cause evidence: `diagnosis_before.md` / `diagnosis_after.md`. "
               "Declaration and post-mortem: `fixloop/FIX_DECLARATION.md`.")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "FIXLOOP_REPORT.md"), "w", encoding="utf-8").write("\n".join(rep) + "\n")
    summary["story"] = cfg.get("story", "").replace("{before}", f"{1000 * b:.0f} mm").replace("{after}", f"{1000 * af:.0f} mm")
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "story"}, indent=1))


if __name__ == "__main__":
    main()
