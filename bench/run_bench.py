"""Run the pipeline on every real benchmark capture in data/, score it against the laser ground truth, write the report.

    python bench/run_bench.py [--only name ...] [--force] [--no-ablation] [--results DIR]

Results (default DIR = results/benchmark): DIR/runs/<capture>/plan.{json,png,svg}, DIR/metrics.json, DIR/BENCHMARK_REPORT.md,
DIR/drift_ablation_<capture>.png.  Existing plans are reused unless --force.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_src = os.environ.get("FLOORSCAN_SRC")  # fix loop: run the pipeline from a snapshot of an earlier source tree
if _src:
    sys.path.insert(0, _src)
os.environ.setdefault("FLOORSCAN_CACHE", os.path.join(ROOT, "bench", "cache"))  # replayable SfM cache (see docs)

from bench import gates as G  # noqa: E402
from bench.metrics import align, evaluate, gt_rect, transform_pts  # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "benchmark")
SPLIT = json.load(open(os.path.join(DATA, "SPLIT.json")))
EXCLUDED = set(SPLIT.get("excluded_sessions", {}))
ROOMS = {v: (part, [s for s in d["sessions"] if s not in EXCLUDED]) for part in ("dev", "test") for v, d in SPLIT[part].items()}
SPLIT_OF = {s: part for v, (part, ss) in ROOMS.items() for s in ss}
# same room captured 2-3 times with the LiDAR tier: every pair of sessions
REPEAT_PAIRS = [("lidar", f"{a}_lidar", f"{b}_lidar") for _, ss in ROOMS.values() for i, a in enumerate(ss) for b in ss[i + 1:]]
ABLATION = tuple(f"{ss[0]}_lidar" for part, ss in ROOMS.values() if part == "test")  # drift handling ON vs OFF


def tier_of(name: str) -> str:
    return "lidar" if "lidar" in name else ("video" if "video" in name else "photos")


def gt_of(name: str, data: str = DATA) -> Dict:
    """Capture '<id>_<tier>' -> ground-truth json via data/gt/index.json (capture id -> laser GT file)."""
    idx = json.load(open(os.path.join(data, "gt", "index.json")))
    return json.load(open(os.path.join(data, "gt", idx[name.split("_")[0]])))


def run_one(name: str, force: bool = False, cfg_over: Dict = None, out_name: str = None):
    from floorscan.cli import run as cli_run
    path = os.path.join(DATA, name)
    out_dir = os.path.join(RES, "runs", out_name or name)
    pj = os.path.join(out_dir, "plan.json")
    if os.path.exists(pj) and not force:
        return json.load(open(pj))
    if cfg_over:
        from floorscan.config import Config
        from floorscan.pipeline.run import run_capture
        from floorscan.render import render_plan
        cfg = Config()
        for k, v in cfg_over.items():
            setattr(cfg, k, v)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        out, *_ = run_capture(path, cfg)
        out["timing"]["total"] = round(time.time() - t0, 3)
        json.dump(out, open(pj, "w"), indent=1)
        render_plan(out, os.path.join(out_dir, "plan.png"), os.path.join(out_dir, "plan.svg"))
        return out
    return cli_run(path, out_dir, quiet=True)


def score(name: str, out: Dict) -> Dict:
    tier = tier_of(name)
    gt = gt_of(name)
    to, tr = G.tol_open(tier)
    m = evaluate(out, gt, tol_open=to, tol_open_rel=tr)
    if gt.get("openings_reliable") is False:
        m["openings"] = []  # no trustworthy opening ground truth (real laser data): not scored
        m["calib"] = [c for c in m["calib"] if not c["kind"].startswith("opening")]
    return dict(name=name, tier=tier, metrics=m, gates=G.gate_summary(m, gt, tier),
                timing=out["timing"], rows=G.room_rows(m, gt, tier))


# ------------------------------------------------------------------------------------------------------
def fmt_mm(x):
    return "—" if x is None else f"{x * 1000:.0f}"


def md_table(header: List[str], rows: List[List[str]]) -> str:
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s


def yn(b):
    return "PASS" if b else "FAIL"


def ablation_figure(name: str, out_on: Dict, out_off: Dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP
    gt = gt_of(name)
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    for ax, out, ttl in ((axs[0], out_off, "drift handling OFF (poses used as-is)"),
                         (axs[1], out_on, "drift handling ON (plane-anchored pose correction)")):
        a = align(out, gt)
        for r in gt["rooms"]:
            ax.add_patch(MP(np.array(gt_rect(r).exterior.coords), closed=True, fill=False, ec="#2a9d8f", lw=2.2,
                            label="ground truth" if r is gt["rooms"][0] else None))
        for i, r in enumerate(out["rooms"]):
            P = transform_pts(a, np.array(r["polygon"]))
            ax.add_patch(MP(P, closed=True, fc="#e76f51", alpha=0.35, ec="#e76f51", lw=1.5,
                            label="measured footprint" if i == 0 else None))
        m = evaluate(out, gt)
        ax.set_title(f"{ttl}\nfootprint err {m['footprint_err_pct']:+.1f} %, IoU {a['iou']:.3f}, overlaps {m['overlap_area']:.2f} m², "
                     f"links {len(out['plan']['adjacency'])}", fontsize=9)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary_section(S: Dict, rep: Dict) -> str:
    """Share of measurements inside each case-study gate, per split (DEV = used for tuning, TEST = held out) and tier."""
    rows = []
    for part in ("test", "dev"):
        for tier in ("lidar", "video", "photos"):
            g = [s["gates"] for n, s in S.items() if s["tier"] == tier and SPLIT_OF.get(n.split("_")[0]) == part]
            if not g:
                continue
            wn, wo = sum(x["walls_n"] for x in g), sum(x["walls_ok"] for x in g)
            missing = sum(x.get("walls_missing", 0) for x in g)
            cn, co = sum(x["ceilings_n"] for x in g), sum(x["ceilings_ok"] for x in g)
            on = sum(x["openings_gt"] + x["openings_phantom"] for x in g)
            oo = sum(x["openings_ok"] for x in g)
            fp = sum(bool(x["footprint_pass"]) for x in g)
            werr = [x["walls_max_abs_err"] for x in g if x["walls_max_abs_err"] is not None]
            rows.append([part.upper(), tier, len(g), f"{wo}/{wn + missing} ({100 * wo / max(wn + missing, 1):.0f} %)",
                         f"{co}/{cn} ({100 * co / max(cn, 1):.0f} %)",
                         f"{oo}/{on} ({100 * oo / max(on, 1):.0f} %)" if on else "-",
                         f"{fp}/{len(g)}", f"{1000 * float(np.median(werr)):.0f}" if werr else "-"])
    rep_txt = ""
    for part in ("test", "dev"):
        x = [r for r in rep.values() if r.get("split") == part]
        if x:
            mx = [max(abs(w["diff"]) for w in r["walls"]) for r in x if r["walls"]] or [float("nan")]
            rep_txt += (f"* {part.upper()} repeatability ({len(x)} LiDAR session pairs): walls within max(1 cm, 0.5 %) in "
                        f"{sum(r['walls_ok'] for r in x)}/{len(x)} pairs, ceiling spread <= 1 cm in "
                        f"{sum(r['height_ok'] for r in x)}/{len(x)}; median largest wall difference "
                        f"{1000 * float(np.median(mx)):.0f} mm.\n")
    return ("## 0. Summary - share of measurements inside each gate\n\n"
            "TEST rooms were never used for any design or tuning decision (data/SPLIT.json); DEV rooms were.\n\n" +
            md_table(["split", "tier", "captures", "walls in gate", "ceilings in gate",
                      "openings within tolerance (incl. missed/phantom)", "footprint in gate", "median worst wall error mm"], rows)
            + "\n" + rep_txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="substrings of capture names to run")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all", help="DEV / TEST rooms of data/SPLIT.json")
    ap.add_argument("--force", action="store_true", help="re-run the pipeline even if a plan already exists")
    ap.add_argument("--no-ablation", action="store_true", help="skip the drift ON/OFF ablation runs")
    ap.add_argument("--results", help="output directory (default results/benchmark; the fix loop passes its own)")
    a = ap.parse_args()
    global RES
    if a.results:
        RES = os.path.abspath(a.results)
    os.makedirs(os.path.join(RES, "runs"), exist_ok=True)
    names = sorted(d for d in os.listdir(DATA) if d != "gt" and os.path.exists(os.path.join(DATA, d, "capture.json")))
    names = [n for n in names if n.split("_")[0] in SPLIT_OF and (a.split == "all" or SPLIT_OF[n.split("_")[0]] == a.split)]
    if a.only:
        names = [n for n in names if any(o in n for o in a.only)]
    S: Dict[str, Dict] = {}
    outs: Dict[str, Dict] = {}
    for n in names:
        t0 = time.time()
        out = run_one(n, a.force)
        outs[n] = out
        S[n] = score(n, out)
        g = S[n]["gates"]
        print(f"{n:18s} {time.time() - t0:6.1f}s walls {g['walls_ok']}/{g['walls_n']} (max {1000 * (g['walls_max_abs_err'] or 0):.0f} mm) "
              f"ceil {g['ceilings_ok']}/{g['ceilings_n']} ({1000 * (g['ceilings_max_abs_err'] or 0):.0f} mm) "
              f"foot {g['footprint_err_pct']:+.1f}% dmg {g['damage_fp']} detections", flush=True)
    # ---- repeatability -------------------------------------------------------------------------------
    rep = {}
    for tier, na, nb in REPEAT_PAIRS:
        if na in outs and nb in outs:
            gt = gt_of(na)
            if not outs[na]["rooms"] or not outs[nb]["rooms"]:  # one capture reconstructed nothing: the pair fails
                r = dict(walls=[], walls_ok=False, height_ok=False, height_a=float("nan"), height_b=float("nan"),
                         height_spread=float("nan"), height_bias=[float("nan"), float("nan")], verdict="NO ROOM in one capture")
            else:
                r = G.repeatability(outs[na], outs[nb], gt, tier)
                r["verdict"] = G.classify_repeat(r, tier, gt["rooms"][0]["ceiling_height"])
            r["split"] = SPLIT_OF.get(na.split("_")[0], "?")
            rep[f"{tier} ({na} vs {nb})"] = r
    # ---- drift ablation ------------------------------------------------------------------------------
    abl = {}
    if not a.no_ablation:
        for n in ABLATION:
            if n in outs:
                off = run_one(n, a.force, dict(local_planar=False, door_graph=False, damage=False), out_name=n + "_driftoff")
                gt = gt_of(n)
                abl[n] = dict(on=G.gate_summary(evaluate(outs[n], gt), gt, tier_of(n)),
                              off=G.gate_summary(evaluate(off, gt), gt, tier_of(n)))
                ablation_figure(n, outs[n], off, os.path.join(RES, f"drift_ablation_{n}.png"))
    # ---- calibration (coverage) ---------------------------------------------------------------------
    cal: Dict[str, Dict[str, List]] = {}
    for n, s in S.items():
        for c in s["metrics"]["calib"]:
            cal.setdefault(s["tier"], {}).setdefault(c["kind"], []).append(c)
    # ---- write json + report -----------------------------------------------------------------------
    json.dump(dict(scores={k: dict(gates=v["gates"], timing=v["timing"], split=SPLIT_OF.get(k.split("_")[0]),
                               calib=v["metrics"]["calib"]) for k, v in S.items()},
                   repeatability=rep, ablation=abl), open(os.path.join(RES, "metrics.json"), "w"), indent=1, default=float)
    gt0 = gt_of(names[0]) if names else {}
    r0 = gt0.get("rooms", [{}])[0]
    lines = ["# Benchmark report - real captures (Apple ARKitScenes) vs laser ground truth\n",
             "> Real iPad Pro captures from Apple's public ARKitScenes dataset (non-commercial research licence): LiDAR depth + ARKit "
             "poses (two captures of the same room), the device video, and 8 frames from the walkthrough as the photo set. "
             f"Ground truth: the FARO laser scan of the same room (`dataprep/room_gt.py`): {r0.get('w', 0):.3f} x {r0.get('d', 0):.3f} m, "
             f"ceiling {r0.get('ceiling_height', 0):.3f} m, main wall planes only (fit figure in `data/gt/`). "
             "Not an iPhone 15+ capture made with our protocol (no scale card). Openings are not scored (no reliable opening ground truth). "
             "The room has no known damage, so every damage detection counts as a false positive. "
             "Rebuild the data with `python dataprep/fetch_sample.py`.\n"]
    lines.append(summary_section(S, rep))
    lines.append("## 1. Gates at all three tiers\n")
    hdr = ["capture", "walls ok (max err mm)", "ceilings ok (max err mm)", "openings", "footprint err %", "damage detections (all FP)", "time s"]
    for tier in ("lidar", "video", "photos"):
        rows = [[n, f"{g['walls_ok']}/{g['walls_n']} ({fmt_mm(g['walls_max_abs_err'])})",
                 f"{g['ceilings_ok']}/{g['ceilings_n']} ({fmt_mm(g['ceilings_max_abs_err'])})",
                 "n/a (no GT)" if g["openings_rate"] is None else f"{g['openings_ok']}/{g['openings_gt']} (+{g['openings_phantom']})",
                 f"{g['footprint_err_pct']:+.1f}", f"{g['damage_fp']}", f"{s['timing'].get('total', 0):.0f}"]
                for n, s in S.items() if s["tier"] == tier for g in [s["gates"]]]
        if not rows:
            continue
        tol = {"lidar": "wall <= max(2 cm, 0.5 %); ceiling <= 1.5 cm; opening width <= 2 cm on >= 85 % incl. missed/phantom",
               "video": "wall / ceiling / opening width within +-3 %; footprint +-3 %",
               "photos": "wall / ceiling / opening width within +-8 %; footprint +-8 %, correct adjacency, no overlaps"}[tier]
        lines.append(f"### {tier} tier  \nGate: {tol}\n")
        lines.append(md_table(hdr, rows))
        prow = [[n, yn(g["walls_pass"]), yn(g["ceiling_pass"]), "n/a" if g["openings_pass"] is None else yn(g["openings_pass"]),
                 yn(g["footprint_pass"])] for n, s in S.items() if s["tier"] == tier for g in [s["gates"]]]
        lines.append(md_table(["capture", "wall lengths", "ceiling height", "openings", "footprint"], prow))
    lines.append("\n## 2. Per-wall errors (measured - laser, mm)\n")
    rows = []
    for n, s in S.items():
        for r in s["metrics"]["rooms"]:
            rows.append([n] + [f"{1000 * r['wall_err'].get(k, float('nan')):+.0f}" for k in ("S", "E", "N", "W")]
                        + [f"{1000 * r['height_err']:+.0f}"])
    lines.append(md_table(["capture", "S", "E", "N", "W", "ceiling"], rows))
    lines.append("\n## 3. Repeatability (same room, same tier, two captures)\n")
    rows = []
    for tier, r in rep.items():
        d = ", ".join(f"{w['wall']}: {w['diff'] * 1000:+.0f} mm (tol {w['tol'] * 1000:.0f})" for w in r["walls"])
        rows.append([tier, d, f"{r['height_a']:.3f} / {r['height_b']:.3f} (spread {r['height_spread'] * 1000:.0f} mm, bias "
                     f"{r['height_bias'][0] * 1000:+.0f}/{r['height_bias'][1] * 1000:+.0f} mm)",
                     yn(r["walls_ok"]), yn(r["height_ok"]), r["verdict"]])
    lines.append(md_table(["pair", "per-wall difference between the two captures", "ceiling height a / b", "walls <= max(1 cm, 0.5 %)",
                           "ceiling spread <= 1 cm", "verdict (repeatable-but-biased vs unrepeatable)"], rows))
    lines.append("\n## 4. Drift accountability (pose correction ON vs poses used as-is)\n")
    rows = []
    for n, d in abl.items():
        for k in ("off", "on"):
            g = d[k]
            rows.append([n, k.upper(), f"{g['walls_ok']}/{g['walls_n']}", fmt_mm(g["walls_max_abs_err"]),
                         fmt_mm(g["ceilings_max_abs_err"]), f"{g['footprint_err_pct']:+.1f} %", f"{g['iou']:.3f}"])
        lines.append(f"![{n}](drift_ablation_{n}.png)\n")
    if rows:
        lines.append(md_table(["capture", "drift handling", "walls in gate", "max wall err mm", "ceiling err mm", "footprint err",
                               "footprint IoU"], rows))
    else:
        lines.append("Ablation not run (`--no-ablation`).\n")
    lines.append("\nMethod: ARKit poses are never used as-is. Per room, frames are chunked in time and each chunk gets an (x, y, yaw, z) "
                 "correction so its points sit on the fitted wall / floor / ceiling planes (joint over chunks with a smoothness prior, 3 "
                 "iterations); in multi-room captures a door-anchored pose graph then places the rooms. OFF = poses used as delivered.\n")
    lines.append("\n## 5. Calibration of the confidence intervals\n")
    rows = []
    for tier in ("lidar", "video", "photos"):
        allc = [c for v in cal.get(tier, {}).values() for c in v]
        if not allc:
            continue
        inside = np.mean([c["inside"] for c in allc])
        garbage = np.mean([abs(c["err"]) > 3 * max(c["half"], 1e-6) for c in allc])
        rows.append([tier, len(allc), f"{100 * inside:.0f} %", f"{100 * garbage:.1f} %",
                     ", ".join(f"{k}: {100 * np.mean([c['inside'] for c in v]):.0f} % (n={len(v)})" for k, v in sorted(cal[tier].items()))])
    lines.append(md_table(["tier", "n measurements", "inside 95 % CI", "error > 3x half-width ('confident garbage')", "by measurement"], rows))
    lines.append("\nIntervals: analytic sigma + per-tier systematic terms (`floorscan/uncertainty.py`) x the calibration factor in "
                 "`floorscan/calibration.json` (currently k = 1: no held-out real calibration captures exist; `bench/calibrate.py --data <dir>`).\n")
    lines.append("\n## 6. Timing (this machine, CPU only)\n")
    rows = [[n, f"{s['timing'].get('total', 0):.1f}", ", ".join(f"{k} {v:.1f}" for k, v in s["timing"].items() if k != "total")]
            for n, s in S.items()]
    lines.append(md_table(["capture", "total s", "stages (s)"], rows))
    open(os.path.join(RES, "BENCHMARK_REPORT.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", os.path.join(RES, "BENCHMARK_REPORT.md"))


if __name__ == "__main__":
    main()
