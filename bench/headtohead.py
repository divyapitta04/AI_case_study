"""Head-to-head vs one consumer scanning app (case-study Part 3).

    python bench/headtohead.py --capture <name> --app "Polycam 3.x (free)" --export <file> [--results results/benchmark]

Inputs
  * our plan.json for the same room(s) at the LiDAR tier (results/benchmark/runs/<capture>/plan.json)
  * ground truth (data/gt/421337.json, the laser-derived ground truth, or your own tape/laser file)
  * the app's export, either
        - Apple RoomPlan CapturedRoom JSON  (walls/doors/windows/openings with `dimensions` + `transform`), or
        - the simple JSON documented below (what you get by transcribing any app's dimension table):
          {"rooms":[{"name":"room","wall_lengths":[4.2,3.6,4.2,3.6],"ceiling_height":2.52,"floor_area":15.1,
                     "openings":[{"type":"door","width":0.86},{"type":"window","width":1.3}]}]}

Output: one table, our error and the app's error dimension by dimension, and the share of shared dimensions on which we
beat or tie (|err| within 5 mm of the app's).  The gate is >= 70 %.

NO INCUMBENT EXPORT EXISTS IN THIS REPOSITORY (the author had no device and no app run).  Without --export the script
writes the table skeleton marked NOT RUN rather than inventing the app's numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TIE_MM = 0.005


def load_roomplan(path: str) -> Dict:
    """Reduce an Apple RoomPlan CapturedRoom JSON (or an array of them) to the simple schema."""
    j = json.load(open(path))
    rooms = j if isinstance(j, list) else [j]
    out = []
    for k, r in enumerate(rooms):
        walls = r.get("walls", [])
        lens = [float(w["dimensions"][0]) for w in walls]
        hs = [float(w["dimensions"][1]) for w in walls]
        pts = []
        for w in walls:
            T = np.array(w["transform"], float).reshape(4, 4).T if len(w["transform"]) == 16 else np.eye(4)
            half = float(w["dimensions"][0]) / 2
            c = T[:3, 3]
            ax = T[:3, 0]
            pts += [c[[0, 2]] + ax[[0, 2]] * half, c[[0, 2]] - ax[[0, 2]] * half]
        area = 0.0
        if len(pts) >= 3:
            from scipy.spatial import ConvexHull
            area = float(ConvexHull(np.array(pts)).volume)
        ops = []
        for kind in ("doors", "windows", "openings"):
            for o in r.get(kind, []):
                ops.append(dict(type=kind[:-1], width=float(o["dimensions"][0])))
        out.append(dict(name=r.get("name", f"room_{k}"), wall_lengths=lens, ceiling_height=float(np.median(hs)) if hs else None,
                        floor_area=area, openings=ops))
    return dict(rooms=out)


def compare(ours: Dict, gt: Dict, app: Dict, room_name: str) -> List[Dict]:
    g = next(r for r in gt["rooms"] if r["name"] == room_name)
    ro = ours["rooms"][0] if len(ours["rooms"]) == 1 else next(r for r in ours["rooms"] if r.get("name") == room_name)
    ar = next((r for r in app["rooms"] if r.get("name") in (room_name, None)), app["rooms"][0])
    rows = []
    # wall lengths: match by sorted order (apps do not label walls)
    ol = sorted(float(np.linalg.norm(np.array(w["p1"]) - np.array(w["p0"]))) for w in ro["walls"])
    gl = sorted(g["walls"].values())
    al = sorted(ar["wall_lengths"])
    for i, (o, t) in enumerate(zip(ol, gl)):
        if i < len(al):
            rows.append(("wall length #%d" % (i + 1), t, o - t, al[i] - t))
    rows.append(("ceiling height", g["ceiling_height"], ro["ceiling_height"]["value"] - g["ceiling_height"],
                 (ar["ceiling_height"] - g["ceiling_height"]) if ar.get("ceiling_height") else None))
    rows.append(("floor area (m2)", g["floor_area"], ro["floor_area"]["value"] - g["floor_area"],
                 (ar["floor_area"] - g["floor_area"]) if ar.get("floor_area") else None))
    gw = sorted(o["width"] for o in g["openings"])
    ow = sorted(o["width"]["value"] for o in ro["openings"])
    aw = sorted(o["width"] for o in ar["openings"])
    for i, t in enumerate(gw):
        o = min(ow, key=lambda x: abs(x - t)) if ow else None
        a = min(aw, key=lambda x: abs(x - t)) if aw else None
        rows.append((f"opening width {t:.2f}", t, (o - t) if o is not None else None, (a - t) if a is not None else None))
    return rows


def write_table(rows, app_name: str, capture: str, path: str, ran: bool):
    lines = [f"# Head-to-head: floorscan (LiDAR tier) vs {app_name}\n", f"Capture: `{capture}`\n"]
    if not ran:
        lines.append("> **NOT RUN.** No incumbent-app export is available in this repository. Provide a Polycam / magicplan / "
                     "RoomPlan export of the same room(s) (`--export`) to fill this table. The numbers are never invented.\n")
        lines.append("| dimension | ground truth | our error (m) | app error (m) | we beat or tie |\n|---|---|---|---|---|\n| (pending a real export) | | | | |\n")
    else:
        lines.append("| dimension | ground truth | our error (mm) | app error (mm) | we beat or tie |\n|---|---|---|---|---|")
        n = w = 0
        for name, t, eo, ea in rows:
            if eo is None or ea is None:
                lines.append(f"| {name} | {t:.3f} | {'—' if eo is None else f'{eo * 1000:+.0f}'} | {'—' if ea is None else f'{ea * 1000:+.0f}'} | n/a |")
                continue
            ok = abs(eo) <= abs(ea) + TIE_MM
            n += 1
            w += ok
            lines.append(f"| {name} | {t:.3f} | {eo * 1000:+.0f} | {ea * 1000:+.0f} | {'yes' if ok else 'no'} |")
        share = w / max(n, 1)
        lines.append(f"\n**Beat or tie on {w}/{n} shared dimensions = {100 * share:.0f} %** (gate >= 70 %: "
                     f"{'PASS' if share >= 0.7 else 'FAIL'}).")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="42444946_lidar")
    ap.add_argument("--app", default="(app name + version)")
    ap.add_argument("--export")
    ap.add_argument("--gt", default=os.path.join(ROOT, "data", "gt", "421337.json"))
    ap.add_argument("--room", default="room")
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "benchmark"))
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)
    out_path = os.path.join(a.results, "HEAD_TO_HEAD.md")
    if not a.export:
        write_table([], a.app, a.capture, out_path, ran=False)
        print("wrote", out_path, "(NOT RUN: no export supplied)")
        return
    ours = json.load(open(os.path.join(a.results, "runs", a.capture, "plan.json")))
    gt = json.load(open(a.gt))
    app = load_roomplan(a.export) if "walls" in open(a.export).read(2000) else json.load(open(a.export))
    rows = compare(ours, gt, app, a.room)
    write_table(rows, a.app, a.capture, out_path, ran=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
