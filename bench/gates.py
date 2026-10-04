"""Gate definitions (case-study Part 2) and their evaluation on top of bench/metrics.py results.

ASSUMPTIONS (the Round-1 document that defines the base LiDAR gates was not provided with the PDF):
  * LiDAR wall length: |error| <= max(2 cm, 0.5 %) per wall
  * video / photo opening widths and ceiling heights use the same relative tolerance as their wall lengths
    (3 % / 8 %)
Everything else is quoted from the PDF.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

REL = {"lidar": None, "video": 0.03, "photos": 0.08}


def wall_ok(tier: str, err: float, gt_len: float) -> bool:
    if tier == "lidar":
        return abs(err) <= max(0.02, 0.005 * gt_len)
    return abs(err) <= REL[tier] * gt_len


def height_ok(tier: str, err: float, gt_h: float) -> bool:
    if tier == "lidar":
        return abs(err) <= 0.015
    return abs(err) <= REL[tier] * gt_h


def tol_open(tier: str):
    return (0.02, 0.0) if tier == "lidar" else (0.0, REL[tier])


def room_rows(m: Dict, gt: Dict, tier: str):
    """Per-room wall / height rows for the report."""
    g = {r["name"]: r for r in gt["rooms"]}
    rows = []
    for r in m["rooms"]:
        G = g[r["gt_name"]]
        for wl, e in r["wall_err"].items():
            L = G["walls"][wl]
            rows.append(dict(room=r["gt_name"], kind="wall " + wl, gt=L, err=e, ok=wall_ok(tier, e, L)))
        rows.append(dict(room=r["gt_name"], kind="ceiling", gt=G["ceiling_height"], err=r["height_err"],
                         ok=height_ok(tier, r["height_err"], G["ceiling_height"])))
    return rows


def gate_summary(m: Dict, gt: Dict, tier: str) -> Dict:
    rows = room_rows(m, gt, tier)
    walls = [x for x in rows if x["kind"].startswith("wall")]
    ceil = [x for x in rows if x["kind"] == "ceiling"]
    ops = m["openings"]
    n_gt = sum(1 for o in ops if o["status"] in ("tp", "missed"))
    n_ph = sum(1 for o in ops if o["status"] == "phantom")
    n_ok = sum(1 for o in ops if o["status"] == "tp" and o["ok"])
    rate = n_ok / max(n_gt + n_ph, 1) if (n_gt + n_ph) else None
    missing_walls = sum(max(0, 4 - r["n_walls_found"]) for r in m["rooms"]) + 4 * m["rooms_missing"]
    out = dict(
        tier=tier,
        walls_n=len(walls), walls_ok=sum(x["ok"] for x in walls),
        walls_max_abs_err=max([abs(x["err"]) for x in walls], default=None),
        walls_rel_err_mean=float(np.mean([abs(x["err"]) / x["gt"] for x in walls])) if walls else None,
        ceilings_n=len(ceil), ceilings_ok=sum(x["ok"] for x in ceil),
        ceilings_max_abs_err=max([abs(x["err"]) for x in ceil], default=None),
        openings_gt=n_gt, openings_tp=sum(1 for o in ops if o["status"] == "tp"), openings_missed=sum(1 for o in ops if o["status"] == "missed"),
        openings_phantom=n_ph, openings_ok=n_ok, openings_rate=rate, openings_pass=None if rate is None else bool(rate >= 0.85),
        footprint_err_pct=m["footprint_err_pct"], iou=m["iou"],
        adjacency_precision=m["adjacency"]["precision"], adjacency_recall=m["adjacency"]["recall"],
        overlap_area=m["overlap_area"], rooms_missing=m["rooms_missing"], rooms_extra=m["rooms_extra"],
        walls_missing=missing_walls,
    )
    out["walls_pass"] = bool(out["walls_n"] and out["walls_ok"] == out["walls_n"] and missing_walls == 0)
    out["ceiling_pass"] = bool(out["ceilings_n"] and out["ceilings_ok"] == out["ceilings_n"])
    lim = 2.0 if tier == "lidar" else 100 * REL[tier]
    out["footprint_pass"] = bool(abs(out["footprint_err_pct"]) <= (8.0 if tier == "photos" else lim)
                                 and out["adjacency_precision"] == 1.0 and out["adjacency_recall"] == 1.0
                                 and out["overlap_area"] < 0.05 and m["rooms_missing"] == 0)
    d = m.get("damage", {})
    out["damage_tp"], out["damage_fp"], out["damage_fn"] = d.get("tp", 0), d.get("fp", 0), d.get("fn", 0)
    out["damage_class_ok"] = d.get("class_ok", 0)
    return out


def repeatability(out_a: Dict, out_b: Dict, gt: Dict, tier: str):
    """Per GT wall: |L_a - L_b| <= max(1 cm, 0.5 %); ceiling spread <= 1 cm.  Also reports bias vs ground truth so
    the report can say 'repeatable but biased' vs 'unrepeatable'."""
    from bench.metrics import align_room, transform_pts, WALLS, gt_wall_seg
    rg = next(r for r in gt["rooms"])
    res = {}
    for key, o in (("a", out_a), ("b", out_b)):
        ro = o["rooms"][0]
        la = align_room(ro, rg)
        lens = {}
        for w in ro["walls"]:
            p0, p1 = transform_pts(la, [w["p0"], w["p1"]])
            mid = 0.5 * (p0 + p1)
            best, bd = None, 9
            for wl in WALLS:
                (ax, ay), (bx, by) = gt_wall_seg(rg, wl)
                horiz = abs(p1[0] - p0[0]) > abs(p1[1] - p0[1])
                if horiz != (wl in "SN"):
                    continue
                dd = np.linalg.norm(mid - np.array([(ax + bx) / 2, (ay + by) / 2]))
                if dd < bd:
                    best, bd = wl, dd
            if best:
                lens[best] = float(np.linalg.norm(p1 - p0))
        res[key] = dict(walls=lens, height=ro["ceiling_height"]["value"])
    rows = []
    for wl in WALLS:
        if wl in res["a"]["walls"] and wl in res["b"]["walls"]:
            la_, lb_ = res["a"]["walls"][wl], res["b"]["walls"][wl]
            gtL = rg["walls"][wl]
            tol = max(0.01, 0.005 * gtL)
            rows.append(dict(wall=wl, a=la_, b=lb_, diff=la_ - lb_, tol=tol, ok=abs(la_ - lb_) <= tol,
                             bias_a=la_ - gtL, bias_b=lb_ - gtL))
    hs = (res["a"]["height"], res["b"]["height"])
    return dict(walls=rows, height_a=hs[0], height_b=hs[1], height_spread=abs(hs[0] - hs[1]),
                height_ok=abs(hs[0] - hs[1]) <= 0.01, height_bias=(hs[0] - rg["ceiling_height"], hs[1] - rg["ceiling_height"]),
                walls_ok=all(r["ok"] for r in rows) and len(rows) == 4)


def classify_repeat(rep: Dict, tier: str, gt_h: float) -> str:
    spread_ok = rep["walls_ok"] and rep["height_ok"]
    bias_ok = all(abs(b) <= (0.015 if tier == "lidar" else REL[tier] * gt_h) for b in rep["height_bias"])
    if spread_ok and bias_ok:
        return "repeatable and unbiased (pass)"
    if spread_ok and not bias_ok:
        return "REPEATABLE BUT BIASED"
    return "UNREPEATABLE"
