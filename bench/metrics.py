"""Score a floorscan output JSON against ground truth (laser-derived for the real benchmark; never imported by the pipeline)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.ops import unary_union

WALLS = "SENW"


def gt_rect(r: Dict) -> Polygon:
    return Polygon([(r["x0"], r["y0"]), (r["x0"] + r["w"], r["y0"]), (r["x0"] + r["w"], r["y0"] + r["d"]),
                    (r["x0"], r["y0"] + r["d"])])


def gt_wall_seg(r: Dict, wl: str):
    x0, y0, w, d = r["x0"], r["y0"], r["w"], r["d"]
    return {"S": ((x0, y0), (x0 + w, y0)), "N": ((x0, y0 + d), (x0 + w, y0 + d)),
            "W": ((x0, y0), (x0, y0 + d)), "E": ((x0 + w, y0), (x0 + w, y0 + d))}[wl]


def gt_opening_center(r: Dict, o: Dict) -> np.ndarray:
    (ax, ay), (bx, by) = gt_wall_seg(r, o["wall"])
    d = np.array([bx - ax, by - ay]) / np.hypot(bx - ax, by - ay)
    return np.array([ax, ay]) + d * o["center"]


def _rot(poly_pts: np.ndarray, k: int, c: np.ndarray) -> np.ndarray:
    a = k * np.pi / 2
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (R @ (poly_pts - c).T).T + c


def align(out: Dict, gt: Dict) -> Dict:
    """Best rigid (k*90deg + translation) transform from pipeline plan frame to GT frame + room assignment."""
    o_polys = [np.array(r["polygon"]) for r in out["rooms"]]
    g_polys = [gt_rect(r) for r in gt["rooms"]]
    g_union = unary_union(g_polys)
    names_ok = all(r.get("name") for r in out["rooms"]) and \
        {r["name"] for r in out["rooms"]} <= {r["name"] for r in gt["rooms"]}
    c0 = np.concatenate(o_polys).mean(0)
    per_k = []
    for k in range(4):
        rp = [_rot(p, k, c0) for p in o_polys]
        ou = unary_union([Polygon(p) for p in rp])
        t0 = np.array(g_union.centroid.coords[0]) - np.array(ou.centroid.coords[0])
        bk = None
        for dx in np.arange(-0.6, 0.61, 0.06):
            for dy in np.arange(-0.6, 0.61, 0.06):
                t = t0 + np.array([dx, dy])
                o2 = translate(ou, t[0], t[1])
                iou = o2.intersection(g_union).area / max(o2.union(g_union).area, 1e-9)
                if bk is None or iou > bk[0]:
                    bk = (iou, k, t)
        for step in (0.02, 0.005):
            cur = bk
            for dx in np.arange(-3, 4) * step:
                for dy in np.arange(-3, 4) * step:
                    tt = bk[2] + np.array([dx, dy])
                    o2 = translate(ou, tt[0], tt[1])
                    v = o2.intersection(g_union).area / max(o2.union(g_union).area, 1e-9)
                    if v > cur[0]:
                        cur = (v, k, tt)
            bk = cur
        per_k.append(bk)
    top = max(p[0] for p in per_k)
    cands = [p for p in per_k if p[0] >= top - 0.01]

    def open_score(cand):
        iou_, k_, t_ = cand
        sc = 0
        for r in out["rooms"]:
            wm = {w["id"]: _rot(np.array([w["p0"], w["p1"]]), k_, c0) + t_ for w in r["walls"]}
            for o in r["openings"]:
                p0, p1 = wm[o["wall_id"]]
                d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
                cc = p0 + d * o["center_offset"]["value"]
                for gr in gt["rooms"]:
                    for go in gr["openings"]:
                        if np.linalg.norm(gt_opening_center(gr, go) - cc) < 0.4:
                            sc += 1
        return sc

    best = max(cands, key=lambda c: (open_score(c), c[0])) if len(cands) > 1 else cands[0]
    iou, k, t = best
    tp = [Polygon(_rot(p, k, c0) + t) for p in o_polys]
    # assignment
    if names_ok:
        gi = {r["name"]: i for i, r in enumerate(gt["rooms"])}
        pairs = [(i, gi[r["name"]]) for i, r in enumerate(out["rooms"])]
    else:
        M = np.zeros((len(tp), len(g_polys)))
        for i, p in enumerate(tp):
            for j, q in enumerate(g_polys):
                M[i, j] = p.intersection(q).area / max(p.union(q).area, 1e-9)
        ri, ci = linear_sum_assignment(-M)
        pairs = [(i, j) for i, j in zip(ri, ci) if M[i, j] > 0.3]
    return dict(k=k, c0=c0, t=t, iou=iou, pairs=pairs, polys=tp)


def align_room(ro: Dict, rg: Dict) -> Dict:
    """Align ONE pipeline room to its GT room (k*90 deg + translation), independent of the stitch quality."""
    P = np.array(ro["polygon"])
    c0 = P.mean(0)
    g = gt_rect(rg)
    best = None
    for k in range(4):
        Pk = _rot(P, k, c0)
        ctr = 0.5 * (Pk.min(0) + Pk.max(0))
        t = np.array([rg["x0"] + rg["w"] / 2, rg["y0"] + rg["d"] / 2]) - ctr
        poly = Polygon(Pk + t)
        iou = poly.intersection(g).area / max(poly.union(g).area, 1e-9)
        sc = 0
        wm = {w["id"]: _rot(np.array([w["p0"], w["p1"]]), k, c0) + t for w in ro["walls"]}
        for o in ro["openings"]:
            p0, p1 = wm[o["wall_id"]]
            d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
            cc = p0 + d * o["center_offset"]["value"]
            for go in rg["openings"]:
                if np.linalg.norm(gt_opening_center(rg, go) - cc) < 0.45:
                    sc += 1
        key = (round(iou * 20), sc, iou)
        if best is None or key > best[0]:
            best = (key, k, t)
    return dict(k=best[1], c0=c0, t=best[2])


def transform_pts(a: Dict, pts: np.ndarray) -> np.ndarray:
    return _rot(np.asarray(pts, float), a["k"], a["c0"]) + a["t"]


def _gt_damage_center(rg: Dict, d: Dict):
    """(surface kind, 3-D centre in GT frame) of a GT damage region."""
    u0, v0, u1, v1 = d["bounds"]
    uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    s = d["surface"]
    if s in "SENW" and len(s) == 1:
        (ax, ay), (bx, by) = gt_wall_seg(rg, s)
        dd = np.array([bx - ax, by - ay]) / np.hypot(bx - ax, by - ay)
        p = np.array([ax, ay]) + dd * uc
        return "wall", np.array([p[0], p[1], vc])
    z = rg["ceiling_height"] if s == "ceiling" else 0.0
    return s, np.array([rg["x0"] + uc, rg["y0"] + vc, z])


def eval_damage(out: Dict, gt: Dict, a: Dict) -> Dict:
    """Match pipeline damage regions to GT regions by surface type and 3-D centre; report class accuracy,
    metric extent errors and false positives."""
    res = dict(tp=0, class_ok=0, fn=0, fp=0, items=[], calib=[])
    for (i, j) in a["pairs"]:
        ro, rg = out["rooms"][i], gt["rooms"][j]
        a = align_room(ro, rg)
        det = []
        for d in ro["damage"]:
            sid = d["surface_id"]
            loc = d["location"]
            xy = transform_pts(a, np.array([loc["plan_xy"]]))[0]
            kind = "ceiling" if sid.endswith(".ceiling") else ("floor" if sid.endswith(".floor") else "wall")
            det.append((d, kind, np.array([xy[0], xy[1], loc["z"]])))
        gts = [(d, *_gt_damage_center(rg, d)) for d in rg["damage"]]
        used = set()
        for (d, kind, c) in det:
            best, bd = None, 0.6
            for gi, (gd, gk, gc) in enumerate(gts):
                if gi in used or gk != kind:
                    continue
                dist = np.linalg.norm(c - gc)
                if dist < bd:
                    best, bd = gi, dist
            if best is None:
                res["fp"] += 1
                res["items"].append(dict(status="fp", cls=d["class"], surface=d["surface_id"], conf=d["confidence"]))
                continue
            used.add(best)
            gd = gts[best][0]
            ext = d["extent"]
            res["tp"] += 1
            ok = d["class"] == gd["cls"]
            res["class_ok"] += int(ok)
            w_gt, h_gt = (gd["width"], gd["height"])
            area_e = ext["area"]["value"] - gd["area"]
            res["calib"].append(dict(kind="extent", err=area_e, half=(ext["area"]["ci95"][1] - ext["area"]["ci95"][0]) / 2,
                                     inside=bool(ext["area"]["ci95"][0] <= gd["area"] <= ext["area"]["ci95"][1]),
                                     val=gd["area"], tier=out["capture"]["tier"]))
            res["items"].append(dict(status="tp", cls=d["class"], gt_cls=gd["cls"], class_ok=ok,
                                     width_err=ext["width"]["value"] - w_gt, height_err=ext["height"]["value"] - h_gt,
                                     area_err=area_e, area_rel=area_e / gd["area"], conf=d["confidence"],
                                     area_inside=bool(ext["area"]["ci95"][0] <= gd["area"] <= ext["area"]["ci95"][1])))
        for gi, (gd, gk, gc) in enumerate(gts):
            if gi not in used:
                res["fn"] += 1
                res["items"].append(dict(status="fn", gt_cls=gd["cls"], surface=gd["surface"]))
    return res


def evaluate(out: Dict, gt: Dict, tol_open: float = 0.02, tol_open_rel: float = 0.0) -> Dict:
    if not out["rooms"]:  # nothing reconstructed: every room, wall and opening is missed
        ops = [dict(room=None, status="missed", gt=o["id"], width_gt=o["width"], kind=o["kind"], ok=False)
               for r in gt["rooms"] for o in r["openings"]]
        return dict(iou=0.0, rooms=[], openings=ops, calib=[], footprint_err_pct=-100.0, rooms_missing=len(gt["rooms"]),
                    rooms_extra=0, adjacency=dict(pred=0, gt=len(gt["adjacency"]), tp=0, precision=0.0,
                                                  recall=0.0 if gt["adjacency"] else 1.0, wrong=[], missing=[]),
                    overlap_area=0.0, align=dict(k=0, iou=0.0),
                    damage=dict(tp=0, class_ok=0, fn=sum(len(r["damage"]) for r in gt["rooms"]), fp=0, items=[], calib=[]))
    a = align(out, gt)
    m: Dict = dict(iou=a["iou"], rooms=[], openings=[], damage=[], calib=[])
    n_gt_rooms = len(gt["rooms"])
    matched_gt = set()
    for (i, j) in a["pairs"]:
        ro, rg = out["rooms"][i], gt["rooms"][j]
        la = align_room(ro, rg)
        matched_gt.add(j)
        rm = dict(out_id=ro["id"], gt_name=rg["name"])
        # walls
        walls_o = []
        for w in ro["walls"]:
            p0, p1 = transform_pts(la, [w["p0"], w["p1"]])
            walls_o.append((p0, p1, w))
        used = set()
        wl_err = {}
        for p0, p1, w in walls_o:
            mid = 0.5 * (p0 + p1)
            best, bd = None, 1e9
            for wl in WALLS:
                if wl in used:
                    continue
                (ax, ay), (bx, by) = gt_wall_seg(rg, wl)
                gm = np.array([(ax + bx) / 2, (ay + by) / 2])
                # orientation must agree
                horiz_o = abs(p1[0] - p0[0]) > abs(p1[1] - p0[1])
                horiz_g = wl in "SN"
                if horiz_o != horiz_g:
                    continue
                dd = np.linalg.norm(mid - gm)
                if dd < bd:
                    best, bd = wl, dd
            if best is not None and bd < 1.0:
                used.add(best)
                L = float(np.linalg.norm(p1 - p0))
                wl_err[best] = L - rg["walls"][best]
                lo, hi = w["length"]["ci95"]
                m["calib"].append(dict(kind="length", err=L - rg["walls"][best],
                                       half=(hi - lo) / 2, inside=bool(lo <= rg["walls"][best] <= hi),
                                       val=rg["walls"][best], tier=out["capture"]["tier"]))
        rm["wall_err"] = wl_err
        rm["n_walls_expected"] = 4
        rm["n_walls_found"] = len(wl_err)
        h = ro["ceiling_height"]
        rm["height_err"] = h["value"] - rg["ceiling_height"]
        m["calib"].append(dict(kind="height", err=rm["height_err"], half=(h["ci95"][1] - h["ci95"][0]) / 2,
                               inside=bool(h["ci95"][0] <= rg["ceiling_height"] <= h["ci95"][1]),
                               val=rg["ceiling_height"], tier=out["capture"]["tier"]))
        ar = ro["floor_area"]
        rm["area_err_pct"] = 100 * (ar["value"] - rg["floor_area"]) / rg["floor_area"]
        m["calib"].append(dict(kind="area", err=ar["value"] - rg["floor_area"],
                               half=(ar["ci95"][1] - ar["ci95"][0]) / 2,
                               inside=bool(ar["ci95"][0] <= rg["floor_area"] <= ar["ci95"][1]),
                               val=rg["floor_area"], tier=out["capture"]["tier"]))
        # openings
        gts = list(rg["openings"])
        w_by_id = {w["id"]: (p0, p1) for p0, p1, w in walls_o}
        det = []
        for o in ro["openings"]:
            p0, p1 = w_by_id[o["wall_id"]]
            d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
            c = p0 + d * o["center_offset"]["value"]
            det.append((o, c))
        pairs_o = []
        cost = np.full((len(det), len(gts)), 1e3)
        for di, (o, c) in enumerate(det):
            for gi_, go in enumerate(gts):
                gc = gt_opening_center(rg, go)
                dd = np.linalg.norm(c - gc)
                if dd < 0.5:
                    cost[di, gi_] = dd
        if det and gts:
            ri, ci = linear_sum_assignment(cost)
            pairs_o = [(r_, c_) for r_, c_ in zip(ri, ci) if cost[r_, c_] < 1e3]
        matched_d = {p[0] for p in pairs_o}
        matched_g = {p[1] for p in pairs_o}
        for di_, gi_ in pairs_o:
            o, _ = det[di_]
            go = gts[gi_]
            w_ = o["width"]
            e = w_["value"] - go["width"]
            hh = o["height"]
            m["openings"].append(dict(room=ro["id"], status="tp", gt=go["id"], width_err=e,
                                      height_err=hh["value"] - go["height"],
                                      sill_err=o["sill"]["value"] - go["sill"], width_gt=go["width"],
                                      type_ok=(o["type"] == go["kind"]) or ({o["type"], go["kind"]} <= {"door", "passage"}),
                                      ok=bool(abs(e) <= tol_open + tol_open_rel * go["width"])))
            m["calib"].append(dict(kind="opening_width", err=e, half=(w_["ci95"][1] - w_["ci95"][0]) / 2,
                                   inside=bool(w_["ci95"][0] <= go["width"] <= w_["ci95"][1]), val=go["width"],
                                   tier=out["capture"]["tier"]))
            m["calib"].append(dict(kind="opening_height", err=hh["value"] - go["height"],
                                   half=(hh["ci95"][1] - hh["ci95"][0]) / 2,
                                   inside=bool(hh["ci95"][0] <= go["height"] <= hh["ci95"][1]), val=go["height"],
                                   tier=out["capture"]["tier"]))
        ignore = [(gt_wall_seg(rg, z["wall"]), z) for z in rg.get("ignore_openings", [])]
        for di_ in range(len(det)):
            if di_ not in matched_d:
                c = det[di_][1]
                hit = False
                for ((ax_, ay_), (bx_, by_)), z in ignore:  # recess (bay, alcove) declared in the GT: not scored
                    dvec = np.array([bx_ - ax_, by_ - ay_]) / np.hypot(bx_ - ax_, by_ - ay_)
                    s_ = float((c - np.array([ax_, ay_])) @ dvec)
                    rv = c - np.array([ax_, ay_])
                    off = abs(float(dvec[0] * rv[1] - dvec[1] * rv[0]))
                    hit |= (z["u0"] - 0.1 <= s_ <= z["u1"] + 0.1) and off < 0.6
                m["openings"].append(dict(room=ro["id"], status="ignored" if hit else "phantom",
                                          width=det[di_][0]["width"]["value"], ok=False))
        for gi_ in range(len(gts)):
            if gi_ not in matched_g:
                m["openings"].append(dict(room=ro["id"], status="missed", gt=gts[gi_]["id"], width_gt=gts[gi_]["width"],
                                          kind=gts[gi_]["kind"], ok=False))
        m["rooms"].append(rm)
    m["rooms_missing"] = n_gt_rooms - len(matched_gt)
    m["rooms_extra"] = len(out["rooms"]) - len(a["pairs"])
    # plan-level
    t = gt.get("wall_thickness", 0.12)
    gfoot = unary_union([gt_rect(r).buffer(t / 2, join_style=2) for r in gt["rooms"]]).area
    ofoot = out["plan"]["footprint_area"]
    if gt.get("coverage") == "captured_room":
        # ground truth covers only the captured room: spaces the capture saw through doors (corridor, next room) are real
        # but unscored, so compare the floor area of the room(s) matched to the ground truth (reported: rooms_extra)
        fa = [out["rooms"][i]["floor_area"] for i, _ in a["pairs"]]
        gfoot = float(sum(r["floor_area"] for r in gt["rooms"]))
        ofoot = dict(value=float(sum(x["value"] for x in fa)), ci95=[float(sum(x["ci95"][0] for x in fa)),
                                                                     float(sum(x["ci95"][1] for x in fa))])
    m["footprint_err_pct"] = 100 * (ofoot["value"] - gfoot) / gfoot
    m["calib"].append(dict(kind="footprint", err=ofoot["value"] - gfoot, half=(ofoot["ci95"][1] - ofoot["ci95"][0]) / 2,
                           inside=bool(ofoot["ci95"][0] <= gfoot <= ofoot["ci95"][1]), val=gfoot,
                           tier=out["capture"]["tier"]))
    # adjacency
    idmap = {out["rooms"][i]["id"]: gt["rooms"][j]["name"] for i, j in a["pairs"]}
    o_adj = {tuple(sorted((idmap.get(x["a"]), idmap.get(x["b"])))) for x in out["plan"]["adjacency"]
             if idmap.get(x["a"]) and idmap.get(x["b"])}
    g_adj = {tuple(x) for x in gt["adjacency"]}
    tp = len(o_adj & g_adj)
    m["adjacency"] = dict(pred=len(o_adj), gt=len(g_adj), tp=tp,
                          precision=tp / len(o_adj) if o_adj else (1.0 if not g_adj else 0.0),
                          recall=tp / len(g_adj) if g_adj else 1.0, wrong=sorted(o_adj - g_adj),
                          missing=sorted(g_adj - o_adj))
    m["overlap_area"] = float(sum(x["area"] for x in out["plan"]["overlaps"]))
    m["align"] = dict(k=int(a["k"]), iou=float(a["iou"]))
    m["damage"] = eval_damage(out, gt, dict(pairs=a["pairs"]))
    m["calib"].extend(m["damage"]["calib"])
    return m
