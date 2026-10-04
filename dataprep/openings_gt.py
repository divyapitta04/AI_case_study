"""Opening ground truth (doors, windows, passages) from the laser scan of a benchmark room.

    python dataprep/openings_gt.py --visit 467293            # pass 1: elevation image of every wall -> data/gt/openings/
    (a person marks each opening roughly in data/gt/openings_annot.json: wall, u0, u1, sill, top, kind)
    python dataprep/openings_gt.py --visit 467293            # pass 2: edges snapped to the laser -> data/gt/<visit>.json

Inputs: the room's laser points in the ground-truth frame written by dataprep/room_gt.py (x from the W wall plane, y from
the S wall plane, z from the floor).  Elevation images show the wall face in its laser colour (within 1.5 cm of the plane),
what the laser saw *through* the wall plane in blue (open doors, passages, window glass), and objects in front in grey.
Refinement (no pipeline involvement): each marked edge is moved, within +-12 cm, to where the share of wall-face points per
1 cm column (or row, for sill / head) crosses 50 %; a closed door leaf sits a few cm behind the face and counts as opening.
Width convention: clear width between the jamb faces at the wall plane; sill 0 for doors and passages.  Recesses that are
part of the room (bay windows, alcoves) are declared as ignore regions: an opening reported there is neither credited nor
counted as phantom (bench/metrics.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GT_DIR = os.path.join(ROOT, "data", "gt")
FACE = 0.015  # m: within this of the plane = wall face


def wall_frame(G: np.ndarray, side: str, w: float, d: float):
    """(u along the wall from its start, outward distance from the plane, z) for one GT wall."""
    x, y, z = G[:, 0], G[:, 1], G[:, 2]
    if side == "S":
        return x, -y, z, w
    if side == "N":
        return x, y - d, z, w
    if side == "W":
        return y, -x, z, d
    return y, x - w, z, d


def elevation(u, o, z, rgb, L, H, res=0.01):
    nu, nv = int(np.ceil((L + 0.4) / res)), int(np.ceil(H / res))
    img = np.full((nv, nu, 3), 255, np.uint8)
    for sel, col in (((o < -0.06) & (o > -1.0), None), ((o > 0.06) & (o < 3.0), "beyond"), ((np.abs(o) < FACE), "face")):
        iu = np.clip(((u[sel] + 0.2) / res).astype(int), 0, nu - 1)
        iv = np.clip(((H - z[sel]) / res).astype(int), 0, nv - 1)
        if col is None:
            img[iv, iu] = (200, 200, 200)
        elif col == "beyond":
            depth = np.clip(o[sel] / 2.0, 0, 1)
            img[iv, iu] = np.stack([60 + 80 * depth, 110 + 80 * depth, np.full_like(depth, 255)], 1).astype(np.uint8)
        else:
            img[iv, iu] = rgb[sel]
    return img


def refine_edge(u, o, z, guess, lo_z, hi_z, along="u", inside_sign=+1, span=0.12, res=0.01):
    """Coordinate (u for jambs, z for sill/head) where the wall-face share crosses 50 % near `guess`.
    inside_sign: +1 if the opening lies on the + side of the edge."""
    coord, other = (u, z) if along == "u" else (z, u)
    m = (other > lo_z) & (other < hi_z) & (np.abs(coord - guess) < span + 0.03) & (o > -0.05)
    bins = np.arange(guess - span, guess + span + res, res)
    tot = np.histogram(coord[m], bins=bins)[0].astype(float)
    face = np.histogram(coord[m & (o < FACE)], bins=bins)[0].astype(float)  # at or in front of the plane (casing) = solid
    frac = np.where(tot > 0, face / np.maximum(tot, 1), np.nan)
    c = 0.5 * (bins[:-1] + bins[1:])
    ok = ~np.isnan(frac)
    if ok.sum() < 5:
        return guess, False
    f = np.interp(c, c[ok], frac[ok])
    if inside_sign < 0:
        f, c = f[::-1], c[::-1]
    # walk from the wall side towards the opening: first crossing below 0.5 that persists for 3 cm (a single see-through
    # column at a jamb corner - grazing rays - is not the edge)
    for i in range(1, len(f)):
        if f[i - 1] >= 0.5 > f[i] and all(x < 0.5 for x in f[i:i + 3]):
            t = (f[i - 1] - 0.5) / max(f[i - 1] - f[i], 1e-6)
            return float(c[i - 1] + t * (c[i] - c[i - 1])), True
    return guess, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visit", required=True)
    ap.add_argument("--laser_root", default=os.path.join(ROOT, "..", "arkitscenes_raw", "laser"))
    a = ap.parse_args()
    z_ = np.load(os.path.join(a.laser_root, a.visit, "room_gt_points.npz"))
    G, rgb, w, d, H = z_["xyz"].astype(np.float64), z_["rgb"], float(z_["w"]), float(z_["d"]), float(z_["H"])
    out_dir = os.path.join(GT_DIR, "openings")
    os.makedirs(out_dir, exist_ok=True)
    ann_path = os.path.join(GT_DIR, "openings_annot.json")
    ann = json.load(open(ann_path)) if os.path.exists(ann_path) else {}
    marks = ann.get(a.visit, {}).get("openings")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axs = plt.subplots(4, 1, figsize=(14, 16))
    results = []
    for ax, side in zip(axs, "SENW"):
        u, o, z, L = wall_frame(G, side, w, d)
        sel = (u > -0.2) & (u < L + 0.2)
        img = elevation(u[sel], o[sel], z[sel], rgb[sel], L, H)
        ax.imshow(img, extent=(-0.2, L + 0.2, 0, H), aspect="equal", interpolation="nearest")
        ax.set_xticks(np.arange(0, L + 0.01, 0.25), minor=True)
        ax.set_xticks(np.arange(0, L + 0.01, 0.5))
        ax.grid(True, which="both", lw=0.3, color="r", alpha=0.4)
        ax.set_title(f"visit {a.visit} wall {side} (length {L:.3f} m, u from the wall start; blue = seen through the plane)",
                     fontsize=9)
        for mk in (marks or []):
            if mk["wall"] != side:
                continue
            zl, zh = max(mk.get("sill", 0) + 0.15, 0.3), min(mk["top"] - 0.15, 2.0)
            # an edge marked "fixed" is a room corner (the perpendicular wall's face), not a jamb: kept as marked
            u0 = mk["u0"] if "u0" in mk.get("fixed", []) else refine_edge(u, o, z, mk["u0"], zl, zh, "u", +1)[0]
            u1 = mk["u1"] if "u1" in mk.get("fixed", []) else refine_edge(u, o, z, mk["u1"], zl, zh, "u", -1)[0]
            top = refine_edge(u, o, z, mk["top"], u0 + 0.1, u1 - 0.1, "z", -1)[0]
            sill = 0.0 if mk["kind"] in ("door", "passage") else refine_edge(u, o, z, mk["sill"], u0 + 0.1, u1 - 0.1, "z", +1)[0]
            results.append(dict(id=f"room.{side}.{len(results)}", wall=side, center=round((u0 + u1) / 2, 4), width=round(u1 - u0, 4),
                                height=round(top - sill, 4), sill=round(sill, 4), kind=mk["kind"], link=None))
            ax.add_patch(Rectangle((u0, sill), u1 - u0, top - sill, fill=False, ec="m", lw=1.5))
            ax.text(u0, top + 0.03, f"{mk['kind']} {100 * (u1 - u0):.1f} cm", color="m", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{a.visit}_walls.png"), dpi=80)
    if marks is None:
        print(f"wrote {out_dir}/{a.visit}_walls.png - mark the openings in {ann_path} and rerun")
        return
    gpath = os.path.join(GT_DIR, f"{a.visit}.json")
    gt = json.load(open(gpath))
    gt["rooms"][0]["openings"] = results
    gt["rooms"][0]["ignore_openings"] = [dict(wall=z["wall"], u0=z["u0"], u1=z["u1"], note=z.get("note", ""))
                                         for z in ann[a.visit].get("ignore", [])]
    gt["openings_reliable"] = True
    gt["openings_method"] = ("dataprep/openings_gt.py: openings marked roughly on laser elevation images by a person, edges snapped "
                             "to the laser wall (50 % solid-share crossing, 1 cm columns)")
    gt.pop("note_openings", None)
    json.dump(gt, open(gpath, "w"), indent=1)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
