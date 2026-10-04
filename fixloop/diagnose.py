"""Evidence for the fix declaration, run on the REAL LiDAR captures against the source tree being diagnosed.

Question: why are the real LiDAR wall lengths 3.5-22 cm off the laser ground truth, and why do the two captures of the same
room disagree by 19 cm?

Steps (per capture):
  1. run the pipeline and capture the wall-band points + walls of the final per-room fit (spy on `refine_walls`);
  2. register those points onto the laser wall-band slice (data/gt/421337_laser_slice.npz; 4 x 90 deg starts + 2-D ICP) and
     report, per wall, the offset of our plane from the laser's main wall plane (which wall is wrong, by how much);
  3. list every candidate plane in each wall's search window (histogram peaks): support, share of the wall length it
     covers, whether its points reach the ceiling (full-height structure vs furniture / sill), and its offset from the laser
     main plane - i.e. which candidate the pipeline took and which one it should have taken.

    python fixloop/diagnose.py                                    # before tree (fixloop/before_src) -> results/fixloop/diagnosis_before.md
    FLOORSCAN_SRC=fixloop/after_src python fixloop/diagnose.py    # fixed tree -> results/fixloop/diagnosis_after.md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_src = os.environ.get("FLOORSCAN_SRC", os.path.join(ROOT, "fixloop", "before_src"))
sys.path.insert(0, os.path.abspath(_src))

CAPTURES = ("42444946_lidar", "42444949_lidar")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "fixloop")


def coverage(along: np.ndarray, s0: float, s1: float, bin_w: float = 0.10) -> float:
    """Share of the wall's length (10 cm bins) that has >= 3 points of the plane."""
    if s1 - s0 < bin_w:
        return 0.0
    h, _ = np.histogram(along, bins=int((s1 - s0) / bin_w), range=(s0, s1))
    return float(np.mean(h >= 3))


def candidates(u: np.ndarray, lo: float, hi: float, min_frac: float = 0.12, min_count: int = 8, sep: float = 0.05):
    """All local histogram maxima (1 cm bins, 5-bin smoothing as in mode_peak) with >= min_frac of the highest one."""
    from scipy import ndimage as ndi
    nb = int(np.ceil((hi - lo) / 0.01))
    h, _ = np.histogram(u, bins=nb, range=(lo, lo + nb * 0.01))
    hs = ndi.uniform_filter1d(h.astype(float), 5, mode="constant")
    if hs.max() <= 0:
        return []
    pk = [i for i in range(nb) if hs[i] >= min_frac * hs.max() and hs[i] >= min_count
          and (i == 0 or hs[i] >= hs[i - 1]) and (i == nb - 1 or hs[i] > hs[i + 1])]
    pk.sort(key=lambda i: -hs[i])
    keep = []
    for i in pk:
        if all(abs(i - j) * 0.01 >= sep for j in keep):
            keep.append(i)
    out = []
    for i in sorted(keep):
        c = lo + (i + 0.5) * 0.01
        sel = u[np.abs(u - c) < 0.02]
        out.append(float(np.median(sel)) if sel.size else c)
    return out


def register(ours: np.ndarray, laser: np.ndarray, center_ours: np.ndarray, center_laser: np.ndarray):
    """Rigid 2-D transform (R, t) taking our wall-band points onto the laser slice: best of 4 x 90 deg starts + ICP."""
    from scipy.spatial import cKDTree
    tree = cKDTree(laser)
    rng = np.random.default_rng(0)
    P = ours[rng.choice(len(ours), min(len(ours), 40000), replace=False)]
    best = None
    for k in range(4):
        c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        R = np.array([[c, -s], [s, c]])
        t = center_laser - R @ center_ours
        for it in range(40):
            Q = P @ R.T + t
            d, j = tree.query(Q)
            m = d < (0.30 if it < 10 else 0.08)
            if m.sum() < 100:
                break
            A, B = Q[m], laser[j[m]]
            ca, cb = A.mean(0), B.mean(0)
            U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            dR = (U @ Vt).T
            if np.linalg.det(dR) < 0:
                Vt[-1] *= -1
                dR = (U @ Vt).T
            R, t = dR @ R, dR @ (t - ca) + cb
        d, _ = tree.query(P @ R.T + t)
        score = float(np.mean(d < 0.05))
        if best is None or score > best[0]:
            best = (score, R, t)
    return best


def laser_support(laser: np.ndarray, side: str, coord: float, lo: float, hi: float) -> int:
    """Laser slice points within 3 cm of the line `coord` on the given side, inside the wall extent."""
    ax, oth = (0, 1) if side in "WE" else (1, 0)
    m = (np.abs(laser[:, ax] - coord) < 0.03) & (laser[:, oth] > lo) & (laser[:, oth] < hi)
    return int(m.sum())


def wall_vs_laser(walls, R, t, laser, planes):
    """Per wall: laser side, our plane and the main plane in the laser frame, outward offset, laser support on both."""
    gt_plane = dict(zip("WESN", planes[:4]))
    res, segs = {}, []
    for w in walls:
        n = R @ np.array(w.out_n)
        p0, p1 = (np.array([w.coord, w.s0]), np.array([w.coord, w.s1])) if w.orient == "V" else \
            (np.array([w.s0, w.coord]), np.array([w.s1, w.coord]))
        q0, q1 = R @ p0 + t, R @ p1 + t
        segs.append((q0, q1))
        side = ("E" if n[0] > 0 else "W") if abs(n[0]) > abs(n[1]) else ("N" if n[1] > 0 else "S")
        ax = 0 if side in "WE" else 1
        coord = float((q0[ax] + q1[ax]) / 2)
        lo, hi = sorted([q0[1 - ax], q1[1 - ax]])
        off = (1 if side in "EN" else -1) * (coord - gt_plane[side])
        res[w.idx] = dict(side=side, coord=coord, main=float(gt_plane[side]), off=off,
                          sup_ours=laser_support(laser, side, coord, lo, hi),
                          sup_main=laser_support(laser, side, gt_plane[side], lo, hi))
    return res, segs


def figure(path, cap_name, laser, planes, segs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xw, xe, ys, yn = planes[:4]
    fig, ax = plt.subplots(figsize=(8, 8))
    k = max(1, len(laser) // 120000)
    ax.scatter(laser[::k, 0], laser[::k, 1], s=0.1, c="0.5", label="laser wall-band slice")
    ax.plot([xw, xe, xe, xw, xw], [ys, ys, yn, yn, ys], "g-", lw=1.5, label="laser main wall planes (ground truth)")
    for i, (q0, q1) in enumerate(segs):
        ax.plot([q0[0], q1[0]], [q0[1], q1[1]], "r-", lw=1.5, label="our walls (registered)" if i == 0 else None)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"{cap_name}: our walls on the laser scan")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    from floorscan.config import Config
    from floorscan.geometry import room as room_mod
    from floorscan.pipeline.run import run_capture
    import floorscan
    gt = json.load(open(os.path.join(DATA, "gt", "421337.json")))["rooms"][0]
    z = np.load(os.path.join(DATA, "gt", "421337_laser_slice.npz"))
    laser, planes = z["xy"].astype(np.float64), z["planes"]
    cfg = Config()
    which = "before" if os.path.abspath(_src).rstrip("/\\").endswith("before_src") else "after"
    os.makedirs(OUT, exist_ok=True)
    L = [f"# Diagnosis: real LiDAR wall lengths (source tree: `{os.path.relpath(os.path.dirname(floorscan.__file__), ROOT)}`)\n",
         f"Laser ground truth (main wall planes): {gt['w']:.3f} x {gt['d']:.3f} m, ceiling {gt['ceiling_height']:.3f} m. "
         "Offsets: + = our plane lies outside the laser main plane (room too big on that side).\n"]
    stats = {}
    for cap_name in CAPTURES:
        calls = []
        orig = room_mod.refine_walls

        def spy(walls, xy, frames, **kw):
            res = orig(walls, xy, frames, **kw)
            calls.append((walls, xy, res, kw))
            return res

        room_mod.refine_walls = spy
        try:
            out, *_ = run_capture(os.path.join(DATA, cap_name), cfg)
        finally:
            room_mod.refine_walls = orig
        walls_in, xy, walls, kw = calls[-1]  # final refine of the (single) room
        zpts, zc = kw.get("z"), kw.get("zc")
        win = kw.get("win", cfg.wall_window)
        score, R, t = register(xy, laser, np.array([np.mean([w.coord for w in walls if w.orient == "V"]),
                                                    np.mean([w.coord for w in walls if w.orient == "H"])]),
                               np.array([(planes[0] + planes[1]) / 2, (planes[2] + planes[3]) / 2]))
        vs, segs = wall_vs_laser(walls, R, t, laser, planes)
        png = f"diagnosis_{which}_{cap_name}.png"
        figure(os.path.join(OUT, png), cap_name, laser, planes, segs)
        L.append(f"## {cap_name}\n")
        L.append(f"Output walls: {', '.join(f'{w['length']['value']:.3f}' for w in out['rooms'][0]['walls'])} m. "
                 f"Registered onto the laser slice by ICP: {100 * score:.0f} % of our wall-band points within 5 cm of a laser point.\n")
        L.append("| wall | side | our plane vs laser main plane | laser pts on our plane / on main plane |\n|---|---|---|---|")
        for w in walls:
            v = vs[w.idx]
            L.append(f"| {w.idx} | {v['side']} | **{100 * v['off']:+.1f} cm** | {v['sup_ours']} / {v['sup_main']} |")
        L.append(f"\n![{cap_name}]({png})\n")
        L.append(f"Candidate planes in each wall's search window (+-{100 * win:.0f} cm around the initial wall):\n")
        L.append("| wall | side | candidate vs laser main plane | our points within 3 cm | share of wall length covered | "
                 "reaches ceiling (97th pct z >= ceiling - 0.5 m) | taken by pipeline |\n|---|---|---|---|---|---|---|")
        for w_in, w in zip(walls_in, walls):
            n = np.array(w_in.out_n)
            sgn = n[0] + n[1]
            ax = 0 if w_in.orient == "V" else 1
            m = (xy[:, 1 - ax] > w_in.s0 + 0.12) & (xy[:, 1 - ax] < w_in.s1 - 0.12)
            u, along, zz = sgn * xy[m, ax], xy[m, 1 - ax], zpts[m]
            chosen = sgn * w.coord
            for c in candidates(u, sgn * w_in.coord - win, sgn * w_in.coord + win):
                inl = np.abs(u - c) < 0.03
                top = float(np.percentile(zz[inl], 97)) if inl.sum() > 30 else float("nan")
                rec = dict(wall=w.idx, side=vs[w.idx]["side"], off=vs[w.idx]["off"] + (c - chosen), n=int(inl.sum()),
                           cov=coverage(along[inl], w_in.s0, w_in.s1), full=bool(top >= zc - 0.5), taken=bool(abs(c - chosen) < 0.015))
                stats.setdefault(cap_name, []).append(rec)
                L.append(f"| {w.idx} | {rec['side']} | {100 * rec['off']:+.1f} cm | {rec['n']} | {100 * rec['cov']:.0f} % | "
                         f"{'yes' if rec['full'] else 'no'} | {'**taken**' if rec['taken'] else ''} |")
        L.append("")
    json.dump(stats, open(os.path.join(OUT, f"diagnosis_{which}.json"), "w"), indent=1)
    open(os.path.join(OUT, f"diagnosis_{which}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
