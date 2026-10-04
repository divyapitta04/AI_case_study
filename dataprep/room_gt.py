"""Ground truth for one benchmark room from its FARO laser scans (ARKitScenes visit).

    python dataprep/room_gt.py --visit 467293            # pass 1: registration + inspection figure (+ candidate planes)
    (write the main wall-plane windows for the visit into data/gt/windows.json after looking at the figure)
    python dataprep/room_gt.py --visit 467293            # pass 2: windows found -> data/gt/<visit>.json (+ figure, slice)

Independent of the floorscan pipeline (only the session's raw depth is used, to find which part of the venue was captured):
  1. every LiDAR session of the visit is registered to the venue's laser scans (dataprep/register.py);
  2. the laser stations standing inside the captured room are kept (a venue can have other rooms / split levels that
     overlap in plan view), cropped to the room footprint;
  3. floor and ceiling = strongest horizontal planes; room axes = the yaw with the sharpest wall histograms;
  4. each main wall plane = strongest plane inside a search window chosen by a person looking at the figure (main face, not
     the back of a bay window / alcove: the same convention as a tape measurement), range^2-weighted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dataprep.register import apply, register, session_points  # noqa: E402

GT_DIR = os.path.join(ROOT, "data", "gt")


def peak(v, lo, hi, bw=0.01, w=None):
    m = (v > lo) & (v < hi)
    if m.sum() < 50:
        return None, 0
    h, e = np.histogram(v[m], bins=max(int(round((hi - lo) / bw)), 1), range=(lo, hi), weights=None if w is None else w[m])
    hs = np.convolve(h, np.ones(3) / 3, "same")
    i = int(np.argmax(hs))
    c = e[i] + bw / 2
    sel = v[m][np.abs(v[m] - c) < 0.02]
    return (float(np.median(sel)) if sel.size else float(c)), float(hs[i])


def extreme_plane(v, lo, hi, which, frac=0.3, bw=0.01):
    """Lowest (floor) or highest (ceiling) horizontal plane holding >= frac of the strongest peak's points: a bed or a
    table top can hold more points than the floor, but it is never the lowest plane."""
    h, e = np.histogram(v, bins=max(int(round((hi - lo) / bw)), 1), range=(lo, hi))
    hs = np.convolve(h, np.ones(3) / 3, "same")
    pk = [i for i in range(1, len(hs) - 1) if hs[i] >= frac * hs.max() and hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1]]
    if not pk:
        return None
    c = e[min(pk) if which == "low" else max(pk)] + bw / 2
    sel = v[np.abs(v - c) < 0.02]
    return float(np.median(sel)) if sel.size else float(c)


def candidates(v, w, lo, hi, n=6, bw=0.01):
    """Strongest histogram peaks (range^2-weighted) in [lo, hi], >= 5 cm apart: (position, relative strength)."""
    h, e = np.histogram(v, bins=max(int(round((hi - lo) / bw)), 1), range=(lo, hi), weights=w)
    hs = np.convolve(h, np.ones(3) / 3, "same")
    out = []
    for i in np.argsort(-hs):
        if hs[i] <= 0 or len(out) >= n:
            break
        c = e[i] + bw / 2
        if all(abs(c - p) >= 0.05 for p, _ in out):
            out.append((float(c), float(hs[i] / hs.max())))
    return sorted(out)


def room_points(visit: str, laser_dir: str, sessions):
    """Laser points of the captured room.  A visit's stations can come from separate scanning sessions with their own
    frames (each has a station at the origin; heights differ): stations are grouped by frame, the session is registered to
    each group, the best-fitting group is used, and within it the stations standing on the captured room's floor near the
    camera path."""
    import glob
    files = sorted(glob.glob(os.path.join(laser_dir, "*_sample.npz")))
    st_all = np.array([np.loadtxt(f.replace("_sample.npz", "_pose.txt"), delimiter=",")[3, :3] for f in files])
    groups = []
    for i, s in enumerate(st_all):
        g = next((g for g in groups if abs(st_all[g[0], 2] - s[2]) < 0.05), None)
        g.append(i) if g is not None else groups.append([i])
    cloud = {gi: np.concatenate([np.load(files[i])["xyz"] for i in g]) for gi, g in enumerate(groups)}
    regs = {}
    for vid in sessions:
        cap = os.path.join(ROOT, "data", f"{vid}_lidar")
        if not os.path.exists(os.path.join(cap, "capture.json")):
            continue
        P, org = session_points(cap)
        rpath = os.path.join(GT_DIR, "registration", f"{vid}.json")
        if os.path.exists(rpath):  # reuse an earlier registration of this session (same laser group ids)
            reg = json.load(open(rpath))
            ids = [os.path.basename(files[i]).split("_")[0] for i in range(len(files))]
            gi = next((k for k, g in enumerate(groups) if [ids[i] for i in g] == reg.get("laser_group")), None)
            if gi is not None:
                regs[vid] = (reg, P, org, gi)
                print(vid, f"ICP inliers {100 * reg['icp_inlier_5cm']:.0f} % (cached registration)", flush=True)
                continue
        fits = []
        for gi, g in enumerate(groups):
            r = register(P, cloud[gi][::2], st_all[g], cams=org)
            r["laser_group"] = [os.path.basename(files[i]).split("_")[0] for i in g]
            fits.append((r["icp_inlier_5cm"], gi, r))
        inl, gi, reg = max(fits, key=lambda x: x[0])
        reg["capture"] = f"{vid}_lidar"
        regs[vid] = (reg, P, org, gi)
        os.makedirs(os.path.join(GT_DIR, "registration"), exist_ok=True)
        json.dump(reg, open(os.path.join(GT_DIR, "registration", f"{vid}.json"), "w"), indent=1)
        print(vid, f"ICP inliers {100 * inl:.0f} % (laser group {gi}: {reg['laser_group']})", flush=True)
    reg0, P0, org0, gi = max(regs.values(), key=lambda r: r[0]["icp_inlier_5cm"])  # best-registered session = reference
    Q, C = apply(reg0, P0), apply(reg0, org0)
    zf = reg0["floor_laser"]
    band = Q[(Q[:, 2] > zf + 1.0) & (Q[:, 2] < zf + 2.0), :2]
    lo, hi = np.percentile(band, 2, axis=0) - 0.4, np.percentile(band, 98, axis=0) + 0.4
    g = groups[gi]
    inside = [i for i in g if np.min(np.linalg.norm(C[:, :2] - st_all[i, :2], axis=1)) < 1.5]
    if not inside:
        inside = [min(g, key=lambda i: np.min(np.linalg.norm(C[:, :2] - st_all[i, :2], axis=1)))]
    pts = np.concatenate([np.load(files[i])["xyz"] for i in inside])
    rgb = np.concatenate([np.load(files[i])["rgb"] for i in inside])
    keep = (pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0]) & (pts[:, 1] > lo[1]) & (pts[:, 1] < hi[1]) & \
           (pts[:, 2] > zf - 0.3) & (pts[:, 2] < zf + 4.5)
    origin = st_all[inside[0]].copy()
    stations_used = [os.path.basename(files[i]).split("_")[0] for i in inside]
    room_points.rgb = rgb[keep]
    return pts[keep] - origin, origin, stations_used, {k: v[0] for k, v in regs.items()}


def fit(P, windows: str):
    z = P[:, 2]
    zf = extreme_plane(z, np.percentile(z, 0.2) - 0.1, np.percentile(z, 0.2) + 1.0, "low")
    zc = extreme_plane(z, zf + 2.0, np.percentile(z, 99.8) + 0.1, "high")
    band = P[(z > zf + 0.9) & (z < zc - 0.3)]
    best = None
    for th in np.deg2rad(np.arange(0, 90, 0.1)):
        c, s = np.cos(th), np.sin(th)
        x, y = band[:, 0] * c + band[:, 1] * s, -band[:, 0] * s + band[:, 1] * c
        sc = (np.histogram(x, bins=800)[0].astype(float) ** 2).sum() + (np.histogram(y, bins=800)[0].astype(float) ** 2).sum()
        if best is None or sc > best[0]:
            best = (sc, th)
    th = best[1]
    c, s = np.cos(th), np.sin(th)
    X, Y = band[:, 0] * c + band[:, 1] * s, -band[:, 0] * s + band[:, 1] * c
    rw = np.hypot(band[:, 0], band[:, 1]) ** 2  # laser density falls with range^2
    planes = {}
    for item in [w for w in windows.split(",") if w]:
        k, lo, hi = item.split(":")
        v = X if k in "WE" else Y
        planes[k], _ = peak(v, float(lo), float(hi), bw=0.005, w=rw)
    cand = {"x": candidates(X, rw, X.min(), X.max(), n=10), "y": candidates(Y, rw, Y.min(), Y.max(), n=10)}
    return dict(zf=zf, zc=zc, th=th, X=X, Y=Y, Z=band[:, 2], planes=planes, cand=cand)


def figure(path, f, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    k = max(1, len(f["X"]) // 200000)
    ax.scatter(f["X"][::k], f["Y"][::k], s=0.1, c="k")
    for p, r in f["cand"]["x"]:
        ax.axvline(p, color="tab:blue", lw=0.4 + 1.2 * r, alpha=0.6)
        ax.text(p, ax.get_ylim()[1] if False else f["Y"].max() + 0.05, f"{p:.2f}", rotation=90, fontsize=6, color="tab:blue")
    for p, r in f["cand"]["y"]:
        ax.axhline(p, color="tab:orange", lw=0.4 + 1.2 * r, alpha=0.6)
        ax.text(f["X"].max() + 0.05, p, f"{p:.2f}", fontsize=6, color="tab:orange")
    pl = f["planes"]
    if all(k in pl and pl[k] is not None for k in "WESN"):
        ax.plot([pl["W"], pl["E"], pl["E"], pl["W"], pl["W"]], [pl["S"], pl["S"], pl["N"], pl["N"], pl["S"]], "r-", lw=1.5)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)
    ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visit", required=True)
    ap.add_argument("--laser_root", default=os.path.join(ROOT, "..", "arkitscenes_raw", "laser"))
    a = ap.parse_args()
    split = json.load(open(os.path.join(ROOT, "data", "SPLIT.json")))
    sessions = {**split["dev"], **split["test"]}[a.visit]["sessions"]
    P, origin, used, regs = room_points(a.visit, os.path.join(a.laser_root, a.visit), sessions)
    wpath = os.path.join(GT_DIR, "windows.json")
    allw = json.load(open(wpath)) if os.path.exists(wpath) else {}
    windows = allw.get(a.visit, {}).get("windows", "")
    f = fit(P, windows)
    H = f["zc"] - f["zf"]
    pl = f["planes"]
    done = all(k in pl and pl[k] is not None for k in "WESN")
    title = (f"visit {a.visit}: laser stations {', '.join(used)}; floor-ceiling {H:.3f} m" +
             (f"; fitted {pl['E'] - pl['W']:.3f} x {pl['N'] - pl['S']:.3f} m" if done else "; candidate planes (blue x, orange y)"))
    figure(os.path.join(GT_DIR, f"{a.visit}_laser_fit.png"), f, title)
    print(title)
    print("x candidates:", [(round(p, 3), round(r, 2)) for p, r in f["cand"]["x"]])
    print("y candidates:", [(round(p, 3), round(r, 2)) for p, r in f["cand"]["y"]])
    if not done:
        print(f"-> add the main wall windows for {a.visit} to {wpath} and rerun")
        return
    w, d = pl["E"] - pl["W"], pl["N"] - pl["S"]
    gt = dict(world="arkitscenes", coverage="captured_room", visit=a.visit, laser_stations=used, wall_thickness=0.12, adjacency=[],
              rooms=[dict(name="room", x0=0.0, y0=0.0, w=float(w), d=float(d), ceiling_height=float(H), floor_area=float(w * d),
                          walls=dict(S=float(w), N=float(w), E=float(d), W=float(d)), openings=[],
                          damage=[], hazards=dict(low_light=False, glossy_floor=False, mirrors=0))],
              method="dataprep/room_gt.py: laser stations inside the captured room, sharpest-histogram yaw, floor/ceiling = "
                     "strongest horizontal planes, each main wall = strongest range^2-weighted plane in a manually chosen window "
                     f"({windows})",
              openings_reliable=False,
              note=allw[a.visit].get("note", ""))
    json.dump(gt, open(os.path.join(GT_DIR, f"{a.visit}.json"), "w"), indent=1)
    k = max(1, len(f["X"]) // 300000)
    # the room's laser points in the ground-truth frame (x from the W plane, y from the S plane, z from the floor),
    # 1 cm voxels, for the opening annotation (dataprep/openings_gt.py); kept next to the raw scans
    c, s_ = np.cos(f["th"]), np.sin(f["th"])
    G = np.stack([P[:, 0] * c + P[:, 1] * s_ - pl["W"], -P[:, 0] * s_ + P[:, 1] * c - pl["S"], P[:, 2] - f["zf"]], 1)
    _, first = np.unique(np.floor(G / 0.01).astype(np.int64), axis=0, return_index=True)
    np.savez_compressed(os.path.join(a.laser_root, a.visit, "room_gt_points.npz"), xyz=G[first].astype(np.float32),
                        rgb=room_points.rgb[first], w=w, d=d, H=H)
    if a.visit == "421337":  # wall-band slice for the fix-loop diagnosis (fixloop/diagnose.py) - only room used there
     np.savez_compressed(os.path.join(GT_DIR, f"{a.visit}_laser_slice.npz"),
                        xy=np.stack([f["X"][::k], f["Y"][::k]], 1).astype(np.float32),
                        planes=np.array([pl["W"], pl["E"], pl["S"], pl["N"], f["zf"], f["zc"]]),
                        names=np.array(["W", "E", "S", "N", "floor", "ceiling"]), yaw=f["th"], origin=origin)
    idx_path = os.path.join(GT_DIR, "index.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    for vid in sessions:
        idx[vid] = f"{a.visit}.json"
    json.dump(dict(sorted(idx.items())), open(idx_path, "w"), indent=1)
    print(f"wrote {a.visit}.json: {w:.3f} x {d:.3f} m, ceiling {H:.3f} m")


if __name__ == "__main__":
    main()
