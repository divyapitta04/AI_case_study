"""Register a LiDAR session (data/<video>_lidar) to its venue's FARO laser scans.

    python dataprep/register.py --capture data/42444946_lidar --laser ../arkitscenes_raw/laser/421337 --out data/gt/reg_42444946.json

Uses ONLY the session's raw depth + ARKit poses (not the floorscan pipeline): wall-band points are back-projected, both clouds
are rotated to their dominant Manhattan direction, the four 90-degree candidates are scored by FFT cross-correlation of 5 cm
occupancy images, the best is refined by 2-D ICP, and the floor levels give the vertical offset.  The transform
(session world -> laser venue frame) is used to crop the laser scan to the captured room (dataprep/room_gt.py) and for
diagnosis figures; it never enters the pipeline.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_laser(laser_dir: str, step: int = 1):
    """All station samples of a venue -> (xyz venue frame, station positions)."""
    pts, st = [], []
    for f in sorted(glob.glob(os.path.join(laser_dir, "*_sample.npz"))):
        pts.append(np.load(f)["xyz"][::step])
        pose = f.replace("_sample.npz", "_pose.txt")
        st.append(np.loadtxt(pose, delimiter=",")[3, :3])
    return np.concatenate(pts), np.array(st)


def peak(v, lo, hi, bw=0.01, w=None):
    m = (v > lo) & (v < hi)
    if m.sum() < 50:
        return None
    h, e = np.histogram(v[m], bins=max(int((hi - lo) / bw), 1), range=(lo, hi), weights=None if w is None else w[m])
    hs = np.convolve(h, np.ones(3) / 3, "same")
    c = e[int(np.argmax(hs))] + bw / 2
    sel = v[m][np.abs(v[m] - c) < 0.02]
    return float(np.median(sel)) if sel.size else float(c)


def manhattan_yaws(xy: np.ndarray, n: int = 3):
    """Dominant Manhattan directions (mod 90 deg): the n best local maxima of the histogram-sharpness score."""
    ths = np.deg2rad(np.arange(0, 90, 0.25))
    sc = []
    for th in ths:
        c, s = np.cos(th), np.sin(th)
        x, y = xy[:, 0] * c + xy[:, 1] * s, -xy[:, 0] * s + xy[:, 1] * c
        sc.append((np.histogram(x, bins=600)[0].astype(float) ** 2).sum() + (np.histogram(y, bins=600)[0].astype(float) ** 2).sum())
    sc = np.array(sc)
    m = len(sc)
    loc = [i for i in range(m) if sc[i] >= sc[(i - 1) % m] and sc[i] >= sc[(i + 1) % m]]
    loc.sort(key=lambda i: -sc[i])
    out = []
    for i in loc:
        if all(min(abs(i - j), m - abs(i - j)) >= 8 for j in out):  # >= 2 deg apart
            out.append(i)
        if len(out) == n:
            break
    return [float(ths[i]) for i in out]


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def occupancy(xy, lo, shape, res):
    ij = np.floor((xy - lo) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0])
    img = np.zeros(shape, np.float32)
    np.add.at(img, (ij[ok, 1], ij[ok, 0]), 1.0)
    return (img >= 2).astype(np.float32)


def session_points(capture: str, stride: int = 4):
    from floorscan.ingest.lidar import load_native
    cap = load_native(capture)
    d = cap.world(range(cap.n_frames), stride=stride)
    return d["pts"], cap.org


def register(P_ours: np.ndarray, P_las: np.ndarray, stations: np.ndarray, res: float = 0.05, cams: np.ndarray = None):
    from scipy.signal import fftconvolve
    from scipy.spatial import cKDTree
    # floors: ours = lowest strong z peak below the cameras; laser = strongest z peak 0.5-2.2 m below the stations
    zf_o = peak(P_ours[:, 2], np.percentile(P_ours[:, 2], 0.5) - 0.05, np.percentile(P_ours[:, 2], 30))
    zst = float(np.median(stations[:, 2]))
    zf_l = peak(P_las[:, 2], zst - 2.2, zst - 0.5)
    band_o = P_ours[(P_ours[:, 2] > zf_o + 1.0) & (P_ours[:, 2] < zf_o + 2.0), :2]
    if cams is not None:  # drop what was seen through windows / doors: keep points within 4 m of the camera path
        from scipy.spatial import cKDTree as _T
        dc, _ = _T(cams[:, :2]).query(band_o, distance_upper_bound=4.0, workers=-1)
        band_o = band_o[np.isfinite(dc)]
    band_l = P_las[(P_las[:, 2] > zf_l + 1.0) & (P_las[:, 2] < zf_l + 2.0), :2]
    band_l = band_l[np.linalg.norm(band_l - stations[:, :2].mean(0), axis=1) < 15]
    rng = np.random.default_rng(0)
    so = band_o[rng.choice(len(band_o), min(len(band_o), 60000), replace=False)]
    sl = band_l[rng.choice(len(band_l), min(len(band_l), 200000), replace=False)]
    yl = manhattan_yaws(sl, 1)[0]
    Al = sl @ rot(-yl).T
    pad = float(np.ptp(so, axis=0).max()) + 1.0  # the laser image must be larger than ours in every direction
    lo_l = Al.min(0) - pad
    shape_l = tuple((np.ceil((Al.max(0) + pad - lo_l) / res).astype(int))[::-1])
    L = occupancy(Al, lo_l, shape_l, res)
    tree = cKDTree(sl)
    so_icp = so[:: max(1, len(so) // 20000)]

    def icp(R, T):
        for it in range(30):
            Q = so_icp @ R.T + T
            d, j = tree.query(Q, distance_upper_bound=0.3, workers=-1)
            m = d < (0.25 if it < 10 else 0.06)
            if m.sum() < 200:
                break
            A, B = Q[m], sl[j[m]]
            ca, cb = A.mean(0), B.mean(0)
            U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            dR = (U @ Vt).T
            if np.linalg.det(dR) < 0:
                Vt[-1] *= -1
                dR = (U @ Vt).T
            R, T = dR @ R, dR @ (T - ca) + cb
        d, _ = tree.query(so @ R.T + T, distance_upper_bound=0.1, workers=-1)
        return R, T, float(np.mean(d < 0.05))

    best = None
    for yo, k in [(y, k) for y in manhattan_yaws(so, 3) for k in range(4)]:
        # 3 yaw hypotheses x 4 x 90 deg x top-3 correlation peaks; every candidate refined by ICP, best inlier ratio wins
        Ak = so @ rot(-yo).T @ rot(k * np.pi / 2).T
        lo_o = Ak.min(0)
        shape_o = tuple((np.ceil((Ak.max(0) - lo_o) / res).astype(int) + 1)[::-1])
        O = occupancy(Ak, lo_o, shape_o, res)
        C = fftconvolve(L, O[::-1, ::-1], mode="valid")
        if C.size == 0:
            continue
        Cw = C.copy()
        for _ in range(3):
            i, j = np.unravel_index(int(np.argmax(Cw)), Cw.shape)
            score = float(Cw[i, j] / max(O.sum(), 1))
            Cw[max(i - 10, 0):i + 11, max(j - 10, 0):j + 11] = -1  # suppress this peak (+-0.5 m)
            t = lo_l + np.array([j, i]) * res - lo_o  # Ak + t lands on the laser (Manhattan-aligned frames)
            R, T, inl = icp(rot(yl) @ rot(k * np.pi / 2) @ rot(-yo), rot(yl) @ t)
            if best is None or inl > best[0]:
                best = (inl, R, T, score)
    inl, R, T, score = best
    return dict(R2=R.tolist(), t2=T.tolist(), dz=float(zf_l - zf_o), yaw_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                corr_score=score, icp_inlier_5cm=inl, floor_ours=zf_o, floor_laser=zf_l)


def apply(reg: dict, P: np.ndarray) -> np.ndarray:
    R, t = np.array(reg["R2"]), np.array(reg["t2"])
    out = P.copy()
    out[:, :2] = P[:, :2] @ R.T + t
    out[:, 2] = P[:, 2] + reg["dz"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--laser", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure")
    a = ap.parse_args()
    P_ours, cams = session_points(a.capture)
    P_las, st = load_laser(a.laser, step=2)
    reg = register(P_ours, P_las, st, cams=cams)
    reg["capture"] = os.path.basename(os.path.normpath(a.capture))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(reg, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in reg.items() if k not in ("R2", "t2")}))
    if a.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Q = apply(reg, P_ours)
        zf = reg["floor_laser"]
        bl = P_las[(P_las[:, 2] > zf + 1.0) & (P_las[:, 2] < zf + 2.0)]
        bo = Q[(Q[:, 2] > zf + 1.0) & (Q[:, 2] < zf + 2.0)]
        c = bo[:, :2].mean(0)
        bl = bl[np.linalg.norm(bl[:, :2] - c, axis=1) < 8]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(bl[::10, 0], bl[::10, 1], s=0.1, c="0.6", label="laser (all stations)")
        ax.scatter(bo[::10, 0], bo[::10, 1], s=0.1, c="r", label="session LiDAR (registered)")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", markerscale=20)
        ax.set_title(f"{reg['capture']}: ICP inliers {100 * reg['icp_inlier_5cm']:.0f} % within 5 cm")
        fig.savefig(a.figure, dpi=90)


if __name__ == "__main__":
    main()
