"""Manhattan yaw, floor/ceiling levels and room segmentation (occupancy carving + persistence watershed)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..frames import Capture


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------------------
def mode_peak(vals: np.ndarray, lo: float, hi: float, bin_w: float = 0.01, min_frac: float = 0.0,
              which: str = "max", smooth: int = 2, refine_w: float = 0.02,
              min_count: int = 3) -> Optional[Tuple[float, int]]:
    """Robust 1-D mode. `which`: 'max' (highest peak), 'low' / 'high' (lowest / highest peak that has at
    least ``min_frac`` of the max peak count).  Returns (value, supporting count)."""
    if vals.size == 0:
        return None
    m = (vals >= lo) & (vals <= hi)
    v = vals[m]
    if v.size < 5:
        return None
    nb = int(np.ceil((hi - lo) / bin_w))
    h, edges = np.histogram(v, bins=nb, range=(lo, lo + nb * bin_w))
    hs = ndi.uniform_filter1d(h.astype(float), 2 * smooth + 1, mode="constant")
    mx = hs.max()
    if mx <= 0:
        return None
    pk = [i for i in range(len(hs)) if hs[i] >= min_frac * mx and hs[i] >= min_count
          and (i == 0 or hs[i] >= hs[i - 1]) and (i == len(hs) - 1 or hs[i] > hs[i + 1])]
    if not pk:
        return None
    if which == "max":
        i = max(pk, key=lambda k: hs[k])
    elif which == "low":
        i = min(pk)
    else:
        i = max(pk)
    c = lo + (i + 0.5) * bin_w
    sel = v[np.abs(v - c) < refine_w]
    if sel.size == 0:
        return c, int(h[i])
    return float(np.median(sel)), int(sel.size)


def estimate_yaw(xy: np.ndarray, res: float = 0.04) -> float:
    """Dominant Manhattan direction (radians, mod 90deg) from wall-band points."""
    lo = xy.min(0) - 0.2
    hi = xy.max(0) + 0.2
    w, h = np.ceil((hi - lo) / res).astype(int)
    H = np.zeros((h, w), np.float32)
    ij = ((xy - lo) / res).astype(int)
    np.add.at(H, (ij[:, 1], ij[:, 0]), 1)
    H = cv2.GaussianBlur(np.log1p(H), (0, 0), 1.5)
    gx = cv2.Sobel(H, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(H, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    th = np.arctan2(gy, gx)
    z = (mag * np.exp(4j * th)).sum()
    return float(np.angle(z) / 4.0)


# ---------------------------------------------------------------------------
class Grid:
    def __init__(self, lo: np.ndarray, hi: np.ndarray, res: float):
        self.res = res
        self.lo = lo - 0.5
        self.shape = tuple(np.ceil((hi + 0.5 - self.lo) / res).astype(int)[::-1])  # (h, w)

    def ij(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((xy - self.lo) / self.res).astype(int)


def carve(cap: Capture, fidx, yaw: float, zf: float, zc: float, res: float = 0.05, stride: int = 3,
          band: Tuple[float, float] = (1.0, 0.25)):
    """2-D occupancy (hits) and free-space (ray carving) in the Manhattan-aligned frame, restricted to the
    wall height band so floor/ceiling/furniture do not pollute the slice."""
    Rm = rot2(-yaw)
    d = cap.world(fidx, stride=stride)
    pts = d["pts"]
    org = d["org"]
    z_lo, z_hi = zf + band[0], zc - band[1]
    xy_all = np.concatenate([pts[:, :2], org[:, :2]]) @ Rm.T
    g = Grid(xy_all.min(0), xy_all.max(0), res)
    H = np.zeros(g.shape, np.float32)
    F = np.zeros(g.shape, np.float32)
    inb = (pts[:, 2] > z_lo) & (pts[:, 2] < z_hi)
    ij = g.ij(pts[inb, :2] @ Rm.T)
    np.add.at(H, (ij[:, 1], ij[:, 0]), 1)
    step = res
    rngs = d["rng"]
    n_max = int(np.ceil(rngs.max() / step)) if rngs.size else 0
    o2 = org[:, :2] @ Rm.T
    dirs = d["dirs"]
    d2 = dirs[:, :2] @ Rm.T
    chunk = 20000
    for a in range(0, len(rngs), chunk):
        sl = slice(a, a + chunk)
        r = rngs[sl]
        ts = np.arange(1, n_max + 1)[None, :] * step
        ok = ts < (r[:, None] - step)
        z = org[sl, 2][:, None] + ts * dirs[sl, 2][:, None]
        ok &= (z > z_lo) & (z < z_hi)
        x = o2[sl, 0][:, None] + ts * d2[sl, 0][:, None]
        y = o2[sl, 1][:, None] + ts * d2[sl, 1][:, None]
        ii = np.floor((x - g.lo[0]) / res).astype(int)
        jj = np.floor((y - g.lo[1]) / res).astype(int)
        ok &= (ii >= 0) & (ii < g.shape[1]) & (jj >= 0) & (jj < g.shape[0])
        lin = (jj * g.shape[1] + ii)[ok]
        F += np.bincount(lin, minlength=g.shape[0] * g.shape[1]).reshape(g.shape).astype(np.float32)
    return g, H, F, Rm


def segment_rooms(g: Grid, H: np.ndarray, F: np.ndarray, frame_xy: np.ndarray, persistence: float = 0.12, max_door: float = 1.3,
                  min_area: float = 1.5):
    """Free space -> rooms by persistence watershed on the distance transform. Returns label image (0 bg)."""
    res = g.res
    free = (F >= 2) & (H < 3)
    # fill tiny non-free specks (noise) without erasing thin walls
    nf, nn = ndi.label(~free)
    sizes = np.bincount(nf.ravel())
    small = np.isin(nf, [i for i in range(1, nn + 1) if sizes[i] < 12])
    free = free | small
    lab, n = ndi.label(free)
    ij = g.ij(frame_xy)
    ij[:, 0] = np.clip(ij[:, 0], 0, g.shape[1] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, g.shape[0] - 1)
    keep = np.unique(lab[ij[:, 1], ij[:, 0]])
    keep = keep[keep > 0]
    free = np.isin(lab, keep)
    free = ndi.binary_fill_holes(free)
    D = ndi.distance_transform_edt(free) * res
    labels = _watershed_persistence(D, free, persistence)
    idx = ndi.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
    filled = labels[idx[0], idx[1]]
    filled[~free] = 0
    out = filled.copy()
    for l in np.unique(out):
        if l == 0:
            continue
        m = out == l
        if m.sum() * res * res < min_area:
            dil = ndi.binary_dilation(m, iterations=2) & ~m & (out > 0)
            if dil.any():
                nb = np.bincount(out[dil]).argmax()
                out[m] = nb
            else:
                out[m] = 0
    return merge_wide_contacts(out, res, max_door), free, D


def merge_wide_contacts(lab: np.ndarray, res: float, max_door: float = 1.3) -> np.ndarray:
    """Two regions that touch along more free space than any doorway is wide are one room: furniture in the middle of
    a room (a table, a bed) makes a false saddle in the distance transform and splits the room (real ARKitScenes rooms),
    whereas rooms are connected through door-sized gaps.  Open-plan spaces joined by wider openings become one room."""
    out = lab.copy()
    while True:
        a, b = out[:, :-1], out[:, 1:]
        c, d = out[:-1, :], out[1:, :]
        pairs = np.concatenate([np.stack([a[(a != b) & (a > 0) & (b > 0)], b[(a != b) & (a > 0) & (b > 0)]], 1),
                                np.stack([c[(c != d) & (c > 0) & (d > 0)], d[(c != d) & (c > 0) & (d > 0)]], 1)])
        if not len(pairs):
            return out
        pairs = np.sort(pairs, 1)
        uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
        # boundary length ~ number of 4-neighbour contacts / 2 (each boundary cell is counted from both sides at most)
        best = int(np.argmax(cnt))
        if cnt[best] * res < max_door:
            return out
        l1, l2 = uniq[best]
        out[out == l2] = l1


def _watershed_persistence(D: np.ndarray, free: np.ndarray, h: float) -> np.ndarray:
    hh, ww = D.shape
    order = np.argsort(-D[free], kind="stable")
    ys, xs = np.nonzero(free)
    ys, xs = ys[order], xs[order]
    parent: Dict[int, int] = {}
    peak: Dict[int, float] = {}
    lab = np.zeros(D.shape, np.int32)
    nxt = 1

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nb8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for y, x in zip(ys, xs):
        d = D[y, x]
        roots = set()
        for dy, dx in nb8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < hh and 0 <= xx < ww and lab[yy, xx] > 0:
                roots.add(find(lab[yy, xx]))
        if not roots:
            lab[y, x] = nxt
            parent[nxt] = nxt
            peak[nxt] = d
            nxt += 1
            continue
        roots = sorted(roots, key=lambda r: (-peak[r], r))
        top = roots[0]
        merged_all = True
        for r in roots[1:]:
            if peak[r] - d < h:
                parent[r] = top
            else:
                merged_all = False
        if merged_all or len(roots) == 1:
            lab[y, x] = top
        else:
            lab[y, x] = 0
    out = np.zeros_like(lab)
    m = lab > 0
    roots = np.array([find(int(l)) for l in lab[m]])
    uniq = {r: i + 1 for i, r in enumerate(np.unique(roots))}
    out[m] = [uniq[r] for r in roots]
    return out
