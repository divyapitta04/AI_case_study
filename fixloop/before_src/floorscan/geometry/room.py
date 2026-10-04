"""Per-room model: rectilinear polygon, wall refinement, plane-anchored drift correction, floor/ceiling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..frames import Capture
from .primitives import Grid, mode_peak


@dataclass
class Wall:
    idx: int
    orient: str  # 'H': y = coord (runs along x)   'V': x = coord (runs along y)
    coord: float
    s0: float
    s1: float
    out_n: Tuple[float, float]
    sigma: float = 0.01  # 1-sigma uncertainty of `coord` (m)
    support: int = 0
    observed: bool = True
    top: float = 0.0  # 97th-percentile height of the wall's inlier points (absolute z)
    is_object: bool = False

    @property
    def length(self) -> float:
        return self.s1 - self.s0


@dataclass
class Opening:
    wall: int
    s0: float  # along-wall start (aligned-frame coordinate)
    s1: float
    z0: float  # height above floor
    z1: float
    kind: str
    sigma_w: float = 0.01
    sigma_h: float = 0.02
    confidence: float = 1.0
    evidence: Dict[str, float] = field(default_factory=dict)
    link: Optional[str] = None


@dataclass
class RoomModel:
    id: str
    walls: List[Wall]
    zf: float
    zc: float
    sigma_zf: float = 0.005
    sigma_zc: float = 0.005
    frames: List[int] = field(default_factory=list)
    openings: List[Opening] = field(default_factory=list)
    Rm: np.ndarray = field(default_factory=lambda: np.eye(2))  # world xy -> aligned xy
    yaw: float = 0.0
    name: str = ""
    notes: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def polygon(self) -> np.ndarray:
        return polygon_from_walls(self.walls)

    @property
    def height(self) -> float:
        return self.zc - self.zf

    def area(self) -> float:
        p = self.polygon
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_from_walls(walls: List[Wall]) -> np.ndarray:
    m = len(walls)
    V = []
    for i in range(m):
        a, b = walls[i], walls[(i + 1) % m]
        if a.orient == "H" and b.orient == "V":
            V.append((b.coord, a.coord))
        elif a.orient == "V" and b.orient == "H":
            V.append((a.coord, b.coord))
        else:  # should not happen for alternating runs
            V.append((a.coord, b.coord))
    return np.array(V)


# ---------------------------------------------------------------------------
def runs_from_mask(mask: np.ndarray, g: Grid, eps: float = 0.12, min_run: float = 0.30):
    """Manhattan-snapped rectilinear runs from a region mask.  Returns list of [orient, coord, length]
    ordered counter-clockwise."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, eps / g.res, True)[:, 0, :]
    V = g.lo + (approx + 0.5) * g.res
    x, y = V[:, 0], V[:, 1]
    if 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0:
        V = V[::-1]
    n = len(V)
    edges = []
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        d = b - a
        o = "H" if abs(d[0]) >= abs(d[1]) else "V"
        mid = 0.5 * (a + b)
        edges.append([o, mid[1] if o == "H" else mid[0], float(np.hypot(*d)), (d[0] if o == "H" else d[1])])
    runs = _group_runs(edges)
    changed = True
    while changed and len(runs) > 4:
        changed = False
        for i, r in enumerate(runs):
            if r[2] < min_run:
                del runs[i]
                runs = _group_runs([[q[0], q[1], q[2], q[3]] for q in runs])
                changed = True
                break
    return runs


def _group_runs(edges):
    # rotate so that the first edge starts a run
    n = len(edges)
    k = 0
    for i in range(n):
        if edges[i][0] != edges[i - 1][0]:
            k = i
            break
    edges = edges[k:] + edges[:k]
    runs = []
    for e in edges:
        if runs and runs[-1][0] == e[0]:
            r = runs[-1]
            tot = r[2] + e[2]
            r[1] = (r[1] * r[2] + e[1] * e[2]) / max(tot, 1e-9)
            r[2] = tot
            r[3] += e[3]
        else:
            runs.append([e[0], e[1], e[2], e[3]])
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        a, b = runs[-1], runs[0]
        tot = a[2] + b[2]
        b[1] = (a[1] * a[2] + b[1] * b[2]) / max(tot, 1e-9)
        b[2] = tot
        b[3] += a[3]
        runs.pop()
    return runs


def walls_from_runs(runs) -> List[Wall]:
    walls = [Wall(i, r[0], float(r[1]), 0.0, 0.0, (0.0, 0.0)) for i, r in enumerate(runs)]
    poly = polygon_from_walls(walls)
    for i, w in enumerate(walls):
        a, b = poly[i - 1], poly[i]  # wall i spans from vertex i-1 to vertex i
        d = b - a
        w.out_n = (float(np.sign(d[1])), float(-np.sign(d[0]))) if w.orient == "H" else (float(np.sign(d[1])), float(-np.sign(d[0])))
        # outward normal = right of direction for CCW polygons
        if w.orient == "H":
            w.out_n = (0.0, -float(np.sign(d[0])))
            w.s0, w.s1 = (min(a[0], b[0]), max(a[0], b[0]))
        else:
            w.out_n = (float(np.sign(d[1])), 0.0)
            w.s0, w.s1 = (min(a[1], b[1]), max(a[1], b[1]))
    return walls


def drop_objects(walls: List[Wall]) -> List[Wall]:
    """Walls that do not reach the ceiling are furniture (wardrobe, bookshelf), not room boundary: remove them
    and let the neighbouring walls close the corner."""
    ws = [w for w in walls]
    if sum(1 for w in ws if w.is_object) == 0:
        return walls
    keep = [w for w in ws if not w.is_object]
    if len(keep) < 4:
        return walls
    # merge consecutive walls with the same orientation (choose the better supported coordinate)
    merged: List[Wall] = []
    for w in keep:
        if merged and merged[-1].orient == w.orient:
            a = merged[-1]
            best = a if a.support >= w.support else w
            a.coord, a.sigma, a.support = best.coord, best.sigma, a.support + w.support
        else:
            merged.append(w)
    if len(merged) > 1 and merged[0].orient == merged[-1].orient:
        a, w = merged[0], merged.pop()
        best = a if a.support >= w.support else w
        a.coord, a.sigma, a.support = best.coord, best.sigma, a.support + w.support
    if len(merged) < 4:
        return walls
    runs = [[w.orient, w.coord] for w in merged]
    nw = walls_from_runs(runs)
    for new, old in zip(nw, merged):
        new.sigma, new.support, new.observed, new.top = old.sigma, old.support, old.observed, old.top
    return nw


def simplify_walls(walls: List[Wall], tol: float = 0.06, max_spur: float = 1.0) -> List[Wall]:
    """Remove collinear spurs/notches: wall triples (a, b, c) with a || c on the same plane and a short
    connecting wall b are merged into one wall (e.g. a door recess or furniture corner in the free-space mask)."""
    ws = list(walls)
    changed = True
    while changed and len(ws) > 4:
        changed = False
        m = len(ws)
        # unobserved short wall (mask artefact): drop it and merge its neighbours
        for i in range(m):
            b = ws[(i + 1) % m]
            if (not b.observed) and b.length < max_spur and m > 4:
                a, c = ws[i], ws[(i + 2) % m]
                if a.orient == c.orient and a is not c:
                    keep = a if a.support >= c.support else c
                    a.coord, a.support, a.sigma = keep.coord, a.support + c.support, keep.sigma
                    a.observed = True
                    for idx in sorted([(i + 1) % m, (i + 2) % m], reverse=True):
                        del ws[idx]
                    changed = True
                    break
        if changed:
            continue
        for i in range(m):
            a, b, c = ws[i], ws[(i + 1) % m], ws[(i + 2) % m]
            if a.orient == c.orient and abs(a.coord - c.coord) < tol and b.length < max_spur and a is not c:
                wa, wc = max(a.support, 1), max(c.support, 1)
                a.coord = (a.coord * wa + c.coord * wc) / (wa + wc)
                a.support = wa + wc
                a.sigma = min(a.sigma, c.sigma)
                a.observed = a.observed or c.observed
                j = (i + 1) % m
                k = (i + 2) % m
                for idx in sorted([j, k], reverse=True):
                    del ws[idx]
                changed = True
                break
    runs = [[w.orient, w.coord] for w in ws]
    nw = walls_from_runs(runs)
    for new, old in zip(nw, ws):
        new.sigma, new.support, new.observed = old.sigma, old.support, old.observed
    return nw


# ---------------------------------------------------------------------------
def _band_points(cap: Capture, fidx, Rm: np.ndarray, zf: float, zc: float, band=(1.0, 0.25), stride: int = 1):
    d = cap.world(fidx, stride=stride)
    z = d["pts"][:, 2]
    m = (z > zf + band[0]) & (z < zc - band[1])
    xy = d["pts"][m, :2] @ Rm.T
    return xy, d["frame"][m], m, d


def pick_wall_peak(u: np.ndarray, cu: float, win: float, min_frac: float = 0.12):
    """Dominant plane within the window; a plane further out (>= 0.18 m behind it) wins only when it has
    at least half the dominant plane's support (wall behind furniture).  Adjacent rooms' faces 12 cm behind
    a partition therefore never beat the wall's own face."""
    best = mode_peak(u, cu - win, cu + win, bin_w=0.01, min_frac=min_frac, which="max", min_count=8)
    if best is None:
        return None
    outer = mode_peak(u, best[0] + 0.18, cu + win, bin_w=0.01, min_frac=min_frac, which="high", min_count=8)
    if outer is not None and outer[1] >= 0.5 * best[1]:
        return outer
    return best


def refine_walls(walls: List[Wall], xy: np.ndarray, frames: np.ndarray, win: float = 0.45,
                 min_frac: float = 0.12, chunk: int = 6, z: Optional[np.ndarray] = None,
                 zc: Optional[float] = None) -> List[Wall]:
    """Snap each wall to the outermost well-supported plane within `win` of its current position."""
    out = []
    for w in walls:
        n = np.array(w.out_n)
        sgn = n[0] + n[1]
        ax = 0 if w.orient == "V" else 1
        oth = 1 - ax
        m = (xy[:, oth] > w.s0 + 0.12) & (xy[:, oth] < w.s1 - 0.12)
        u = sgn * xy[m, ax]
        cu = sgn * w.coord
        res = pick_wall_peak(u, cu, win, min_frac)
        nw = Wall(w.idx, w.orient, w.coord, w.s0, w.s1, w.out_n, w.sigma, w.support, w.observed)
        if res is not None:
            nw.coord = float(sgn * res[0])
            nw.support = int(res[1])
            nw.observed = nw.support >= 600
            inl = np.abs(u - res[0]) < 0.03
            if z is not None and zc is not None and inl.sum() > 30:
                nw.top = float(np.percentile(z[m][inl], 97))
                nw.is_object = nw.top < zc - 0.5
            fr = frames[m][inl]
            uu = u[inl]
            cid = fr // chunk
            meds = [np.median(uu[cid == c]) for c in np.unique(cid) if (cid == c).sum() >= 15]
            s_chunk = float(np.std(meds)) / np.sqrt(len(meds)) if len(meds) >= 3 else 0.01
            nw.sigma = float(np.sqrt(0.002 ** 2 + s_chunk ** 2 + (np.std(uu) / np.sqrt(max(inl.sum() / 40, 1))) ** 2))
        else:
            nw.observed = False
        out.append(nw)
    return out


def assign_to_walls(walls: List[Wall], xy: np.ndarray, tol: float = 0.10):
    """Index of the wall each point belongs to (-1 none) and signed distance to that wall plane."""
    best = np.full(len(xy), -1)
    bd = np.full(len(xy), tol)
    sd = np.zeros(len(xy))
    for k, w in enumerate(walls):
        n = np.array(w.out_n)
        ax = 0 if w.orient == "V" else 1
        oth = 1 - ax
        sg = n[0] + n[1]
        r = sg * xy[:, ax] - sg * w.coord
        ok = (np.abs(r) < bd) & (xy[:, oth] > w.s0 - 0.08) & (xy[:, oth] < w.s1 + 0.08)
        best[ok] = k
        bd[ok] = np.abs(r[ok])
        sd[ok] = r[ok]
    return best, sd


def planar_correct(cap: Capture, fidx: Sequence[int], walls: List[Wall], Rm: np.ndarray, zf: float, zc: float,
                   chunk: int = 6, sigma_t: float = 0.10, sigma_th: float = np.deg2rad(2.0),
                   sigma_s: float = 0.012, sigma_sth: float = np.deg2rad(0.25)) -> float:
    """Plane-anchored drift correction.

    Frames are grouped into time chunks.  For every chunk we solve for a 3-DOF correction (dx, dy, dyaw) so that
    its wall points sit on the fitted wall planes, and a dz so floor / ceiling points sit on their planes.  The
    chunk corrections are solved *jointly* with a smoothness term between consecutive chunks (visual-inertial drift
    changes slowly), so a chunk that only sees one wall still inherits the other direction from its neighbours.
    Applied to the capture's poses in place; returns the wall-point rms before the correction."""
    fidx = list(fidx)
    d = cap.world(fidx)
    z = d["pts"][:, 2]
    band = (z > zf + 0.5) & (z < zc - 0.3)
    xy = d["pts"][band, :2] @ Rm.T
    fr = d["frame"][band]
    which, sdist = assign_to_walls(walls, xy)
    rms0 = float(np.sqrt(np.mean(sdist[which >= 0] ** 2))) if (which >= 0).any() else 0.0
    zfl = d["pts"][:, 2]
    fl = np.abs(zfl - zf) < 0.06
    ce = np.abs(zfl - zc) < 0.06
    Rinv = Rm.T  # aligned -> world xy
    cids = sorted(set(f // chunk for f in fidx))
    C = len(cids)
    Hm = np.zeros((3 * C, 3 * C))
    gv = np.zeros(3 * C)
    cen_w = {}
    dzs = {}
    prior = np.diag([1 / sigma_t ** 2, 1 / sigma_t ** 2, 1 / sigma_th ** 2])
    for k, c in enumerate(cids):
        fs = [f for f in fidx if f // chunk == c]
        sel = np.isin(fr, fs) & (which >= 0)
        dz_parts, dz_w = [], []
        mf = np.isin(d["frame"], fs)
        if (mf & fl).sum() > 30:
            dz_parts.append(zf - np.mean(zfl[mf & fl]))
            dz_w.append((mf & fl).sum())
        if (mf & ce).sum() > 30:
            dz_parts.append(zc - np.mean(zfl[mf & ce]))
            dz_w.append((mf & ce).sum())
        dzs[c] = float(np.average(dz_parts, weights=dz_w)) if dz_parts else 0.0
        Hm[3 * k:3 * k + 3, 3 * k:3 * k + 3] += prior
        if sel.sum() < 40:
            cen_w[c] = np.zeros(3)
            continue
        p = xy[sel]
        cen = p.mean(0)
        q = p - cen
        nrm = np.array([walls[j].out_n for j in which[sel]])
        r = sdist[sel]
        J = np.stack([nrm[:, 0], nrm[:, 1], -nrm[:, 0] * q[:, 1] + nrm[:, 1] * q[:, 0]], 1)
        Hm[3 * k:3 * k + 3, 3 * k:3 * k + 3] += J.T @ J
        gv[3 * k:3 * k + 3] += -J.T @ r
        cen_w[c] = np.array([*(Rinv @ cen), 0.0])
    W = np.diag([1 / sigma_s ** 2, 1 / sigma_s ** 2, 1 / sigma_sth ** 2])
    for k in range(C - 1):
        if cids[k + 1] - cids[k] == 1:  # consecutive in time
            i, j = 3 * k, 3 * (k + 1)
            Hm[i:i + 3, i:i + 3] += W
            Hm[j:j + 3, j:j + 3] += W
            Hm[i:i + 3, j:j + 3] -= W
            Hm[j:j + 3, i:i + 3] -= W
    sol = np.linalg.solve(Hm, gv)
    for k, c in enumerate(cids):
        fs = [f for f in fidx if f // chunk == c]
        dx = sol[3 * k:3 * k + 2]
        th = float(sol[3 * k + 2])
        tw = Rinv @ dx
        cap.apply_correction(fs, th, np.array([tw[0], tw[1], dzs[c]]), cen_w[c])
    return rms0


def levels(cap: Capture, fidx, poly: np.ndarray, Rm: np.ndarray, zc_hint: Optional[float] = None):
    """Floor and ceiling heights from the z-histogram of points inside the room polygon."""
    d = cap.world(fidx)
    xy = d["pts"][:, :2] @ Rm.T
    inside = _in_poly(xy, poly, shrink=0.25)
    z = d["pts"][inside, 2]
    cz = float(np.median(cap.org[list(fidx), 2]))
    fl = mode_peak(z, cz - 2.2, cz - 0.6, bin_w=0.01, min_frac=0.3, which="low", min_count=20)
    ce = mode_peak(z, cz + 0.4, cz + 2.0, bin_w=0.01, min_frac=0.3, which="high", min_count=20)
    return fl, ce, d, xy, inside


def _in_poly(xy: np.ndarray, poly: np.ndarray, shrink: float = 0.0) -> np.ndarray:
    # fast raster test using matplotlib Path would be heavier; cv2.pointPolygonTest per-point is slow for
    # millions of points, so rasterise the polygon instead.
    lo = poly.min(0) - 0.5
    hi = poly.max(0) + 0.5
    res = 0.02
    w, h = np.ceil((hi - lo) / res).astype(int)
    img = np.zeros((h, w), np.uint8)
    cv2.fillPoly(img, [((poly - lo) / res).astype(np.int32)], 1)
    if shrink > 0:
        k = int(shrink / res)
        img = cv2.erode(img, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    ij = np.floor((xy - lo) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    out = np.zeros(len(xy), bool)
    out[ok] = img[ij[ok, 1], ij[ok, 0]] > 0
    return out
