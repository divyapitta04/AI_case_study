"""Photo tier: per-room folders of stills (2-8 per room, no depth, no poses) -> the same plan contract.

Per room: metric monocular depth -> gravity from the depth-normal Manhattan structure -> yaw registration of the
stills by pure-rotation homographies -> rectangular room from dominant wall planes -> openings from depth
discontinuities (doors) and glass-like colour (windows) -> scale from a printed ArUco card or the standard door
height.  Rooms are then placed by door constraints (`*_to_<room>.jpg` doorway shots name the neighbour).
Anything not observable (a wall in no photo, no ceiling in any photo) is reported with a wide interval and flagged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import Config
from ..frames import KIND_HIT, Capture, Frame, RaySet
from ..geometry import openings as op_mod
from ..geometry import stitch as st
from ..geometry.primitives import mode_peak, rot2
from ..geometry.room import Opening, RoomModel, walls_from_runs
from ..ingest import photos as ph_mod
from .. import damage as damage_mod
from .. import reasoning
from .assemble import assemble

DOOR_H_PRIOR = 2.04  # m, typical interior door (EU 2.04-2.10, US 2.03)
DOOR_H_SIGMA = 0.035  # relative
DEPTH_MODEL_SIGMA = 0.055  # relative; assumed accuracy of a metric monocular model on indoor stills
PRIOR_CEILING = 2.5
PRIOR_ROOM = (3.8, 3.4)


@dataclass
class PData:
    ph: ph_mod.Photo
    img: np.ndarray
    depth: np.ndarray  # (h, w) z-depth (metres), at the model's resolution
    Kd: np.ndarray
    P: np.ndarray  # (h, w, 3) camera-frame points
    nrm: np.ndarray  # (h, w, 3) camera-frame normals (towards camera)
    ok: np.ndarray  # (h, w) bool: reliable normal
    up: np.ndarray = None
    M: np.ndarray = None  # camera -> gravity frame (x_w, y_w, z_w rows)
    floor_z: Optional[float] = None  # height of floor relative to camera (negative)
    ceil_z: Optional[float] = None
    yaw: Optional[float] = None
    scale: float = 1.0
    marker: Optional[dict] = None
    feats: tuple = None
    prof: np.ndarray = None
    psi: Optional[float] = None


def _normals(P: np.ndarray):
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=2, keepdims=True)
    n = n / np.maximum(nn, 1e-9)
    # towards the camera
    flip = (np.sum(n * P[1:-1, 1:-1], axis=2) > 0)[..., None]
    n = np.where(flip, -n, n)
    jump = (np.abs(P[1:-1, 2:, 2] - P[1:-1, :-2, 2]) > 0.25) | (np.abs(P[2:, 1:-1, 2] - P[:-2, 1:-1, 2]) > 0.25)
    N = np.zeros_like(P)
    N[1:-1, 1:-1] = n
    ok = np.zeros(P.shape[:2], bool)
    ok[1:-1, 1:-1] = ~jump & (nn[..., 0] > 1e-9)
    return N, ok


def _estimate_up(N: np.ndarray, ok: np.ndarray) -> np.ndarray:
    n = N[ok]
    if len(n) > 5000:
        n = n[np.random.default_rng(0).choice(len(n), 5000, replace=False)]
    cands = []
    for pol in np.deg2rad(np.arange(0, 56, 2.0)):
        for az in np.deg2rad(np.arange(0, 360, 15.0)):
            if pol == 0 and az > 0:
                continue
            cands.append((pol, az))

    def vec(pol, az):
        # polar angle from camera "up" (0,-1,0); azimuth around it
        up0 = np.array([0, -1.0, 0])
        a = np.array([1.0, 0, 0])
        b = np.array([0, 0, 1.0])
        return np.cos(pol) * up0 + np.sin(pol) * (np.cos(az) * a + np.sin(az) * b)

    def score(u):
        s = (n @ u) ** 2
        return float(np.sum(s ** 2 + (1 - s) ** 2))

    best = max(cands, key=lambda c: score(vec(*c)))
    pol, az = best
    bs = score(vec(pol, az))
    for step in (np.deg2rad(1.0), np.deg2rad(0.3)):
        improved = True
        while improved:
            improved = False
            for dp, da in [(step, 0), (-step, 0), (0, step * 3), (0, -step * 3)]:
                p2, a2 = max(pol + dp, 0.0), az + da
                sc = score(vec(p2, a2))
                if sc > bs:
                    pol, az, bs, improved = p2, a2, sc, True
    return vec(pol, az)


def process_photo(ph: ph_mod.Photo, backend, with_marker: bool, marker_mm: float) -> PData:
    img = ph.load()
    depth = backend(ph, img)
    h, w = depth.shape
    K = ph.K.copy()
    sx, sy = w / img.shape[1], h / img.shape[0]
    Kd = K.copy()
    Kd[0, :] *= sx * (img.shape[1] / ph.width)
    Kd[1, :] *= sy * (img.shape[0] / ph.height)
    ds = cv2.GaussianBlur(depth, (0, 0), 1.0)
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    x = (uu + 0.5 - Kd[0, 2]) / Kd[0, 0]
    y = (vv + 0.5 - Kd[1, 2]) / Kd[1, 1]
    P = np.stack([x * ds, y * ds, ds], -1)
    N, ok = _normals(P)
    pd = PData(ph, img, depth, Kd, P, N, ok)
    pd.up = _estimate_up(N, ok)
    u = pd.up
    xw = np.array([1.0, 0, 0]) - u * u[0]
    xw /= np.linalg.norm(xw)
    yw = np.cross(u, xw)
    pd.M = np.stack([xw, yw, u], 0)
    if with_marker:
        Ki = K.copy()
        Ki[0, :] *= img.shape[1] / ph.width
        Ki[1, :] *= img.shape[0] / ph.height
        det = ph_mod.detect_marker(img, Ki, marker_mm / 1000.0, ident=7)
        if det:
            m = det[0]
            # compare with depth estimate at the marker centre
            px = m["px"]
            ix = int(np.clip(px[0] * sx, 0, w - 1))
            iy = int(np.clip(px[1] * sy, 0, h - 1))
            dz = float(np.median(depth[max(iy - 2, 0):iy + 3, max(ix - 2, 0):ix + 3]))
            if dz > 0.3:
                m["scale"] = m["z"] / dz
                pd.marker = m
    # floor / ceiling relative to the camera, along `up`
    zc = P @ u
    nv = np.abs(N @ u) > 0.85
    sel = ok & nv
    # the floor is the LOWEST strong horizontal plane (furniture tops are higher); the ceiling the highest
    fl = mode_peak(zc[sel & (zc < -0.4)], -2.2, -0.4, bin_w=0.04, min_frac=0.25, which="low", min_count=60)
    ce = mode_peak(zc[sel & (zc > 0.3)], 0.3, 2.6, bin_w=0.04, min_frac=0.25, which="high", min_count=60)
    pd.floor_z = fl[0] if fl else None
    pd.ceil_z = ce[0] if ce else None
    return pd


def _features(pd: PData):
    g = cv2.cvtColor(pd.img, cv2.COLOR_RGB2GRAY)
    if float(g.mean()) < 90:  # dim room: suppress sensor noise before contrast stretching
        g = cv2.fastNlMeansDenoising(g, None, h=9, templateWindowSize=5, searchWindowSize=15)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6)).apply(g)  # painted walls are low-contrast
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.006, edgeThreshold=15)
    kp, des = sift.detectAndCompute(g, None)
    pd.feats = (np.array([k.pt for k in kp], np.float32), des)


def _pair_rotation(a: PData, b: PData, K_a: np.ndarray, K_b: np.ndarray):
    if a.feats[1] is None or b.feats[1] is None or len(a.feats[0]) < 30 or len(b.feats[0]) < 30:
        return None
    bf = cv2.BFMatcher()
    m = bf.knnMatch(a.feats[1], b.feats[1], k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 25:
        return None
    pa = a.feats[0][[g.queryIdx for g in good]]
    pb = b.feats[0][[g.trainIdx for g in good]]
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None or inl.sum() < 25:
        return None
    Rm = np.linalg.inv(K_b) @ H @ K_a
    Rm /= np.cbrt(np.linalg.det(Rm))
    U, S, Vt = np.linalg.svd(Rm)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    if S.max() / S.min() > 1.35:
        return None
    return R, int(inl.sum())


def _profile(pd: PData):
    """Horizontal range profile (1 deg bins) around the camera in its own gravity frame + Manhattan offset."""
    Pw = pd.P.reshape(-1, 3) @ pd.M.T
    band = np.abs(Pw[:, 2]) < 0.5
    az = (np.rad2deg(np.arctan2(Pw[band, 1], Pw[band, 0])) % 360).astype(int)
    rho = np.hypot(Pw[band, 0], Pw[band, 1])
    prof = np.full(360, np.nan)
    for b in np.unique(az):
        v = rho[az == b]
        if len(v) >= 3:
            prof[b] = np.median(v)
    Nw = pd.nrm.reshape(-1, 3) @ pd.M.T
    okh = pd.ok.reshape(-1) & (np.abs(Nw[:, 2]) < 0.25)
    th = np.arctan2(Nw[okh, 1], Nw[okh, 0])
    psi = float(np.angle(np.exp(4j * th).sum()) / 4.0) if len(th) > 50 else None
    return prof, psi


def _geo_edge(pi: PData, pj: PData, expected: Optional[float] = None):
    """Relative yaw (y_j - y_i) from Manhattan offsets + range-profile consistency (works without texture)."""
    if pi.prof is None or pj.prof is None or pi.psi is None or pj.psi is None:
        return None
    best = []
    for m in range(4):
        delta = pi.psi - pj.psi + m * np.pi / 2
        sh = int(round(np.rad2deg(delta)))
        costs = []
        for b in range(360):
            ri = pi.prof[b]
            if np.isnan(ri):
                continue
            rj = pj.prof[(b - sh) % 360]
            if np.isnan(rj):
                continue
            costs.append(abs(ri - rj) / (0.5 * (ri + rj)))
        if len(costs) >= 15:
            best.append((float(np.median(costs)), len(costs), delta))
    if not best:
        return None
    best.sort()
    c, n, delta = best[0]
    second = best[1][0] if len(best) > 1 else 1.0
    if expected is not None:
        # consecutive shots of a person turning on the spot: the 90-degree ambiguity is resolved by the turn prior
        cand = [b for b in best if b[0] < 0.10 and b[1] >= 20]
        if not cand:
            return None
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
        c, n, delta = min(cand, key=lambda b: abs(wrap(b[2] - expected)))
        if abs(wrap(delta - expected)) > np.deg2rad(40):
            return None
        return float(wrap(delta)), float(n)
    if c > 0.06 or c > 0.5 * second or n < 30:
        return None
    return float((delta + np.pi) % (2 * np.pi) - np.pi), float(n)


def register_yaws(pds: List[PData]):
    n = len(pds)
    if n == 0:
        return
    for p in pds:
        _features(p)
        p.prof, p.psi = _profile(p)
    edges = []
    have_feat = set()
    turn = []
    for i in range(n):
        for j in range(i + 1, n):
            r = _pair_rotation(pds[i], pds[j], pds[i].ph.K, pds[j].ph.K)
            if r is None:
                continue
            R, ni = r  # x_j = R x_i
            Q = pds[j].M @ R @ pds[i].M.T
            ang = np.arctan2(Q[1, 0], Q[0, 0])
            if abs(Q[2, 2] - 1.0) <= 0.08:
                edges.append((i, j, -ang, float(ni)))
                have_feat.add((i, j))
                if j == i + 1:
                    turn.append(np.sign(-ang))
    sign = float(np.sign(np.sum(turn))) if turn and np.sum(turn) != 0 else -1.0  # default: clockwise turn
    step = sign * np.deg2rad(45.0)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in have_feat:
                continue
            g = _geo_edge(pds[i], pds[j], expected=step if j == i + 1 else None)
            if g is not None:
                edges.append((i, j, g[0], 0.25 * g[1]))
    # maximum-weight spanning forest (best edges first), largest tree = registered set
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = {i: [] for i in range(n)}
    for i, j, d, w in sorted(edges, key=lambda e: -e[3]):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree[i].append((j, d))
            tree[j].append((i, -d))
    comps: Dict[int, List[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    comp = max(comps.values(), key=lambda c: (len(c), -min(c)))
    deg = {i: sum(1 for e in edges if e[0] == i or e[1] == i) for i in comp}
    anchor = max(comp, key=lambda i: (deg[i], -i))
    yaw = {anchor: 0.0}
    stack = [anchor]
    while stack:
        u = stack.pop()
        for v, d in tree[u]:
            if v not in yaw:
                yaw[v] = yaw[u] + d
                stack.append(v)
    # robust refinement over all edges inside the component
    inside = [(i, j, d, w) for (i, j, d, w) in edges if i in yaw and j in yaw]
    for _ in range(6):
        num = {k: 0.0 for k in yaw}
        den = {k: 0.0 for k in yaw}
        for i, j, d, w in inside:
            r = (yaw[j] - yaw[i]) - d
            r = (r + np.pi) % (2 * np.pi) - np.pi
            hub = 1.0 if abs(r) < np.deg2rad(6) else np.deg2rad(6) / abs(r)
            ww = w * hub
            num[j] -= ww * r
            den[j] += ww
            num[i] += ww * r
            den[i] += ww
        for k in yaw:
            if k != anchor and den[k] > 0:
                yaw[k] += 0.5 * num[k] / den[k]
    for k, p in enumerate(pds):
        p.yaw = yaw.get(k)


def _world_rays(pd: PData, s: float = 1.0):
    h, w = pd.depth.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    x = (uu + 0.5 - pd.Kd[0, 2]) / pd.Kd[0, 0]
    y = (vv + 0.5 - pd.Kd[1, 2]) / pd.Kd[1, 1]
    dc = np.stack([x, y, np.ones_like(x)], -1).reshape(-1, 3)
    nrm = np.linalg.norm(dc, axis=1)
    dir_c = (dc / nrm[:, None]).astype(np.float32)
    rng = (pd.depth.reshape(-1) * nrm * s).astype(np.float32)
    return dir_c, rng, uu.reshape(-1), vv.reshape(-1)


def fit_room_photos(name: str, pds: List[PData], cfg: Config, scale: float) -> Tuple[RoomModel, Capture, dict]:
    reg = [p for p in pds if p.yaw is not None]
    info: Dict = dict(n_photos=len(pds), n_registered=len(reg), notes=[])
    # camera height / ceiling height
    hs = [-p.floor_z * scale for p in pds if p.floor_z is not None]
    h_cam = float(np.median(hs)) if hs else 1.40
    if not hs:
        info["notes"].append("floor not visible in any photo: camera height assumed 1.40 m")
    Hs = []
    for p in pds:
        if p.ceil_z is not None:
            hk = (-p.floor_z * scale) if p.floor_z is not None else h_cam
            Hs.append(p.ceil_z * scale + hk)
    if Hs:
        H = float(np.median(Hs))
        sigH = float(max(np.std(Hs) / np.sqrt(len(Hs)), 0.01)) if len(Hs) > 1 else 0.04
    else:
        H, sigH = PRIOR_CEILING, 0.3
        info["notes"].append("ceiling not visible in any photo: ceiling height is a prior (2.5 m)")
    # world rays (gravity frame, yaw-registered), camera at (0,0,h_cam)
    frames, parts = [], []
    horiz_ang = []
    wall_pts, wall_nrm, all_pts = [], [], []
    for k, p in enumerate(reg):
        dir_c, rng, uu, vv = _world_rays(p, scale)
        Rwc = np.array([[np.cos(p.yaw), -np.sin(p.yaw), 0], [np.sin(p.yaw), np.cos(p.yaw), 0], [0, 0, 1.0]]) @ p.M
        # camera->world rotation: dir_w = Rz(yaw) M dir_c ; our Frame.R is cam->world so R = Rz M
        fr = Frame(len(frames), Rwc, np.array([0.0, 0.0, h_cam]), p.ph.K, (p.ph.width, p.ph.height),
                   rgb=(lambda im=p.img: im), name=p.ph.name, kind="doorway" if p.ph.is_door else "frame",
                   link=p.ph.link)
        frames.append(fr)
        img = p.img
        sx = img.shape[1] / p.depth.shape[1]
        sy = img.shape[0] / p.depth.shape[0]
        col = img[np.clip(((vv + 0.5) * sy).astype(int), 0, img.shape[0] - 1),
                  np.clip(((uu + 0.5) * sx).astype(int), 0, img.shape[1] - 1)]
        sig = (0.06 * p.depth.reshape(-1)).astype(np.float32)
        parts.append((dir_c, rng, np.full(len(rng), KIND_HIT, np.uint8), np.full(len(rng), len(frames) - 1, np.int32),
                      np.stack([uu, vv], 1).astype(np.int16), sig, col.astype(np.uint8)))
        # horizontal normals in the world frame for the Manhattan yaw
        Nw = (p.nrm.reshape(-1, 3) @ p.M.T)
        Nw2 = Nw @ np.array([[np.cos(p.yaw), np.sin(p.yaw), 0], [-np.sin(p.yaw), np.cos(p.yaw), 0], [0, 0, 1.0]])
        okh = p.ok.reshape(-1) & (np.abs(Nw[:, 2]) < 0.25)
        th = np.arctan2(Nw2[okh, 1], Nw2[okh, 0])
        horiz_ang.append(th)
        Pw = (p.P.reshape(-1, 3) @ p.M.T) * scale
        Rz = np.array([[np.cos(p.yaw), -np.sin(p.yaw)], [np.sin(p.yaw), np.cos(p.yaw)]])
        xyw = Pw[:, :2] @ Rz.T
        zrel = Pw[:, 2] + h_cam
        okb = okh & (zrel > 0.8) & (zrel < H - 0.3)
        wall_pts.append(xyw[okb])
        wall_nrm.append(Nw2[okb, :2])
        okall = (zrel > 0.8) & (zrel < H - 0.3)
        all_pts.append(xyw[okall])
    rays = RaySet(*(np.concatenate([q[i] for q in parts]) for i in range(6)))
    rays.rgb = np.concatenate([q[6] for q in parts])
    cap = Capture(frames, rays, dict(tier="photos", depth_size=(reg[0].depth.shape[1], reg[0].depth.shape[0])))
    # Manhattan yaw
    th = np.concatenate(horiz_ang) if horiz_ang else np.zeros(0)
    yaw_m = float(np.angle(np.exp(4j * th).sum()) / 4.0) if len(th) else 0.0
    Rm = rot2(-yaw_m)
    xy_all = (np.concatenate(all_pts) if all_pts else np.zeros((0, 2))) @ Rm.T
    xy = (np.concatenate(wall_pts) if wall_pts else np.zeros((0, 2))) @ Rm.T
    nn = (np.concatenate(wall_nrm) if wall_nrm else np.zeros((0, 2))) @ Rm.T
    # each wall = the dominant plane among points whose normal faces the camera from that direction
    walls_c: Dict[str, Optional[Tuple[float, int]]] = {}
    for key, sign, ax in (("E", 1, 0), ("W", -1, 0), ("N", 1, 1), ("S", -1, 1)):
        facing = -sign * nn[:, ax]  # normal pointing back to the camera
        sel = (facing > 0.85) & (sign * xy[:, ax] > 0.25)
        along = sign * xy[sel, ax]
        res = mode_peak(along, 0.25, 14.0, bin_w=0.04, min_frac=0.0, which="max", smooth=2, refine_w=0.06,
                        min_count=40)
        walls_c[key] = res
    # Corridor check: when both spans look like a narrow room, the "walls" found along the long axis are door-jamb
    # ramps near the camera; the real end walls are the outermost well-supported planes.
    def span(a, b):
        return (walls_c[a][0] + walls_c[b][0]) if walls_c[a] and walls_c[b] else None

    sx, sy = span("E", "W"), span("N", "S")
    long_keys = None
    if sy is not None and sy < 1.8 and (sx is None or sx < 2.8):
        long_keys = (("E", 1, 0), ("W", -1, 0))
    elif sx is not None and sx < 1.8 and (sy is None or sy < 2.8):
        long_keys = (("N", 1, 1), ("S", -1, 1))
    if long_keys:
        for key, sign, ax in long_keys:
            facing = -sign * nn[:, ax]
            sel = (facing > 0.85) & (sign * xy[:, ax] > 0.25)
            res = mode_peak(sign * xy[sel, ax], 0.25, 14.0, bin_w=0.04, min_frac=0.08, which="high", smooth=2,
                            refine_w=0.06, min_count=60)
            if res is not None:
                walls_c[key] = res
        info["notes"].append("narrow room (corridor): end walls taken as the outermost well-supported planes")
    obs = {k: (v is not None and v[1] >= 250) for k, v in walls_c.items()}
    # a wall with no planar support but plenty of points beyond the camera: the room extends at least that far
    inferred = {}
    for key, sign, ax in (("E", 1, 0), ("W", -1, 0), ("N", 1, 1), ("S", -1, 1)):
        if obs[key]:
            continue
        along = sign * xy_all[:, ax]
        along = along[along > 0.25]
        if len(along) >= 300 and np.percentile(along, 98) > 1.0:
            walls_c[key] = (float(np.percentile(along, 98)), 0)
            inferred[key] = True
            info["notes"].append(f"{key} wall has no planar support: placed at the extent of the observed points (+-0.3 m)")
    dist = {k: (sg_ * 1.0) for k, sg_ in [(k, v[0]) for k, v in walls_c.items() if v is not None]}
    obs_geom = {k: (obs[k] or inferred.get(k, False)) for k in obs}
    # rectangle in the Manhattan frame, camera at the origin
    def coord(k, default):
        return dist.get(k, default)

    xmax = coord("E", PRIOR_ROOM[0] / 2)
    xmin = -coord("W", PRIOR_ROOM[0] / 2)
    ymax = coord("N", PRIOR_ROOM[1] / 2)
    ymin = -coord("S", PRIOR_ROOM[1] / 2)
    pair_x = obs_geom["E"] and obs_geom["W"]
    pair_y = obs_geom["N"] and obs_geom["S"]
    if not pair_x:
        info["notes"].append("east/west walls not both observed: room width is a prior")
        if obs_geom["E"] or obs_geom["W"]:
            xmin, xmax = (xmax - PRIOR_ROOM[0], xmax) if obs_geom["E"] else (xmin, xmin + PRIOR_ROOM[0])
    if not pair_y:
        info["notes"].append("north/south walls not both observed: room depth is a prior")
        if obs_geom["N"] or obs_geom["S"]:
            ymin, ymax = (ymax - PRIOR_ROOM[1], ymax) if obs_geom["N"] else (ymin, ymin + PRIOR_ROOM[1])
    runs = [["H", ymin], ["V", xmax], ["H", ymax], ["V", xmin]]
    walls = walls_from_runs(runs)
    nph = max(len(reg), 1)
    pos_sigma = 0.06 / np.sqrt(nph) + 0.03
    for w, key in zip(walls, ("S", "E", "N", "W")):
        w.observed = obs[key]
        w.support = int(walls_c[key][1]) if walls_c[key] else 0
        w.sigma = float(pos_sigma if obs[key] else (0.3 if inferred.get(key) else 0.9))
    rm = RoomModel(id=name, walls=walls, zf=0.0, zc=H, frames=list(range(len(frames))), Rm=Rm, yaw=yaw_m, name=name)
    rm.sigma_zf = 0.01
    rm.sigma_zc = sigH
    rm.stats["n_photos"] = len(pds)
    rm.notes.extend(info["notes"])
    info["h_cam"] = h_cam
    return rm, cap, info


def _photo_openings(rm: RoomModel, cap: Capture):
    ops = op_mod.detect_openings(cap, rm, beyond_ok=None, min_area=0.10, cos_min=0.12, photo_mode=True)
    return ops


def _door_scale(rm: RoomModel) -> Optional[float]:
    hs = [o.z1 - o.z0 for o in rm.openings if o.kind in ("door", "passage") and o.z0 < 0.05 and (o.z1 - o.z0) < 2.4
          and (o.s1 - o.s0) < 1.3]
    if not hs:
        return None
    return DOOR_H_PRIOR / float(np.median(hs))


def run_photos(path: str, cfg: Config):
    T: Dict[str, float] = {}
    t0 = time.time()
    rooms_ph = ph_mod.list_photos(path)
    backend = ph_mod.DepthAnythingV2()
    rms: List[RoomModel] = []
    infos: List[dict] = []
    extra_rel: Dict[str, float] = {}
    notes: List[str] = []
    n_photos = 0
    for name, phs in rooms_ph.items():
        pds = []
        for p in phs:
            if hasattr(backend, "available") and not backend.available(p):
                notes.append(f"{name}/{p.name}: no depth available, skipped")
                continue
            pds.append(process_photo(p, backend, True, cfg.scale_card_mm))
        n_photos += len(pds)
        if not pds:
            continue
        register_yaws(pds)
        if all(p.yaw is None for p in pds):
            pds[0].yaw = 0.0
        # --- scale: card (per photo, shared per room) else depth model; refined by door prior after pass 1
        card = [p.marker["scale"] for p in pds if p.marker]
        s_card = float(np.median(card)) if card else None
        scale = s_card if s_card else 1.0
        rm, cap, info = fit_room_photos(name, pds, cfg, scale)
        rm.openings = _photo_openings(rm, cap)
        sig_depth = DEPTH_MODEL_SIGMA * (0.7 + 0.3 / np.sqrt(max(len(pds), 1)))
        sig_rel = 0.012 if s_card else sig_depth
        if not s_card:
            ks = _door_scale(rm)
            if ks and not (0.88 <= ks <= 1.12):
                rm.notes.append(f"door-height prior rejected (implied scale x{ks:.2f} is implausible)")
                ks = None
            if ks:
                # inverse-variance fusion of the depth model (scale 1) and the door-height prior
                w1 = 1 / sig_depth ** 2
                w2 = 1 / (DOOR_H_SIGMA ** 2 + 0.02 ** 2)
                s_f = (w1 * 1.0 + w2 * ks) / (w1 + w2)
                if abs(s_f - 1) > 0.01:
                    scale = float(s_f)
                    rm, cap, info = fit_room_photos(name, pds, cfg, scale)
                    rm.openings = _photo_openings(rm, cap)
                sig_rel = float(np.sqrt(1 / (w1 + w2)))
                rm.notes.append(f"scale from door-height prior (x{scale:.3f})")
        else:
            rm.notes.append(f"scale from printed card (x{scale:.3f})")
        extra_rel[name] = float(sig_rel)
        rm.stats["scale"] = scale
        rm.stats["scale_source"] = 1.0 if s_card else 0.0
        rm.photo_info = info
        rm.cap_frames_names = [(f.name, f.kind, f.link) for f in cap.frames]
        rm.cap = cap
        rm.pds = [p for p in pds if p.yaw is not None]
        rms.append(rm)
        infos.append(info)
    T["rooms"] = time.time() - t0
    t0 = time.time()
    # --- door links from doorway shots --------------------------------------------------------
    links = _links_from_doorway_shots(rms)
    places, wall_t = _place_rooms(rms, links, cfg)
    T["stitch"] = time.time() - t0
    drift = dict(method="door-anchored room placement from doorway shots (photo tier has no shared frame)",
                 local_planar=False, door_graph=True, links=len(links), loops=max(0, len(links) - (len(rms) - 1)),
                 door_residual_rms_before=0.0,
                 door_residual_rms_after=round(float(np.sqrt(np.mean([l.residual ** 2 for l in links]))) if links else 0.0, 4),
                 wall_thickness_source="default")
    meta = dict(photos=n_photos, hazards=[])
    if cfg.damage:
        t0 = time.time()
        for rm in rms:
            regs = damage_mod.analyse_room(rm.cap, rm, "photos", extra_rel=extra_rel.get(rm.id, 0.0))
            rm.regs = regs
            rm.concealed_json = reasoning.concealed_flags(rm.id, regs, "photos", extra_rel=extra_rel.get(rm.id, 0.0))
        T["damage"] = time.time() - t0
    for rm in rms:
        notes.extend(f"{rm.id}: {n}" for n in rm.notes)
    out = assemble("photos", f"photos/{backend.name}", rms, places, links,
                   wall_t, meta, T, notes, drift, "iPhone 15+ (stills)", extra_rel=extra_rel)
    if cfg.damage:
        for rm, rj in zip(rms, out["rooms"]):
            surf = {s_["id"]: s_ for s_ in rj["surfaces"]}
            rj["scope"] = reasoning.scope_items(rm.id, rm.regs, rm.concealed_json, surf, "photos",
                                                extra_rel=extra_rel.get(rm.id, 0.0))
    return out, _PhotoRec(rms), places, links


@dataclass
class _PhotoRec:
    rooms: List[RoomModel]


def _shot_target(rm: RoomModel, fi: int):
    """Wall and along-wall position that a doorway shot is aimed at (optical axis x walls), or None."""
    fr = rm.cap.frames[fi]
    d = fr.R @ np.array([0.0, 0.0, 1.0])
    d2 = d[:2] @ rm.Rm.T
    nrm = np.linalg.norm(d2)
    if nrm < 1e-6:
        return None
    d2 = d2 / nrm
    o2 = fr.org[:2] @ rm.Rm.T
    best = None
    for wi, w in enumerate(rm.walls):
        ax = 0 if w.orient == "V" else 1
        oth = 1 - ax
        if abs(d2[ax]) < 1e-6:
            continue
        t = (w.coord - o2[ax]) / d2[ax]
        if t <= 0.1:
            continue
        s_hit = o2[oth] + t * d2[oth]
        if s_hit < w.s0 - 0.3 or s_hit > w.s1 + 0.3:
            continue
        if best is None or t < best[0]:
            best = (t, wi, float(np.clip(s_hit, w.s0 + 0.5, w.s1 - 0.5)))
    return None if best is None else (best[1], best[2])


def _links_from_doorway_shots(rms: List[RoomModel]) -> List[st.Link]:
    """Each `*_to_<room>` shot is aimed at a doorway.  Its optical axis hits a wall; the door/passage nearest to the hit
    is that doorway.  When no opening was detected there (the hall's doors are seen only at grazing angles) a doorway
    is *inferred* at the hit point with a wide uncertainty - the plan then still gets a link, flagged as inferred."""
    idx = {r.id: i for i, r in enumerate(rms)}
    shots: Dict[Tuple[int, str], Tuple[int, bool]] = {}  # (room idx, target) -> (opening idx, inferred?)
    used_by_room: Dict[int, set] = {}
    for ri, rm in enumerate(rms):
        for fi, (nm, kind, link) in enumerate(rm.cap_frames_names):
            if kind != "doorway" or link is None:
                continue
            tgt = _shot_target(rm, fi)
            if tgt is None:
                continue
            wi, s_hit = tgt
            used = used_by_room.setdefault(ri, set())
            cands = sorted(((abs(0.5 * (o.s0 + o.s1) - s_hit), oi) for oi, o in enumerate(rm.openings)
                            if o.wall == wi and o.kind in ("door", "passage") and oi not in used))
            inferred = False
            if cands and cands[0][0] < 0.8:
                oi = cands[0][1]
            else:
                rm.openings.append(Opening(wi, s_hit - 0.45, s_hit + 0.45, 0.0, DOOR_H_PRIOR, "door", sigma_w=0.25,
                                           sigma_h=0.3, confidence=0.3, evidence=dict(inferred=1.0)))
                oi = len(rm.openings) - 1
                inferred = True
                rm.notes.append(f"doorway to '{link}' inferred from the doorway shot (not detected in depth)")
            used.add(oi)
            shots[(ri, link)] = (oi, inferred)
    links: List[st.Link] = []
    seen = set()
    for (ri, tgt), (oi, inf_a) in shots.items():
        if tgt not in idx:
            continue
        rj = idx[tgt]
        key = tuple(sorted((ri, rj)))
        if key in seen:
            continue
        seen.add(key)
        other = shots.get((rj, rms[ri].id))
        if other is None:
            cand = [k for k, o in enumerate(rms[rj].openings) if o.kind in ("door", "passage")]
            wa = rms[ri].openings[oi]
            used = {v[0] for (r_, _), v in shots.items() if r_ == rj}
            cand = [k for k in cand if k not in used] or cand
            if not cand:
                continue
            oj, inf_b = min(cand, key=lambda k: abs((rms[rj].openings[k].s1 - rms[rj].openings[k].s0) - (wa.s1 - wa.s0))), True
        else:
            oj, inf_b = other
        lk = st.Link(ri, oi, rj, oj, source="photo")
        if inf_a or inf_b:
            lk.sigma_tan = 0.35
        links.append(lk)
    return links


def _place_rooms(rms: List[RoomModel], links: List[st.Link], cfg: Config):
    t = cfg.default_wall_thickness
    n = len(rms)
    if n == 0:
        return [], t
    if not links:
        places = [st.Placement() for _ in rms]
        x = 0.0
        for i, r in enumerate(rms):
            p = r.polygon
            places[i].p = np.array([x - p[:, 0].min(), -p[:, 1].min()])
            x += (p[:, 0].max() - p[:, 0].min()) + 0.5
            if i:
                r.notes.append("no adjacency evidence: placed beside the others (not a real position)")
        return places, t
    places, reached = st.bfs_place(rms, links, t)
    # rooms unreachable by doors: lay out beside the cluster
    xs = max([(st.polygons_plan([rms[i]], [places[i]])[0].bounds[2]) for i in reached])
    for i in range(n):
        if i not in reached:
            p = (places[i].R @ rms[i].polygon.T).T
            places[i].p = np.array([xs + 0.5 - p[:, 0].min(), -p[:, 1].min()])
            xs += (p[:, 0].max() - p[:, 0].min()) + 0.5
            rms[i].notes.append("not linked by any doorway shot: placed beside the others (not a real position)")
    st.solve_translations(rms, places, links, t, sigma_tan=0.06, sigma_norm=0.05, sigma_prior=3.0, anchor=0)
    return places, t
