"""Per-surface damage regions with class + metric extent.

1. Each surface (wall / ceiling / floor) is ortho-rectified into a 2 cm/px texture from every frame that
   sees it (depth-based occlusion test when depth exists, incidence/range weighting).
2. A quadratic lighting model is fitted to luminance; chroma is compared with the surface median.
3. Colour rules: yellow-brown chroma shift -> water_stain; dark speckled cluster -> mold; dark solid
   compact blob -> hole.  Axis-aligned near-perfect rectangles are treated as wall decorations.

This is a *classical, disclosed* detector (no learned model).  Known weak spots: paint colours that look
like stains, dark decorations without a rectangular outline, low light (we down-weight confidence), wet-look
glare (masked, never reported as damage).  See docs/KNOWN_FAILURES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import uncertainty as U
from .frames import Capture
from .geometry.room import RoomModel

RES = 0.02  # m / texel


@dataclass
class Texture:
    kind: str  # wall | ceiling | floor
    surface_id: str
    rgb: np.ndarray  # (nv, nu, 3) uint8; row 0 = v max (top) for walls
    weight: np.ndarray  # (nv, nu)
    valid: np.ndarray  # (nv, nu) bool: texel inside the surface and observed
    nu: int
    nv: int
    res: float
    origin_world: np.ndarray  # world point at (u=0, v=0)
    u_dir: np.ndarray  # world unit vector for +u
    v_dir: np.ndarray  # world unit vector for +v
    normal: np.ndarray
    low_light: bool = False
    resid: Optional[np.ndarray] = None  # mean (observed depth - plane depth), m; NaN = no depth observation
    dropfrac: Optional[np.ndarray] = None  # fraction of observations that were depth dropouts (glass/mirror)


def _surface_grid(room: RoomModel, kind: str, idx: int):
    Rinv = room.Rm.T
    zf, zc = room.zf, room.zc
    poly = room.polygon
    if kind == "wall":
        w = room.walls[idx]
        v0, v1 = poly[idx - 1], poly[idx]
        L = float(np.linalg.norm(v1 - v0))
        d = (v1 - v0) / max(L, 1e-9)
        n = np.array(w.out_n)
        o_xy = Rinv @ v0
        u_dir = np.array([*(Rinv @ d), 0.0])
        v_dir = np.array([0, 0, 1.0])
        normal = np.array([*(Rinv @ (-n)), 0.0])  # pointing into the room
        origin = np.array([o_xy[0], o_xy[1], zf])
        nu, nv = int(np.ceil(L / RES)), int(np.ceil((zc - zf) / RES))
        return origin, u_dir, v_dir, normal, nu, nv, None
    lo = poly.min(0)
    hi = poly.max(0)
    nu, nv = int(np.ceil((hi[0] - lo[0]) / RES)), int(np.ceil((hi[1] - lo[1]) / RES))
    zz = zc if kind == "ceiling" else zf
    origin_al = lo
    o_xy = Rinv @ origin_al
    u_dir = np.array([*(Rinv @ np.array([1.0, 0])), 0.0])
    v_dir = np.array([*(Rinv @ np.array([0, 1.0])), 0.0])
    normal = np.array([0, 0, -1.0 if kind == "ceiling" else 1.0])
    mask = np.zeros((nv, nu), np.uint8)
    cv2.fillPoly(mask, [((poly - lo) / RES).astype(np.int32)], 1)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    return np.array([o_xy[0], o_xy[1], zz]), u_dir, v_dir, normal, nu, nv, mask.astype(bool)


def build_texture(cap: Capture, room: RoomModel, kind: str, idx: int = 0, max_frames: int = 80) -> Optional[Texture]:
    origin, u_dir, v_dir, normal, nu, nv, mask = _surface_grid(room, kind, idx)
    if nu < 10 or nv < 10:
        return None
    uu, vv = np.meshgrid((np.arange(nu) + 0.5) * RES, (np.arange(nv) + 0.5) * RES)
    P = origin[None, None, :] + uu[..., None] * u_dir[None, None, :] + vv[..., None] * v_dir[None, None, :]
    P = P.reshape(-1, 3)
    acc = np.zeros((P.shape[0], 3), np.float64)
    wsum = np.zeros(P.shape[0], np.float64)
    rsum = np.zeros(P.shape[0], np.float64)
    rw = np.zeros(P.shape[0], np.float64)
    dsum = np.zeros(P.shape[0], np.float64)
    dtot = np.zeros(P.shape[0], np.float64)
    have_depth = False
    fr = room.frames
    if len(fr) > max_frames:
        fr = [fr[i] for i in np.linspace(0, len(fr) - 1, max_frames).astype(int)]
    for f in fr:
        frm = cap.frames[f]
        if frm.rgb is None:
            continue
        img = frm.rgb()
        if img is None:
            continue
        R, org = cap.R[f], cap.org[f]
        Pc = (P - org) @ R  # camera coords
        z = Pc[:, 2]
        ok = z > 0.3
        if not ok.any():
            continue
        K = frm.K
        sx = img.shape[1] / frm.size[0]
        sy = img.shape[0] / frm.size[1]
        px = K[0, 0] * Pc[:, 0] / np.where(ok, z, 1) + K[0, 2]
        py = K[1, 1] * Pc[:, 1] / np.where(ok, z, 1) + K[1, 2]
        ok &= (px >= 2) & (px < frm.size[0] - 2) & (py >= 2) & (py < frm.size[1] - 2)
        dvec = org - P
        dist = np.linalg.norm(dvec, axis=1)
        cosang = np.abs(dvec @ normal) / np.maximum(dist, 1e-6)
        ok &= cosang > 0.3
        if frm.depth is not None and ok.any():
            D, Kd = frm.depth()
            qx = np.clip((Kd[0, 0] * Pc[:, 0] / np.where(ok, z, 1) + Kd[0, 2]).astype(int), 0, D.shape[1] - 1)
            qy = np.clip((Kd[1, 1] * Pc[:, 1] / np.where(ok, z, 1) + Kd[1, 2]).astype(int), 0, D.shape[0] - 1)
            dd = D[qy, qx]
            vis = (dd <= 0) | (np.abs(dd - z) < 0.10 + 0.03 * z)
            ok &= vis
            have_depth = True
            wd = (cosang ** 2) / np.maximum(dist, 0.5) ** 2
            m_ = ok
            dsum[m_] += wd[m_] * (dd[m_] <= 0)
            dtot[m_] += wd[m_]
            m2 = ok & (dd > 0) & (np.abs(dd - z) < 0.10 + 0.03 * z)
            rsum[m2] += wd[m2] * (dd[m2] - z[m2])
            rw[m2] += wd[m2]
        if not ok.any():
            continue
        xi = np.clip((px[ok] * sx).astype(int), 0, img.shape[1] - 1)
        yi = np.clip((py[ok] * sy).astype(int), 0, img.shape[0] - 1)
        col = img[yi, xi].astype(np.float64)
        w = (cosang[ok] ** 2) / np.maximum(dist[ok], 0.5) ** 2
        acc[ok] += col * w[:, None]
        wsum[ok] += w
    valid = wsum > 0
    resid = None
    dropfrac = None
    if have_depth:
        resid = np.where(rw > 0, rsum / np.maximum(rw, 1e-12), np.nan).astype(np.float32).reshape(nv, nu)
        dropfrac = np.where(dtot > 0, dsum / np.maximum(dtot, 1e-12), 0.0).astype(np.float32).reshape(nv, nu)
        valid &= (dropfrac.reshape(-1) < 0.5)  # glass / mirror / unknown surface: not analysed
    rgb = np.zeros_like(acc)
    rgb[valid] = acc[valid] / wsum[valid, None]
    rgb = rgb.reshape(nv, nu, 3).astype(np.uint8)
    valid = valid.reshape(nv, nu)
    if mask is not None:
        valid &= mask
    if kind == "wall":
        # door / window openings (and a margin around them): other room, glass, jamb - never damage
        poly = room.polygon
        w_ = room.walls[idx]
        v0, v1 = poly[idx - 1], poly[idx]
        axp = 1 if w_.orient == "V" else 0
        dirn = 1.0 if v1[axp] >= v0[axp] else -1.0
        vv_ = valid.reshape(nv, nu)
        for o in room.openings:
            if o.wall != idx:
                continue
            ua, ub = (o.s0 - v0[axp]) * dirn, (o.s1 - v0[axp]) * dirn
            lo_u, hi_u = min(ua, ub) - 0.10, max(ua, ub) + 0.10
            i0, i1 = max(int(lo_u / RES), 0), min(int(np.ceil(hi_u / RES)), nu)
            j0, j1 = max(int((o.z0 - 0.10) / RES), 0), min(int(np.ceil((o.z1 + 0.10) / RES)), nv)
            vv_[j0:j1, i0:i1] = False
        valid = vv_.reshape(-1) if valid.ndim == 1 else vv_
    if kind == "wall":  # exclude baseboard strip and corners
        valid[: int(0.04 / RES), :] = False  # top 4 cm (v max) edge
        valid[nv - int(0.12 / RES):, :] = False  # baseboard zone (v < 12 cm)
        valid[:, : int(0.04 / RES)] = False
        valid[:, nu - int(0.04 / RES):] = False
    if kind == "wall":
        rgb = rgb[::-1]  # row 0 = top
        valid = valid[::-1]
        wsum = wsum.reshape(nv, nu)[::-1]
        if resid is not None:
            resid = resid[::-1]
            dropfrac = dropfrac[::-1]
    else:
        wsum = wsum.reshape(nv, nu)
    sid = f"{room.id}.w{idx}" if kind == "wall" else f"{room.id}.{kind}"
    return Texture(kind, sid, rgb, wsum, valid, nu, nv, RES, origin, u_dir, v_dir, normal, resid=resid,
                   dropfrac=dropfrac)


# ---------------------------------------------------------------------------
def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _fit_lighting(L: np.ndarray, valid: np.ndarray) -> np.ndarray:
    h, w = L.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx / max(w, 1) - 0.5)
    y = (yy / max(h, 1) - 0.5)
    X = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)[valid]
    t = L[valid]
    keep = np.ones(len(t), bool)
    coef = np.zeros(6)
    for _ in range(4):
        if keep.sum() < 20:
            break
        coef, *_ = np.linalg.lstsq(X[keep], t[keep], rcond=None)
        r = t - X @ coef
        s = 1.4826 * np.median(np.abs(r[keep])) + 1e-3
        keep = np.abs(r) < 2.5 * s
    full = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1) @ coef
    return full


@dataclass
class Region:
    cls: str
    mask: np.ndarray
    confidence: float
    area: float
    width: float
    height: float
    sigma_a: float
    bbox: Tuple[int, int, int, int]
    polygon_uv: List[List[float]]
    note: str = ""


def detect_regions(tex: Texture, low_light: bool = False) -> List[Region]:
    valid0 = tex.valid.copy()
    if valid0.sum() < 400:
        return []
    # fill speckle (isolated unobserved / dropout texels) so it neither splits regions nor counts as occlusion
    rgb_in = tex.rgb
    nf, nn = ndi.label(~valid0)
    if nn:
        sizes = np.bincount(nf.ravel())
        fill_m = np.isin(nf, [i for i in range(1, nn + 1) if sizes[i] < 25])
        if fill_m.any():
            rgb_in = cv2.inpaint(tex.rgb, cv2.dilate(fill_m.astype(np.uint8), np.ones((3, 3), np.uint8)), 3,
                                 cv2.INPAINT_TELEA)
            valid0 |= fill_m
    # keep a safety margin from invalid areas (occlusions, glass, surface edges)
    valid = cv2.erode(valid0.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    lab = _lab(rgb_in)
    glare = rgb_in.max(2) > 248
    ok = valid & ~glare
    if ok.sum() < 300:
        return []
    bg = _fit_lighting(lab[..., 0], ok)
    Lr = lab[..., 0] / np.maximum(bg, 1.0)
    b_med = np.median(lab[..., 2][ok])
    dB_s = cv2.GaussianBlur(lab[..., 2] - b_med, (0, 0), 1.2)
    Lr_s = cv2.GaussianBlur(Lr, (0, 0), 1.0)
    out: List[Region] = []
    has_depth = tex.resid is not None
    if low_light and not has_depth:
        return []  # too dark to judge colour from RGB alone: say so (room note) rather than hallucinate damage
    if has_depth:
        res_s = cv2.GaussianBlur(np.nan_to_num(tex.resid, nan=0.0), (0, 0), 1.5)
        deep = (res_s > 0.022) & ok
    # --- dark / deep regions -----------------------------------------------------------------
    dark = (Lr < 0.65) & ok
    cand = dark | (deep if has_depth else False)
    closed = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    lab_d, nd = ndi.label(closed)
    for c in range(1, nd + 1):
        m = lab_d == c
        area_px = int(m.sum())
        a_m2 = area_px * tex.res ** 2
        if a_m2 < 0.0015:
            continue
        raw = cand & m
        fill = raw.sum() / max(area_px, 1)
        _, nspots = ndi.label(raw)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        rectness = cv2.contourArea(cnt) / max(rect[1][0] * rect[1][1], 1.0)
        short_side = min(rect[1]) * tex.res
        long_side = max(rect[1]) * tex.res
        if short_side < 0.07 and area_px * tex.res ** 2 < 0.3:
            continue  # slivers along occlusion edges / seams
        if long_side / max(short_side, 1e-3) > 4.5:
            continue  # elongated: grazing-angle artefact or furniture edge, not damage
        res_frac = float((deep & m).sum() / max(area_px, 1)) if has_depth else 0.0
        if has_depth and res_frac > 0.5:
            if tex.kind != "wall" or a_m2 > 0.35:
                continue  # holes are wall openings; a big deep region on a ceiling/floor is a grazing-angle artefact
            cls, conf = "hole", float(np.clip(0.6 + 0.4 * res_frac, 0, 0.95))
        elif nspots >= (4 if has_depth else 6) and fill < 0.8 and a_m2 < 1.2 and tex.kind != "floor":  # floor seams look like speckle
            cls, conf = "mold", float(np.clip(0.5 + 0.04 * nspots, 0.5, 0.9))
        elif (not has_depth) and fill > 0.85 and nspots <= 2 and not (rectness > 0.965 and tex.kind == "wall")                 and 0.004 < a_m2 < 0.5:
            cls, conf = "hole", 0.5
        else:
            continue  # shadow, furniture edge, decoration, picture: flat and not speckled
        ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~m
        if (ring & ~valid0).sum() > 0.15 * max(ring.sum(), 1):
            continue
        out.append(_region(tex, cls, m, conf * (0.7 if low_light else 1.0)))
    # --- stains ---------------------------------------------------------------------------------
    stain = ((dB_s > 4.5) | ((dB_s > 3.0) & (Lr_s < 0.9))) & (Lr_s < 1.02) & (Lr_s > 0.6) & ok
    sm = cv2.morphologyEx(stain.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    sm = ndi.binary_fill_holes(sm)
    lab_s, ns = ndi.label(sm)
    invalid = ~valid0
    for c in range(1, ns + 1):
        m = lab_s == c
        if m.sum() * tex.res ** 2 < 0.02:
            continue
        ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~m
        if (ring & invalid).sum() > 0.25 * max(ring.sum(), 1):
            continue  # runs into an occluded / unobserved area: unreliable
        if has_depth and (deep & m).sum() > 0.3 * m.sum():
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        rectness = cv2.contourArea(cnt) / max(rect[1][0] * rect[1][1], 1.0)
        ang = rect[2] % 90
        if rectness > 0.93 and min(ang, 90 - ang) < 5 and tex.kind == "wall":
            continue  # picture / switch plate
        Lm = lab[..., 0][m]
        if Lm.std() / max(Lm.mean(), 1.0) > 0.16:
            continue  # busy texture: decoration, not a stain
        if min(rect[1]) * tex.res < 0.07 or m.sum() * tex.res ** 2 > 1.2:
            continue  # too thin, or too big to be a stain (lighting gradient / glare patch)
        if max(rect[1]) / max(min(rect[1]), 1e-3) > 5:
            continue
        strength = float(np.clip(np.mean(dB_s[m]) / 12.0, 0, 1))
        out.append(_region(tex, "water_stain", m, (0.45 + 0.45 * strength) * (0.7 if low_light else 1.0)))
    return out


def _region(tex: Texture, cls: str, m: np.ndarray, conf: float) -> Region:
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    per = cv2.arcLength(cnt, True) * tex.res
    approx = cv2.approxPolyDP(cnt, 1.5, True)[:, 0, :]
    # metric uv (u from left, v measured up from the floor for walls / from texture origin otherwise)
    poly = []
    for x, y in approx:
        u = (x + 0.5) * tex.res
        v = ((tex.nv - (y + 0.5)) if tex.kind == "wall" else (y + 0.5)) * tex.res
        poly.append([round(float(u), 3), round(float(v), 3)])
    area = float(m.sum() * tex.res ** 2)
    return Region(cls, m, float(conf), area, float((x1 - x0) * tex.res), float((y1 - y0) * tex.res),
                  float(per * 0.015), (int(x0), int(y0), int(x1), int(y1)), poly)


# ---------------------------------------------------------------------------
def analyse_room(cap: Capture, room: RoomModel, tier: str, extra_rel: float = 0.0, max_frames: int = 80):
    """Fill room.damage_json, returning list of (surface_id, Region) for downstream reasoning."""
    regs = []
    low = False
    # luminance-based low-light flag from the room's frames
    lum = []
    for f in room.frames[:: max(1, len(room.frames) // 12)]:
        im = cap.frames[f].rgb() if cap.frames[f].rgb else None
        if im is not None:
            lum.append(float(im.mean()))
    if lum and np.median(lum) < 95:
        low = True
        room.notes.append("low light: damage confidence reduced")
    surfaces = [("wall", i) for i in range(len(room.walls))] + [("ceiling", 0), ("floor", 0)]
    djson = []
    n = 0
    for kind, idx in surfaces:
        if kind == "wall" and not room.walls[idx].observed:
            continue
        tex = build_texture(cap, room, kind, idx, max_frames=max_frames)
        if tex is None:
            continue
        for r in detect_regions(tex, low):
            area_m = U.measure(r.area, r.sigma_a, tier, "extent", extra_rel)
            djson.append(dict(
                id=f"{room.id}.dmg{n}", surface_id=tex.surface_id, **{"class": r.cls},
                confidence=round(min(r.confidence, 0.99), 3),
                extent=dict(width=U.measure(r.width, 0.02, tier, "extent", extra_rel),
                            height=U.measure(r.height, 0.02, tier, "extent", extra_rel),
                            area=area_m),
                polygon_uv=r.polygon_uv))
            regs.append((tex.surface_id, kind, idx, r, f"{room.id}.dmg{n}"))
            n += 1
    room.damage_json = djson
    return regs
