"""Opening (door / passage / window) detection from ray evidence against fitted wall planes.

For every wall we rasterise, in (along-wall, height) at 1 cm, three kinds of evidence from the room's
own frames: W = ray ended on the wall plane, O = ray crossed the plane and ended beyond it, D = ray
crossed the plane and got no return (glass / outdoors).  Occluded rays (ended in front of the plane)
carry no information.  Openings are connected regions dominated by O or D; edges are located by sub-cell
interpolation of the wall/opening evidence profile.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from scipy import ndimage as ndi

from ..frames import KIND_DROP, KIND_HIT, Capture
from .room import Opening, RoomModel

CELL = 0.01


def _profile_cross(frac_open: np.ndarray, known: np.ndarray, i0: int, i1: int, rising: bool) -> Optional[float]:
    """Sub-cell location where open-fraction crosses 0.5 within [i0, i1)."""
    f = frac_open[i0:i1]
    k = known[i0:i1]
    if len(f) < 3:
        return None
    # smooth along the profile
    num = ndi.uniform_filter1d(f * k, 3, mode="nearest")
    den = ndi.uniform_filter1d(k.astype(float), 3, mode="nearest")
    fo = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    idx = range(len(fo) - 1) if rising else range(len(fo) - 2, -1, -1)
    for i in idx:
        a, b = fo[i], fo[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if rising and a < 0.5 <= b:
            t = (0.5 - a) / max(b - a, 1e-6)
            return i0 + i + 0.5 + t
        if (not rising) and b < 0.5 <= a:
            t = (a - 0.5) / max(a - b, 1e-6)
            return i0 + i + 0.5 + (1 - t)
    return None


def _rgb_samples(cap: Capture, d: dict, mask: np.ndarray, n: int = 500) -> Optional[np.ndarray]:
    """Mean-able RGB samples at the image pixels of the selected rays (deterministic subsample)."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return None
    rng = np.random.default_rng(0)
    if idx.size > n:
        idx = np.sort(rng.choice(idx, n, replace=False))
    ds = cap.meta.get("depth_size")
    cols = []
    cache = {}
    for i in idx:
        f = int(d["frame"][i])
        if f not in cache:
            cache[f] = cap.frames[f].rgb() if cap.frames[f].rgb else None
        img = cache[f]
        if img is None:
            continue
        ridx = d["idx"][i]
        u, v = cap.rays.px[ridx]
        sx = img.shape[1] / (ds[0] if ds else img.shape[1])
        sy = img.shape[0] / (ds[1] if ds else img.shape[0])
        x = int(np.clip(u * sx, 0, img.shape[1] - 1))
        y = int(np.clip(v * sy, 0, img.shape[0] - 1))
        cols.append(img[y, x].astype(float))
    return np.array(cols) if cols else None


def detect_openings(cap: Capture, room: RoomModel, thresh_open: float = 0.55,
                    min_area: float = 0.12, cos_min: float = 0.30, photo_mode: bool = False,
                    beyond_diff: Optional[float] = None, beyond_ok: Optional[Callable[[np.ndarray], bool]] = None,
                    rays_stride: int = 1) -> List[Opening]:
    d = cap.world(room.frames, kinds=(KIND_HIT, KIND_DROP), stride=rays_stride)
    if len(d["rng"]) == 0:
        return []
    Rm = room.Rm
    o2 = d["org"][:, :2] @ Rm.T
    dir2 = d["dirs"][:, :2] @ Rm.T
    dz = d["dirs"][:, 2]
    oz = d["org"][:, 2]
    rng = d["rng"]
    kind = d["kind"]
    glass_ray = np.zeros(len(rng), bool)
    if photo_mode and cap.rays.rgb is not None:
        col = cap.rays.rgb[d["idx"]].astype(np.float32)
        lum_ = 0.299 * col[:, 0] + 0.587 * col[:, 1] + 0.114 * col[:, 2]
        glass_ray = ((col[:, 2] - col[:, 0]) > 30) & (lum_ > 140)  # sky-tinted bright glass
    out: List[Opening] = []
    zf, zc = room.zf, room.zc
    for w in room.walls:
        if not w.observed:
            continue
        n = np.array(w.out_n)
        sg = n[0] + n[1]
        ax = 0 if w.orient == "V" else 1
        oth = 1 - ax
        den = dir2[:, ax] * sg
        inside = sg * (o2[:, ax] - w.coord) < -0.05
        okc = inside & (den > cos_min)  # steep incidence rejected: plane-position error is amplified by tan(theta)
        if not okc.any():
            continue
        t = np.full(len(rng), np.nan)
        t[okc] = (w.coord - o2[okc, ax]) / dir2[okc, ax]
        t = np.where(okc, t, 0.0)
        s = o2[:, oth] + t * dir2[:, oth]
        z = oz + t * dz - zf
        sel = okc & (s > w.s0 - 0.02) & (s < w.s1 + 0.02) & (z > -0.02) & (z < zc - zf + 0.02)
        if sel.sum() < 100:
            continue
        diff = rng - t
        is_hit = kind == KIND_HIT
        is_w = sel & is_hit & (np.abs(diff) < (0.10 if photo_mode else 0.06)) & ~glass_ray
        thr_o = beyond_diff if beyond_diff is not None else (0.20 if photo_mode else 0.08)
        is_o = sel & is_hit & (diff > thr_o)
        is_d = sel & ((kind == KIND_DROP) | (glass_ray & is_hit & (np.abs(diff) < 0.15)))
        nx = int(np.ceil((w.s1 - w.s0) / CELL)) + 1
        nz = int(np.ceil((zc - zf) / CELL)) + 1
        si = np.clip(((s - w.s0) / CELL).astype(int), 0, nx - 1)
        zi = np.clip((z / CELL).astype(int), 0, nz - 1)

        wgt = np.where(okc, np.clip(den, 0, 1) ** (1 if photo_mode else 3), 0.0)  # frontal rays localise edges best

        def acc(mask):
            lin = (zi[mask] * nx + si[mask])
            return np.bincount(lin, weights=wgt[mask], minlength=nx * nz).reshape(nz, nx).astype(np.float32)

        W, O, Dd = acc(is_w), acc(is_o), acc(is_d)
        box = 7
        Ws, Os, Ds = (ndi.uniform_filter(a, box) * box * box for a in (W, O, Dd))
        tot = Ws + Os + Ds
        known_cells = tot[tot > 0.5]
        dens_ref = float(np.median(known_cells)) if known_cells.size else 0.0
        tot_min = max(1.5, 0.25 * dens_ref)
        frac = np.where(tot > 0.01, (Os + Ds) / np.maximum(tot, 1e-6), 0)
        mask = (tot >= tot_min) & (frac > thresh_open)
        mask = ndi.binary_opening(mask, np.ones((5, 5), bool))
        lab, ncomp = ndi.label(mask)
        for c in range(1, ncomp + 1):
            m = lab == c
            area = m.sum() * CELL * CELL
            rows, cols = np.nonzero(m)
            if area < min_area or (cols.max() - cols.min()) * CELL < 0.25:
                continue
            if float(O[m].sum() + Dd[m].sum()) < 120:
                continue
            r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
            # raw evidence totals inside the component
            ev = dict(W=float(W[m].sum()), O=float(O[m].sum()), D=float(Dd[m].sum()))
            known_raw = (W + O + Dd)
            open_raw = (O + Dd)
            # --- horizontal edges: profile over the core rows ------------------------------
            h = r1 - r0
            ra, rb = int(r0 + 0.15 * h), int(r1 - 0.15 * h)
            ra, rb = (ra, rb) if rb > ra else (r0, r1)
            kn = known_raw[ra:rb + 1].sum(0)
            op = open_raw[ra:rb + 1].sum(0)
            frac_c = np.where(kn > 0, op / np.maximum(kn, 1e-6), 0.0)
            lo_i, hi_i = max(c0 - 25, 0), min(c1 + 26, nx)
            left = _profile_cross(frac_c, kn > 2, lo_i, c0 + 8, True)
            right = _profile_cross(frac_c, kn > 2, c1 - 8, hi_i, False)
            if left is None or right is None or right <= left:
                left, right = c0 + 0.0, c1 + 1.0
            # --- vertical edges: profile over the core columns -----------------------------
            wd = c1 - c0
            ca, cb = int(c0 + 0.15 * wd), int(c1 - 0.15 * wd)
            ca, cb = (ca, cb) if cb > ca else (c0, c1)
            knv = known_raw[:, ca:cb + 1].sum(1)
            opv = open_raw[:, ca:cb + 1].sum(1)
            frac_v = np.where(knv > 0, opv / np.maximum(knv, 1e-6), 0.0)
            if r0 <= 4:
                bottom = 0.0
            else:
                b_ = _profile_cross(frac_v, knv > 2, max(r0 - 25, 0), r0 + 8, True)
                bottom = b_ if b_ is not None else float(r0)
            top_ = _profile_cross(frac_v, knv > 2, r1 - 8, min(r1 + 26, nz), False)
            top = top_ if top_ is not None else float(r1 + 1)
            if r1 >= nz - 6:
                top = float(min(top, nz - 1))
            s0 = w.s0 + left * CELL
            s1 = w.s0 + right * CELL
            z0, z1 = bottom * CELL, top * CELL
            # --- classify ---------------------------------------------------------------
            tot_ev = ev["O"] + ev["D"] + 1e-9
            d_frac = ev["D"] / tot_ev
            # jamb evidence + wall thickness from beyond-rays close to the edges
            jam = 0
            thick = None
            near = is_o & (diff < 0.5) & (((s > s0 - 0.2) & (s < s0 + 0.02)) | ((s > s1 - 0.02) & (s < s1 + 0.2)))                 & (z > z0) & (z < z1)
            jam = int(near.sum())
            if jam >= 20:
                thick = float(np.percentile(diff[near], 90))
            cen_s = 0.5 * (s0 + s1)
            beyond = False
            if beyond_ok is not None:
                p = np.zeros(2)
                p[oth] = cen_s
                p[ax] = w.coord + sg * 1.0
                beyond = bool(beyond_ok(p))  # p is in the aligned frame
            through = (jam >= 15 or beyond) and ev["O"] >= 0.1 * tot_ev
            if photo_mode and not through:
                # no neighbour regions in the photo tier: a tall floor-touching region of "beyond" depth is a doorway
                through = ev["O"] >= 0.6 * tot_ev and (z1 - z0) >= 1.5 and z0 < 0.35
            typ = "through" if through else ("dropout" if d_frac > 0.80 else "mixed")
            if typ == "through" and z0 > 0.12:
                # a doorway whose lower part is only seen at grazing angles: no wall hits below it -> it reaches the floor
                ia, ib = int((s0 + 0.05 - w.s0) / CELL), int((s1 - 0.05 - w.s0) / CELL)
                za, zb = int(0.05 / CELL), int(max(z0 - 0.05, 0.06) / CELL)
                wb = float(W[za:zb, ia:ib].sum()) if ib > ia and zb > za else 0.0
                allb = float((W + O + Dd)[za:zb, ia:ib].sum()) if ib > ia and zb > za else 0.0
                if wb <= 0.1 * max(allb, 1e-6) or allb < 1.0:
                    z0 = 0.0
            kind_s = "window" if z0 > 0.12 else ("passage" if (s1 - s0) > 1.3 else "door")
            keep = False
            note = ""
            glass = {}
            if typ == "through":
                keep = True
                note = "through"
            elif typ == "dropout":
                keep = True
                if z0 > 0.12:  # window candidate: glass looks bright / sky-tinted, a mirror reflects the room
                    in_rect = is_d & (s > s0) & (s < s1) & (z > z0) & (z < z1)
                    ref_rect = is_w & (z > 1.0) & (z < zc - zf - 0.3)
                    a = _rgb_samples(cap, d, in_rect)
                    b = _rgb_samples(cap, d, ref_rect)
                    if a is not None and b is not None and len(a) > 20:
                        lum = lambda c: 0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]  # noqa: E731
                        la, lb = float(lum(a).mean()), float(np.median(lum(b)))
                        tint = float((a[:, 2] - a[:, 0]).mean())
                        glass = dict(lum_opening=la, lum_wall=lb, blue_tint=tint)
                        if not (tint > 15 or la > lb + 22):
                            keep = False
                            note = "suspect_mirror"
            else:
                note = "suspect_mirror_or_glass"
            pa = cap.meta.get("px_angle")
            if pa and not photo_mode:
                # FIX (fixloop): a boundary depth pixel straddles the edge, so the evidence mask is dilated by ~half a pixel
                # per side.  Move every edge inward by half a depth-pixel footprint at the range of the rays that locate it.
                def _rng(mask):
                    return float(np.median(t[mask])) if mask.sum() >= 10 else 2.5

                inside_z = (z > z0) & (z < z1)
                dl = 0.5 * pa * _rng(sel & (np.abs(s - s0) < 0.06) & inside_z)
                dr = 0.5 * pa * _rng(sel & (np.abs(s - s1) < 0.06) & inside_z)
                dt = 0.5 * pa * _rng(sel & (np.abs(z - z1) < 0.06) & (s > s0) & (s < s1))
                s0, s1, z1 = s0 + dl, s1 - dr, z1 - dt
                if z0 > 0.05:
                    z0 += 0.5 * pa * _rng(sel & (np.abs(z - z0) < 0.06) & (s > s0) & (s < s1))
            # --- evidence quality -> per-opening uncertainty ---------------------------------------------
            near_e = sel & (((np.abs(s - s0) < 0.15) | (np.abs(s - s1) < 0.15))) & (z > z0) & (z < z1)
            cosq = float(np.clip(np.average(den[near_e]), 0, 1)) if near_e.sum() >= 10 else 0.3
            ia, ib = int((s0 + 0.05 - w.s0) / CELL), int((s1 - 0.05 - w.s0) / CELL)
            ja, jb = int((z0 + 0.10) / CELL), int((z1 - 0.10) / CELL)
            conflict = 0.0
            if ib > ia and jb > ja:
                win = float(W[ja:jb, ia:ib].sum())
                tot_in = float((W + O + Dd)[ja:jb, ia:ib].sum())
                conflict = win / tot_in if tot_in > 1e-6 else 0.0
            sig_w = float(np.hypot(0.006, np.hypot(0.05 * (1.0 - cosq) ** 2, 0.20 * conflict)))
            op = Opening(w.idx, float(s0), float(s1), float(z0), float(z1), kind_s,
                         sigma_w=sig_w, sigma_h=float(np.hypot(0.012, 0.5 * sig_w)),
                         confidence=float(min(1.0, 0.5 + 0.5 * min(ev["O"] + ev["D"], 2000) / 2000)),
                         evidence=dict(ev, jamb=float(jam), beyond=float(beyond), type_=typ, cosq=cosq, conflict=float(conflict),
                                       thickness=float(thick) if thick else -1.0, **glass))
            if keep:
                out.append(op)
            else:
                room.notes.append(f"rejected {note} candidate on wall {w.idx} ({s1 - s0:.2f}x{z1 - z0:.2f} m)")
    out.sort(key=lambda o: (o.wall, o.s0))
    return out
