"""Reconstruction for captures that share one world frame (LiDAR, video): segment rooms, fit each one."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi

from ..config import Config
from ..frames import Capture
from ..geometry import room as room_mod
from ..geometry.primitives import carve, estimate_yaw, mode_peak, rot2, segment_rooms
from ..geometry.room import RoomModel


@dataclass
class Reconstruction:
    cap: Capture
    rooms: List[RoomModel]
    yaw: float
    Rm: np.ndarray
    zf0: float
    zc0: float
    labels: np.ndarray
    grid: object
    timings: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    lab_full: Optional[np.ndarray] = None


PRIOR_CEILING, PRIOR_CEILING_SIGMA = 2.5, 0.25  # used only when the ceiling was not observed


def _height_scale() -> float:
    import json
    import os
    if os.environ.get("FLOORSCAN_NO_DEPTHCAL"):
        return 1.0
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibration.json")) as fh:
            return float(json.load(fh).get("_lidar_height", {}).get("scale", 1.0))
    except Exception:
        return 1.0


def _lookup_label(labels, g, xy):
    ij = g.ij(xy)
    ij[:, 0] = np.clip(ij[:, 0], 0, g.shape[1] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, g.shape[0] - 1)
    return labels[ij[:, 1], ij[:, 0]]


def reconstruct_global(cap: Capture, cfg: Config) -> Reconstruction:
    T: Dict[str, float] = {}
    t0 = time.time()
    n = cap.n_frames
    d = cap.world(range(n), stride=4)
    cam_z = float(np.median(cap.org[:, 2]))
    z = d["pts"][:, 2]
    fl = mode_peak(z, cam_z - 2.5, cam_z - 0.3, bin_w=0.02, min_frac=0.1, which="low", min_count=30)  # floor = lowest real plane (beds hide most of it)
    ce = mode_peak(z, cam_z + 0.3, cam_z + 2.5, bin_w=0.02, min_frac=0.3, which="high", min_count=30)
    notes = []
    if ce and fl and room_mod.plane_coverage(d["pts"][:, :2], z, ce[0]) < 0.3 * room_mod.plane_coverage(d["pts"][:, :2], z, fl[0]):
        ce = None  # furniture top, not the ceiling (see room.levels)
    if ce and fl and ce[0] - fl[0] < room_mod.MIN_CEILING:
        ce = None
    zf0 = fl[0] if fl else cam_z - 1.45
    zc0 = ce[0] if ce else zf0 + 2.5
    if not fl or not ce:
        notes.append("floor/ceiling level not observed; assumed from camera height")
    m = (z > zf0 + 1.0) & (z < zc0 - 0.25)
    yaw = estimate_yaw(d["pts"][m, :2])
    Rm = rot2(-yaw)
    T["levels_yaw"] = time.time() - t0
    t0 = time.time()
    g, H, F, _ = carve(cap, range(n), yaw, zf0, zc0, res=cfg.grid_res)
    frame_xy = cap.org[:, :2] @ Rm.T
    labels, free, D = segment_rooms(g, H, F, frame_xy, persistence=cfg.persistence, min_area=cfg.min_room_area)
    T["segment"] = time.time() - t0
    t0 = time.time()
    # assign each frame to a room
    if (labels == 0).any():
        idx = ndi.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
        lab_full = labels[idx[0], idx[1]]
    else:
        lab_full = labels
    flab = _lookup_label(lab_full, g, frame_xy)
    rooms: List[RoomModel] = []
    for li in [l for l in np.unique(labels) if l > 0]:
        fr = [int(f) for f in np.nonzero(flab == li)[0]]
        if len(fr) < 4:
            notes.append(f"region {li}: only {len(fr)} frames, skipped")
            continue
        mask = labels == li
        lab_cc, ncc = ndi.label(mask)
        if ncc > 1:
            big = np.argmax(np.bincount(lab_cc.ravel())[1:]) + 1
            mask = lab_cc == big
        mask = ndi.binary_fill_holes(mask)
        runs = room_mod.runs_from_mask(mask, g)
        walls = room_mod.walls_from_runs(runs)
        rm = RoomModel(id=f"room_{len(rooms) + 1}", walls=walls, zf=zf0, zc=zc0, frames=fr, Rm=Rm, yaw=yaw)
        rm.label = int(li)
        rooms.append(rm)
    T["regions"] = time.time() - t0
    # ---- per-room fit -----------------------------------------------------------------------
    t0 = time.time()
    for rm in rooms:
        _fit_room(cap, rm, cfg)
    keep = []
    for rm in rooms:  # a fragment that collapsed during the fit (fewer than 4 walls / tiny) is not a room
        if len(rm.walls) >= 4 and rm.area() >= cfg.min_room_area:
            keep.append(rm)
        else:
            notes.append(f"region of {len(rm.frames)} frames dropped: no closed room outline after fitting")
    if not keep and rooms:  # never return an empty plan: keep the largest closed outline
        best = max((r for r in rooms if len(r.walls) >= 4), key=lambda r: r.area(), default=None)
        keep = [best] if best is not None else []
    rooms = keep
    T["room_fit"] = time.time() - t0
    # order rooms by first frame visited, rename deterministically
    rooms.sort(key=lambda r: min(r.frames))
    for i, r in enumerate(rooms):
        r.id = f"room_{i + 1}"
    return Reconstruction(cap, rooms, yaw, Rm, zf0, zc0, labels, g, T, notes, lab_full)


def _fit_room(cap: Capture, rm: RoomModel, cfg: Config):
    Rm = rm.Rm
    walls = rm.walls
    zf, zc = rm.zf, rm.zc
    for it in range(cfg.planar_iters if cfg.local_planar else 1):
        xy, fr, bm, bd = room_mod._band_points(cap, rm.frames, Rm, zf, zc)
        walls = room_mod.refine_walls(walls, xy, fr, win=cfg.wall_window, min_frac=cfg.wall_min_frac,
                                      chunk=cfg.chunk_frames, z=bd["pts"][bm, 2], zc=zc)
        walls = room_mod.drop_objects(walls)
        walls = room_mod.simplify_walls(walls)
        poly = room_mod.polygon_from_walls(walls)
        fl, ce, *_ = room_mod.levels(cap, rm.frames, poly, Rm)
        if fl:
            zf = fl[0]
        if ce:
            zc = ce[0]
        elif fl:
            zc = zf + PRIOR_CEILING  # ceiling not observed: prior for the wall band
        if cfg.local_planar and it < cfg.planar_iters - 1:
            rms0 = room_mod.planar_correct(cap, rm.frames, walls, Rm, zf, zc, chunk=cfg.chunk_frames)
            rm.stats[f"rms_iter{it}"] = rms0
    rm.walls = walls
    rm.zf, rm.zc = zf, zc
    # levels uncertainty from per-chunk consistency
    poly = room_mod.polygon_from_walls(walls)
    fl, ce, d, xy, inside = room_mod.levels(cap, rm.frames, poly, Rm)
    if fl:
        rm.zf = fl[0]
    if ce:
        rm.zc = ce[0]
    zc_pts = d["pts"][:, 2]
    for name, val in (("zf", rm.zf), ("zc", rm.zc)):
        sel = inside & (np.abs(zc_pts - val) < 0.03)
        cid = d["frame"][sel] // cfg.chunk_frames
        meds = [np.median(zc_pts[sel][cid == c]) for c in np.unique(cid) if (cid == c).sum() >= 20]
        s = float(np.std(meds) / np.sqrt(len(meds))) if len(meds) >= 3 else 0.006
        setattr(rm, "sigma_" + name, float(np.sqrt(0.002 ** 2 + s ** 2)))
    if ce and cap.meta.get("tier") == "lidar":
        # dev-fitted height calibration: device LiDAR floor-to-ceiling reads ~1 % short vs the laser (horizontal
        # distances do not, so ranges are not rescaled) - dataprep/fit_lidar_scale.py, calibration.json "_lidar_height"
        rm.zc = rm.zf + (rm.zc - rm.zf) * _height_scale()
    if not ce:  # the capture never looked at the ceiling: report the prior with an honest interval
        rm.zc = rm.zf + PRIOR_CEILING
        rm.sigma_zc = PRIOR_CEILING_SIGMA
        rm.notes.append(f"ceiling not observed (no horizontal plane above the walls covers the room): height is a prior "
                        f"{PRIOR_CEILING:.1f} m +- {PRIOR_CEILING_SIGMA:.2f} m")
    rm.stats["n_frames"] = len(rm.frames)
    rm.stats["ceiling_peak_support"] = float(ce[1]) if ce else 0.0
