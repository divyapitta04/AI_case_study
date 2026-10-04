"""Tier-agnostic observation container.

Every input tier is reduced to the same thing: a set of camera frames (pose, intrinsics) and a bag of
*rays* with a measured range (or a dropout).  World frame: Z up, metres, gravity aligned; yaw arbitrary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

KIND_IGNORE, KIND_HIT, KIND_DROP = 0, 1, 2


@dataclass
class Frame:
    index: int
    R: np.ndarray  # (3,3) camera(CV: x right, y down, z fwd) -> world
    org: np.ndarray  # (3,) camera centre in world
    K: np.ndarray  # (3,3) intrinsics of the RGB image
    size: tuple  # (w, h) of RGB image
    rgb: Optional[Callable[[], np.ndarray]] = None  # lazy loader -> HxWx3 uint8 RGB
    room_hint: Optional[str] = None  # photo tier: room folder
    name: str = ""
    kind: str = "frame"  # frame | doorway
    link: Optional[str] = None
    depth: Optional[Callable] = None  # lazy loader -> (HxW depth z-metres, K_depth) for occlusion tests


@dataclass
class RaySet:
    dir_cam: np.ndarray  # (N,3) float32 unit direction in camera frame
    rng: np.ndarray  # (N,) float32 euclidean range, 0 where no return
    kind: np.ndarray  # (N,) uint8
    frame: np.ndarray  # (N,) int32 frame index
    px: np.ndarray  # (N,2) int16 (u,v) pixel in the *ray source* image
    sigma: np.ndarray  # (N,) float32 1-sigma range error (m)
    starts: np.ndarray = field(default=None)  # (F+1,) offsets (rays are frame-contiguous)
    rgb: Optional[np.ndarray] = None  # (N,3) uint8 colour at the ray's pixel (photo / video tiers)

    def __post_init__(self):
        if self.starts is None:
            n_f = int(self.frame.max()) + 1 if len(self.frame) else 0
            self.starts = np.searchsorted(self.frame, np.arange(n_f + 1))


class Capture:
    """Frames + rays + helpers.  Poses can be corrected in place (drift handling)."""

    def __init__(self, frames: List[Frame], rays: RaySet, meta: Optional[dict] = None):
        self.frames = frames
        self.rays = rays
        self.meta = meta or {}
        self.R = np.stack([f.R for f in frames]).astype(np.float64)
        self.org = np.stack([f.org for f in frames]).astype(np.float64)

    # ---- geometry ---------------------------------------------------------
    def world(self, fidx: Sequence[int], kinds=(KIND_HIT,), stride: int = 1):
        """Return dict with world dirs/points for the rays of the given frames."""
        rs = self.rays
        sel_all, dirs, pts, org, fr, rg, kd, sg = [], [], [], [], [], [], [], []
        for f in fidx:
            a, b = rs.starts[f], rs.starts[f + 1]
            if b <= a:
                continue
            idx = np.arange(a, b, stride)
            k = rs.kind[idx]
            m = np.isin(k, kinds)
            idx = idx[m]
            if idx.size == 0:
                continue
            d = rs.dir_cam[idx].astype(np.float64) @ self.R[f].T
            r = rs.rng[idx].astype(np.float64)
            dirs.append(d)
            pts.append(self.org[f][None, :] + d * r[:, None])
            org.append(np.broadcast_to(self.org[f], d.shape))
            fr.append(np.full(idx.size, f, np.int32))
            rg.append(r)
            kd.append(rs.kind[idx])
            sg.append(rs.sigma[idx])
            sel_all.append(idx)
        if not dirs:
            z3 = np.zeros((0, 3))
            return dict(dirs=z3, pts=z3, org=z3, frame=np.zeros(0, np.int32), rng=np.zeros(0),
                        kind=np.zeros(0, np.uint8), sigma=np.zeros(0), idx=np.zeros(0, np.int64))
        return dict(dirs=np.concatenate(dirs), pts=np.concatenate(pts), org=np.concatenate(org),
                    frame=np.concatenate(fr), rng=np.concatenate(rg), kind=np.concatenate(kd),
                    sigma=np.concatenate(sg), idx=np.concatenate(sel_all))

    def apply_correction(self, fidx: Sequence[int], theta: float, t: np.ndarray, c0: np.ndarray):
        """Rigid correction (yaw theta about gravity through c0, then translation t) on frames."""
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        for f in fidx:
            self.org[f] = Rz @ (self.org[f] - c0) + c0 + t
            self.R[f] = Rz @ self.R[f]

    @property
    def n_frames(self) -> int:
        return len(self.frames)
