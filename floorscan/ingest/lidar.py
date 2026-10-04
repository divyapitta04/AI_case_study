"""LiDAR-tier ingest: depth + ARKit poses + intrinsics -> Capture.

Supported on-disk layouts
  * ``floorscan-lidar-v1``  (capture.json + frames/*.json + depth/*.png (16-bit mm) + rgb/*.jpg)
  * ``polycam-raw``         (keyframes/{cameras,depth,images}; Polycam "raw data" export, built to
                            Polycam's public documentation - verify on a real export)
"""
from __future__ import annotations

import glob
import json
import os
from typing import List

import cv2
import numpy as np

from ..frames import KIND_DROP, KIND_HIT, KIND_IGNORE, Capture, Frame, RaySet

# ARKit world (Y up) -> our world (Z up):  (x, y, z) -> (x, -z, y)
_M_INV = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
_C = np.diag([1, -1, -1.0])  # ARKit camera (x right, y up, -z fwd) -> CV camera
STRIDE = 2

MAX_RANGE = 5.0


def arkit_pose_to_cv(T: np.ndarray):
    R = _M_INV @ T[:3, :3] @ _C
    return R, _M_INV @ T[:3, 3]


def flying_pixel_mask(D: np.ndarray) -> np.ndarray:
    """True where a depth pixel is a likely mixed/flying pixel (large 2nd difference)."""
    d = D.astype(np.float32)
    valid = d > 0
    bad = np.zeros(d.shape, bool)
    thr = 0.05 + 0.03 * d
    for ax in (0, 1):
        a = np.roll(d, 1, ax)
        b = np.roll(d, -1, ax)
        va = np.roll(valid, 1, ax) & np.roll(valid, -1, ax) & valid
        sec = np.abs(a + b - 2 * d)
        bad |= va & (sec > thr)
    return bad


def _rays_from_depth(D: np.ndarray, K: np.ndarray, frame_idx: int):
    dh, dw = D.shape
    us = np.arange(0, dw, STRIDE)
    vs = np.arange(0, dh, STRIDE)
    uu, vv = np.meshgrid(us, vs)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dc = np.stack([(uu + 0.5 - cx) / fx, (vv + 0.5 - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
    nrm = np.linalg.norm(dc, axis=1)
    dc = (dc / nrm[:, None]).astype(np.float32)
    z = D[vv, uu].reshape(-1).astype(np.float64)
    fly = flying_pixel_mask(D)[vv, uu].reshape(-1)
    rng = (z * nrm).astype(np.float32)
    kind = np.where(z > 0, KIND_HIT, KIND_DROP).astype(np.uint8)
    kind[(z > MAX_RANGE) | fly] = KIND_IGNORE
    rng[kind != KIND_HIT] = 0
    sig = (0.004 + 0.004 * z).astype(np.float32)
    px = np.stack([uu.reshape(-1), vv.reshape(-1)], 1).astype(np.int16)
    fr = np.full(dc.shape[0], frame_idx, np.int32)
    return dc, rng, kind, fr, px, sig


def _lazy_rgb(path):
    def f():
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return f


def _lazy_depth(path, K, w, h):
    def f():
        D = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        Kd = K.copy()
        Kd[0, :] *= D.shape[1] / w
        Kd[1, :] *= D.shape[0] / h
        return D, Kd
    return f


def load_native(path: str, max_frames: int = 0) -> Capture:
    files = sorted(glob.glob(os.path.join(path, "frames", "*.json")))
    if max_frames:
        files = files[:max_frames]
    frames: List[Frame] = []
    parts = []
    for i, fp in enumerate(files):
        j = json.load(open(fp))
        T = np.array(j["camera_to_world"], float)
        R, org = arkit_pose_to_cv(T)
        it = j["intrinsics"]
        K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1.0]])
        D = cv2.imread(os.path.join(path, j["depth_file"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        dh, dw = D.shape
        Kd = K.copy()
        Kd[0, :] *= dw / it["width"]
        Kd[1, :] *= dh / it["height"]
        frames.append(Frame(i, R, org, K, (it["width"], it["height"]),
                            rgb=_lazy_rgb(os.path.join(path, j["rgb_file"])), name=os.path.basename(fp),
                            depth=_lazy_depth(os.path.join(path, j["depth_file"]), K, it["width"], it["height"])))
        parts.append(_rays_from_depth(D, Kd, i))
    rays = RaySet(*(np.concatenate([p[k] for p in parts]) for k in range(6)))
    return Capture(frames, rays, dict(tier="lidar", source="floorscan-lidar-v1", depth_size=(dw, dh),
                                      px_angle=float(1.0 / Kd[0, 0])))


def load_polycam(path: str, max_frames: int = 0) -> Capture:
    """Polycam raw-data export. Camera JSON: fx,fy,cx,cy,width,height and t_ij (3x4 row-major,
    ARKit camera-to-world, +Y up).  Depth: 16-bit PNG in millimetres."""
    kf = os.path.join(path, "keyframes")
    cams = sorted(glob.glob(os.path.join(kf, "cameras", "*.json")))
    if max_frames:
        cams = cams[:max_frames]
    frames, parts = [], []
    for i, cp in enumerate(cams):
        stem = os.path.splitext(os.path.basename(cp))[0]
        j = json.load(open(cp))
        T = np.eye(4)
        for r in range(3):
            for c in range(4):
                T[r, c] = j[f"t_{r}{c}"]
        R, org = arkit_pose_to_cv(T)
        K = np.array([[j["fx"], 0, j["cx"]], [0, j["fy"], j["cy"]], [0, 0, 1.0]])
        dp = os.path.join(kf, "depth", stem + ".png")
        if not os.path.exists(dp):
            continue
        D = cv2.imread(dp, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        dh, dw = D.shape[:2]
        Kd = K.copy()
        Kd[0, :] *= dw / j["width"]
        Kd[1, :] *= dh / j["height"]
        img = os.path.join(kf, "images", stem + ".jpg")
        frames.append(Frame(len(frames), R, org, K, (int(j["width"]), int(j["height"])), rgb=_lazy_rgb(img), name=stem,
                            depth=_lazy_depth(dp, K, int(j["width"]), int(j["height"]))))
        parts.append(_rays_from_depth(D, Kd, len(frames) - 1))
    rays = RaySet(*(np.concatenate([p[k] for p in parts]) for k in range(6)))
    return Capture(frames, rays, dict(tier="lidar", source="polycam-raw", depth_size=(dw, dh),
                                      px_angle=float(1.0 / Kd[0, 0])))
