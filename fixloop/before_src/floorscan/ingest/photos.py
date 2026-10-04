"""Photo-tier ingest: per-room folders of stills (any iPhone 15+), EXIF intrinsics, monocular metric depth.

Depth: Depth-Anything-V2-Metric-Indoor-Small (HuggingFace transformers; weights fetched by scripts/fetch_models.py),
a pretrained metric monocular depth network, run on CPU.  The same backend serves the video tier.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

DIAG35 = 43.27
DEFAULT_F35 = 26.0  # iPhone 15 main camera


@dataclass
class Photo:
    room: str
    path: str
    name: str
    width: int
    height: int
    f35: float
    f35_from_exif: bool
    t: float
    is_door: bool = False
    link: Optional[str] = None

    @property
    def f_px(self) -> float:
        return self.f35 * float(np.hypot(self.width, self.height)) / DIAG35

    @property
    def K(self) -> np.ndarray:
        f = self.f_px
        return np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1.0]])

    def load(self) -> np.ndarray:
        im = Image.open(self.path)
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


_ORD = ["IMG", "PXL"]


def _exif(path: str):
    try:
        im = Image.open(path)
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
        f35 = sub.get(0xA405)
        dt = sub.get(0x9003) or ex.get(0x0132)
        return im.size, ex.get(0x0112, 1), (float(f35) if f35 else None), dt
    except Exception:
        return None, 1, None, None


def _parse_dt(dt: Optional[str], fallback: float) -> float:
    if not dt:
        return fallback
    m = re.match(r"(\d+):(\d+):(\d+) (\d+):(\d+):(\d+)", str(dt))
    if not m:
        return fallback
    y, mo, d, h, mi, s = map(int, m.groups())
    return ((((y * 12 + mo) * 31 + d) * 24 + h) * 60 + mi) * 60 + s


def list_photos(root: str) -> Dict[str, List[Photo]]:
    rooms: Dict[str, List[Photo]] = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p) or d.startswith("."):
            continue
        files = sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg")))
        ph: List[Photo] = []
        for i, f in enumerate(files):
            size, orient, f35, dt = _exif(f)
            if size is None:
                continue
            w, h = size
            if orient in (5, 6, 7, 8):
                w, h = h, w
            name = os.path.basename(f)
            m = re.search(r"_to_([A-Za-z0-9\-\s]+)\.", name)
            ph.append(Photo(room=d, path=f, name=name, width=w, height=h, f35=f35 or DEFAULT_F35,
                            f35_from_exif=f35 is not None, t=_parse_dt(dt, float(i)),
                            is_door=bool(m), link=m.group(1) if m else None))
        ph.sort(key=lambda x: (x.t, x.name))
        if ph:
            rooms[d] = ph
    return rooms


# ---------------------------------------------------------------------------------------------
class DepthAnythingV2:
    name = "da_v2"
    MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

    def __init__(self):
        self._pipe = None

    def available(self, ph: Photo) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    def __call__(self, ph: Photo, img: np.ndarray) -> np.ndarray:
        from transformers import pipeline
        if self._pipe is None:
            self._pipe = pipeline(task="depth-estimation", model=self.MODEL, device=-1)
        h, w = img.shape[:2]
        s = 518.0 / max(h, w)
        small = Image.fromarray(img).resize((int(w * s), int(h * s)))
        out = self._pipe(small)
        d = out["predicted_depth"]
        d = d.squeeze().cpu().numpy() if hasattr(d, "cpu") else np.asarray(d).squeeze()
        # model output is already metric depth (m); resample to a modest grid for speed
        d = cv2.resize(d.astype(np.float32), (160, int(160 * h / w)), interpolation=cv2.INTER_AREA)
        return d


# ---------------------------------------------------------------------------------------------
def detect_marker(img: np.ndarray, K: np.ndarray, size_m: float = 0.10, ident: Optional[int] = None):
    """ArUco 4x4 marker -> list of dict(center_cam, z, id).  Uses PnP for metric distance."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    out = []
    if ids is None:
        return out
    L = size_m
    obj = np.array([[-L / 2, L / 2, 0], [L / 2, L / 2, 0], [L / 2, -L / 2, 0], [-L / 2, -L / 2, 0]], np.float32)
    for c, i in zip(corners, ids.ravel()):
        if ident is not None and int(i) != ident:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float32), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        side_px = float(np.mean([np.linalg.norm(c[0][k] - c[0][(k + 1) % 4]) for k in range(4)]))
        tv = tvec.ravel()
        out.append(dict(center_cam=tv, z=float(tv[2]), id=int(i), side_px=side_px,
                        px=c.reshape(4, 2).mean(0)))
    return out
