from __future__ import annotations

import glob
import json
import os


def detect_tier(path: str) -> str:
    """Pick the input tier from what is on disk (so one command works for any capture)."""
    if os.path.isfile(os.path.join(path, "capture.json")):
        try:
            t = json.load(open(os.path.join(path, "capture.json"))).get("tier")
            if t in ("lidar", "video", "photos"):
                return t
        except Exception:
            pass
    if os.path.isdir(os.path.join(path, "keyframes", "depth")) or glob.glob(os.path.join(path, "frames", "*.json")):
        return "lidar"
    vids = [f for f in os.listdir(path) if f.lower().endswith((".mp4", ".mov", ".m4v"))]
    if vids:
        return "video"
    subs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")]
    if any(glob.glob(os.path.join(path, d, "*.jp*g")) or glob.glob(os.path.join(path, d, "*.JP*G")) for d in subs):
        return "photos"
    raise ValueError(f"cannot detect capture tier in {path}")
