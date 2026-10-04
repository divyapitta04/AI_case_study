"""Confidence intervals.

Every reported measurement is  value +/- 1.96 * k * sqrt(sigma_fit^2 + floor^2 + (rel*value)^2)  where
  sigma_fit  analytic 1-sigma from the fit (histogram-peak scatter, chunk-to-chunk consistency, ...),
  floor      tier-specific systematic floor (m),
  rel        tier-specific relative (scale) error,
  k          calibration factor fitted on a *held-out* calibration set so the empirical coverage is ~95 %
             (see bench/calibrate.py, floorscan/calibration.json).
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

_CAL_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")

DEFAULTS = {
    "lidar": {
        "length": dict(k=1.0, floor=0.006, rel=0.003),
        "height": dict(k=1.0, floor=0.006, rel=0.003),
        "area": dict(k=1.0, floor=0.0, rel=0.006),
        "opening_width": dict(k=1.0, floor=0.008, rel=0.004),
        "opening_height": dict(k=1.0, floor=0.012, rel=0.004),
        "offset": dict(k=1.0, floor=0.01, rel=0.003),
        "extent": dict(k=1.0, floor=0.015, rel=0.03),
        "footprint": dict(k=1.0, floor=0.0, rel=0.01),
    },
    "video": {
        "length": dict(k=1.0, floor=0.015, rel=0.015),
        "height": dict(k=1.0, floor=0.015, rel=0.015),
        "area": dict(k=1.0, floor=0.0, rel=0.03),
        "opening_width": dict(k=1.0, floor=0.02, rel=0.02),
        "opening_height": dict(k=1.0, floor=0.03, rel=0.02),
        "offset": dict(k=1.0, floor=0.03, rel=0.015),
        "extent": dict(k=1.0, floor=0.03, rel=0.05),
        "footprint": dict(k=1.0, floor=0.0, rel=0.03),
    },
    "photos": {
        "length": dict(k=1.0, floor=0.05, rel=0.05),
        "height": dict(k=1.0, floor=0.05, rel=0.05),
        "area": dict(k=1.0, floor=0.0, rel=0.08),
        "opening_width": dict(k=1.0, floor=0.05, rel=0.06),
        "opening_height": dict(k=1.0, floor=0.06, rel=0.06),
        "offset": dict(k=1.0, floor=0.08, rel=0.05),
        "extent": dict(k=1.0, floor=0.05, rel=0.08),
        "footprint": dict(k=1.0, floor=0.0, rel=0.07),
    },
}


def load_calibration() -> Dict:
    cal = json.loads(json.dumps(DEFAULTS))
    if os.path.exists(_CAL_PATH) and not os.environ.get("FLOORSCAN_NO_CAL"):
        try:
            with open(_CAL_PATH) as fh:
                user = json.load(fh)
            for tier, d in user.items():
                if tier.startswith("_"):
                    continue
                for kind, p in d.items():
                    cal.setdefault(tier, {}).setdefault(kind, {}).update(p)
        except Exception:
            pass
    return cal


_CAL = None


def calibration() -> Dict:
    global _CAL
    if _CAL is None:
        _CAL = load_calibration()
    return _CAL


def sigma_total(value: float, sigma_fit: float, tier: str, kind: str, extra_rel: float = 0.0) -> float:
    p = calibration()[tier][kind]
    return float(p["k"] * np.sqrt(sigma_fit ** 2 + p["floor"] ** 2 + (p["rel"] * abs(value)) ** 2
                                 + (extra_rel * abs(value)) ** 2))


def measure(value: float, sigma_fit: float, tier: str, kind: str, extra_rel: float = 0.0,
            lo_clip: float = None) -> Dict:
    s = sigma_total(value, sigma_fit, tier, kind, extra_rel)
    lo, hi = value - 1.96 * s, value + 1.96 * s
    if lo_clip is not None:
        lo = max(lo, lo_clip)
    return dict(value=round(float(value), 4), ci95=[round(float(lo), 4), round(float(hi), 4)],
                sigma=round(float(s), 5))
