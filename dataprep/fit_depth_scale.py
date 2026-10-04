"""Fit the focal-length correction of the depth network on the DEV rooms (never the TEST rooms).

    python dataprep/fit_depth_scale.py      # -> floorscan/calibration.json "_depth_net": {f0, sigma, n, ...}

For every DEV photo and video capture the pipeline is run with the correction disabled (FLOORSCAN_NO_NETCAL=1); the
lengths it would have reported with the depth network as the only scale source are compared with the laser ground truth
(walls + ceiling, median ratio per capture).  A metric monocular network reports depth x f0 / f for a lens of normalised
focal f (focal / longer image side), so f0 = f / ratio; f0 is the median over captures and sigma the relative spread
(1.4826 x MAD of the log ratios, at least 3 %).  Runs go to results/dev_fit/.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["FLOORSCAN_NO_NETCAL"] = "1"
os.environ.setdefault("FLOORSCAN_CACHE", os.path.join(ROOT, "bench", "cache"))


def f_norm_of(path: str, out: dict) -> float:
    cj = json.load(open(os.path.join(path, "capture.json")))
    if cj["tier"] == "video":
        import cv2
        from floorscan.ingest.photos import DIAG35
        cap = cv2.VideoCapture(os.path.join(path, cj["video"]))
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return float(cj["lens_f35_hint"]) * float(np.hypot(w, h)) / DIAG35 / max(w, h)
    from floorscan.ingest.photos import list_photos
    ph = next(iter(list_photos(path).values()))[0]
    return ph.f_px / max(ph.width, ph.height)


def main():
    from bench.metrics import evaluate
    from bench.run_bench import gt_of
    from floorscan.cli import run
    split = json.load(open(os.path.join(ROOT, "data", "SPLIT.json")))
    rows = []
    for visit, d in split["dev"].items():
        excluded = set(split.get("excluded_sessions", {}))
        s0 = next(v for v in d["sessions"] if v not in excluded)  # photo/video sets come from the first valid session
        for tier in ("photos", "video"):
            name = f"{s0}_{tier}"
            path = os.path.join(ROOT, "data", name)
            if not os.path.exists(os.path.join(path, "capture.json")):
                continue
            out = run(path, os.path.join(ROOT, "results", "dev_fit", name), quiet=True)
            gt = gt_of(name)
            m = evaluate(out, gt)
            sc = out["quality"].get("scale", {})
            k = sc.get("scale_net_over_final") if tier == "video" else next(iter(sc.values()), {}).get("scale_net_over_final")
            if not k or not m["rooms"]:
                continue
            g = gt["rooms"][0]
            ratios = [g["walls"][wl] / (g["walls"][wl] + e) / k for wl, e in m["rooms"][0]["wall_err"].items()]
            ratios.append(g["ceiling_height"] / (g["ceiling_height"] + m["rooms"][0]["height_err"]) / k)
            ratio = float(np.median(ratios))
            fn = f_norm_of(path, out)
            rows.append(dict(capture=name, f_norm=fn, ratio_true_over_net=ratio, f0=fn / ratio))
            print(rows[-1], flush=True)
    lr = np.log([r["ratio_true_over_net"] for r in rows])
    f0 = float(np.median([r["f0"] for r in rows]))
    sigma = max(0.03, float(1.4826 * np.median(np.abs(lr - np.median(lr)))))
    cal_path = os.path.join(ROOT, "floorscan", "calibration.json")
    cal = json.load(open(cal_path))
    cal["_depth_net"] = dict(f0=round(f0, 4), sigma=round(sigma, 4), n=len(rows), captures=rows,
                             note="fitted by dataprep/fit_depth_scale.py on DEV photo/video captures only")
    json.dump(cal, open(cal_path, "w"), indent=1)
    print(json.dumps(dict(f0=f0, sigma=sigma, n=len(rows))))


if __name__ == "__main__":
    main()
