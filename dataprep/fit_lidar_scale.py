"""Fit the LiDAR floor-to-ceiling (height) correction on the DEV sessions (never the TEST sessions).

    python dataprep/fit_lidar_scale.py      # -> floorscan/calibration.json "_lidar_height": {scale, n, spread, sessions}

For every DEV LiDAR session the floor-to-ceiling distance is read from its raw depth (lowest / highest strong horizontal
plane, the same rule as the laser ground truth) and compared with the laser room's.  scale = median(laser / session).
Sessions whose ceiling was never seen (more than 0.2 m short) and excluded sessions (data/SPLIT.json) are skipped.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["FLOORSCAN_NO_DEPTHCAL"] = "1"


def session_height(capture: str) -> float:
    from dataprep.room_gt import extreme_plane
    from floorscan.ingest.lidar import load_native
    cap = load_native(capture)
    z = cap.world(range(cap.n_frames), stride=6)["pts"][:, 2]
    zf = extreme_plane(z, np.percentile(z, 0.2) - 0.1, np.percentile(z, 0.2) + 1.0, "low")
    zc = extreme_plane(z, zf + 1.5, np.percentile(z, 99.8) + 0.1, "high")
    return float(zc - zf)


def main():
    split = json.load(open(os.path.join(ROOT, "data", "SPLIT.json")))
    excluded = set(split.get("excluded_sessions", {}))
    idx = json.load(open(os.path.join(ROOT, "data", "gt", "index.json")))
    rows = []
    for visit, d in split["dev"].items():
        for vid in d["sessions"]:
            if vid in excluded or vid not in idx:
                continue
            H_las = json.load(open(os.path.join(ROOT, "data", "gt", idx[vid])))["rooms"][0]["ceiling_height"]
            H = session_height(os.path.join(ROOT, "data", f"{vid}_lidar"))
            used = H > H_las - 0.2
            rows.append(dict(session=vid, laser=round(H_las, 4), raw=round(H, 4), ratio=round(H_las / H, 5), used=used))
            print(rows[-1], flush=True)
    r = np.array([x["ratio"] for x in rows if x["used"]])
    cal_path = os.path.join(ROOT, "floorscan", "calibration.json")
    cal = json.load(open(cal_path))
    cal["_lidar_height"] = dict(scale=round(float(np.median(r)), 4), spread=round(float(r.std()), 4), n=int(len(r)), sessions=rows,
                               note="fitted by dataprep/fit_lidar_scale.py on DEV sessions only")
    json.dump(cal, open(cal_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in cal["_lidar_height"].items() if k != "sessions"}))


if __name__ == "__main__":
    main()
