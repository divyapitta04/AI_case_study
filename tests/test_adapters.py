"""Polycam raw-data adapter (built to Polycam's public documentation): repackaging the real ARKitScenes LiDAR capture as a
Polycam raw export must load to the same frames and rays as the native reader.  Still to verify on a real Polycam export."""
import glob
import json
import os
import shutil

import numpy as np


def test_polycam_adapter_matches_native(real_lidar, tmp_path):
    from floorscan.ingest.lidar import load_native, load_polycam
    cap_dir = real_lidar
    kf = tmp_path / "keyframes"
    for sub in ("cameras", "depth", "images"):
        (kf / sub).mkdir(parents=True)
    for i, fp in enumerate(sorted(glob.glob(os.path.join(cap_dir, "frames", "*.json")))):
        j = json.load(open(fp))
        T = np.array(j["camera_to_world"])
        it = j["intrinsics"]
        cam = dict(fx=it["fx"], fy=it["fy"], cx=it["cx"], cy=it["cy"], width=it["width"], height=it["height"])
        for r in range(3):
            for c in range(4):
                cam[f"t_{r}{c}"] = float(T[r, c])
        json.dump(cam, open(kf / "cameras" / f"{i:05d}.json", "w"))
        shutil.copy(os.path.join(cap_dir, j["depth_file"]), kf / "depth" / f"{i:05d}.png")
        shutil.copy(os.path.join(cap_dir, j["rgb_file"]), kf / "images" / f"{i:05d}.jpg")
    a, b = load_native(cap_dir), load_polycam(str(tmp_path))
    assert a.n_frames == b.n_frames
    assert np.allclose(a.org, b.org, atol=1e-6)
    assert len(a.rays.rng) == len(b.rays.rng)
