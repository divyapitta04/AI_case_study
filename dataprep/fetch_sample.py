"""Download the pre-registered benchmark sample (data/SPLIT.json) from Apple's ARKitScenes and convert it to our tiers.

    python dataprep/fetch_sample.py [--work ../arkitscenes_raw] [--visits 467293 ...] [--no-convert]

Per visit (room):
  * every session: LiDAR depth, low-res RGB, intrinsics, ARKit poses        -> data/<video>_lidar   (repeatability)
  * first session: the device video (.mov) and the hi-res frames           -> data/<video>_video, data/<video>_photos
  * every FARO laser station of the visit: 1/8 strided sample via HTTP range -> ground truth (dataprep/room_gt.py)
Raw files stay in --work (several GB per visit, ARKitScenes non-commercial research licence); only the converted captures
go to data/.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import subprocess
import sys
import http.client
import urllib.parse
import urllib.request
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
LIDAR_ASSETS = ["lowres_depth.zip", "lowres_wide.zip", "lowres_wide_intrinsics.zip", "lowres_wide.traj"]
FRAMES = [("wide.zip", "wide_intrinsics.zip"), ("vga_wide.zip", "vga_wide_intrinsics.zip")]  # hi-res frames, else 640x480


def get(url: str, dst: str):
    """Resumable download with retries and stall detection (curl), urllib as a fallback."""
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        subprocess.check_call(["curl", "-sSfL", "--retry", "8", "--retry-delay", "3", "--retry-all-errors",
                               "--speed-limit", "20000", "--speed-time", "60", "-C", "-", "-o", dst + ".part", url])
    except FileNotFoundError:
        urllib.request.urlretrieve(url, dst + ".part")
    os.replace(dst + ".part", dst)
    return dst


PLY_TYPES = {"double": "f8", "float": "f4", "uchar": "u1", "int": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}


class _Ranged:
    """Byte-range reader over one persistent HTTPS connection (hundreds of small requests per scan)."""

    def __init__(self, url: str):
        p = urllib.parse.urlsplit(url)
        self.host, self.path = p.netloc, p.path
        self.conn = http.client.HTTPSConnection(self.host, timeout=60)

    def get(self, a: int, b: int) -> bytes:
        for attempt in range(4):
            try:
                self.conn.request("GET", self.path, headers={"Range": f"bytes={a}-{b}"})
                r = self.conn.getresponse()
                data = r.read()
                if r.status in (200, 206):
                    return data
            except (http.client.HTTPException, OSError):
                self.conn.close()
                self.conn = http.client.HTTPSConnection(self.host, timeout=60)
        raise IOError(f"range request failed: {self.path} {a}-{b}")


def get_laser_sample(url: str, dst: str, every: int = 8, chunk_pts: int = 12_000):
    """Strided sample of a FARO scan via HTTP range requests: every `every`-th block of `chunk_pts` consecutive points
    (the scan stores points column by column, so this keeps ~1/every of the columns, spread over all directions).
    Saves xyz (venue frame, float64) + rgb as .npz.  A 2 GB scan -> ~1/8 of the bytes."""
    if os.path.exists(dst):
        return dst
    rd = _Ranged(url)
    _range = lambda a, b: rd.get(a, b)  # noqa: E731
    head = _range(0, 4095)
    end = head.index(b"end_header\n") + len(b"end_header\n")
    props, n = [], 0
    for line in head[:end].decode("ascii").splitlines():
        t = line.split()
        if t[:2] == ["element", "vertex"]:
            n = int(t[2])
        elif t[:1] == ["element"] and t[1] != "vertex":
            break
        elif t[:1] == ["property"] and n and t[1] != "list":
            props.append((t[2], "<" + PLY_TYPES[t[1]]))
    dt = np.dtype(props)
    xyz, rgb = [], []
    for k in range(0, n, chunk_pts * every):
        cnt = min(chunk_pts, n - k)
        buf = _range(end + k * dt.itemsize, end + (k + cnt) * dt.itemsize - 1)
        a = np.frombuffer(buf, dtype=dt, count=len(buf) // dt.itemsize)
        xyz.append(np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64))
        rgb.append(np.stack([a["red"], a["green"], a["blue"]], 1) if "red" in a.dtype.names else np.zeros((len(a), 3), np.uint8))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst + ".part.npz", xyz=np.concatenate(xyz), rgb=np.concatenate(rgb).astype(np.uint8), n_total=n, every=every)
    os.replace(dst + ".part.npz", dst)
    return dst


def exists(url: str) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            return r.status == 200
    except Exception:
        return False


def laser_ids(visit: str, work: str):
    path = get(f"{BASE}/laser_scanner_point_clouds/laser_scanner_point_clouds_mapping.csv",
               os.path.join(work, "laser", "mapping.csv"))
    return [r["laser_scanner_point_clouds_id"] for r in csv.DictReader(open(path)) if r["visit_id"] == visit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "..", "arkitscenes_raw"))
    ap.add_argument("--visits", nargs="*")
    ap.add_argument("--no-convert", action="store_true")
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    split = json.load(open(os.path.join(ROOT, "data", "SPLIT.json")))
    visits = {k: v for part in ("dev", "test") for k, v in split[part].items()}
    if a.visits:
        visits = {k: v for k, v in visits.items() if k in a.visits}
    jobs = []
    for visit, d in visits.items():
        for i, vid in enumerate(d["sessions"]):
            assets = list(LIDAR_ASSETS)
            if i == 0:
                fr = next((p for p in FRAMES if os.path.exists(os.path.join(work, vid, p[0])) or
                           exists(f"{BASE}/Validation/{vid}/{p[0]}")), FRAMES[-1])
                assets += list(fr) + [f"{vid}.mov"]
            jobs += [(get, f"{BASE}/Validation/{vid}/{f}", os.path.join(work, vid, f)) for f in assets]
        for lid in laser_ids(visit, work):
            u = f"{BASE}/laser_scanner_point_clouds/{visit}/{lid}"
            jobs.append((get, u + "_pose.txt", os.path.join(work, "laser", visit, lid + "_pose.txt")))
            jobs.append((get_laser_sample, u + ".ply", os.path.join(work, "laser", visit, lid + "_sample.npz")))
    print(f"{len(jobs)} files to fetch", flush=True)
    failed = []
    with cf.ThreadPoolExecutor(12) as ex:
        futs = {ex.submit(fn, u, p): u for fn, u, p in jobs}
        for f in cf.as_completed(futs):
            try:
                print("got", os.path.relpath(f.result(), work), flush=True)
            except Exception as e:  # report and continue; rerun the script to retry
                failed.append(futs[f])
                print("FAILED", futs[f], e, flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} downloads failed - rerun to retry")
    if a.no_convert:
        return
    out = os.path.join(ROOT, "data")
    conv = os.path.join(ROOT, "dataprep", "arkitscenes.py")
    for visit, d in visits.items():
        for i, vid in enumerate(d["sessions"]):
            vdir = os.path.join(work, vid)
            for z in os.listdir(vdir):
                if z.endswith(".zip") and not os.path.isdir(os.path.join(vdir, z[:-4])):
                    zipfile.ZipFile(os.path.join(vdir, z)).extractall(vdir)
            extra = ["--video", "--photos"] if i == 0 else []
            subprocess.check_call([sys.executable, conv, "--video_dir", vdir, "--video_id", vid, "--out", out,
                                   "--sky", d["sky"]] + extra)
    print("done", flush=True)


if __name__ == "__main__":
    main()
