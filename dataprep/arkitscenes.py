"""Convert a REAL capture from Apple's public ARKitScenes dataset into the three floorscan tier formats.

ARKitScenes (Apple, non-commercial research licence; https://github.com/apple/ARKitScenes) records iPad Pro sessions with
ARKit: 256x192 LiDAR depth (mm PNG), RGB, per-frame intrinsics (.pincam) and camera poses (.traj: world-to-camera,
axis-angle + translation, OpenCV camera axes), plus the raw .mov video and FARO laser scans of the same room.

    python dataprep/arkitscenes.py --video_dir <dir with extracted assets> --video_id 42444946 --out data \
        --sky Left [--video] [--photos]

Writes  <out>/<video_id>_lidar  (floorscan-lidar-v1),  <out>/<video_id>_video (mp4, upright),  <out>/<video_id>_photos (8 stills,
EXIF focal length) - i.e. the SAME real room at all three tiers.  No ground truth is touched here (see room_gt.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

C = np.diag([1.0, -1.0, -1.0])  # OpenCV camera <-> ARKit camera
DIAG35 = 43.27


def read_traj(path):
    out = {}
    for line in open(path):
        t = line.split()
        if len(t) != 7:
            continue
        R, _ = cv2.Rodrigues(np.array([float(x) for x in t[1:4]]))
        E = np.eye(4)
        E[:3, :3] = R
        E[:3, 3] = [float(x) for x in t[4:7]]
        out[float(t[0])] = np.linalg.inv(E)  # camera(OpenCV) -> world
    return out


def nearest(ts_list, t):
    i = int(np.searchsorted(ts_list, t))
    c = [j for j in (i - 1, i) if 0 <= j < len(ts_list)]
    j = min(c, key=lambda k: abs(ts_list[k] - t))
    return ts_list[j], abs(ts_list[j] - t)


def world_up(poses, depth_dir, intr_dir, n=40):
    """The dataset's world frame is gravity-aligned but its 'up' axis is not documented here: pick the world axis along
    which the back-projected point heights cluster most sharply (floor and ceiling planes)."""
    files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))[:: max(1, len(glob.glob(os.path.join(depth_dir, '*.png'))) // n)]
    tss = sorted(poses)
    pts = []
    for f in files:
        ts = float(os.path.basename(f)[:-4].split("_")[-1])
        tp, dt = nearest(tss, ts)
        if dt > 0.05:
            continue
        D = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        w, h, fx, fy, cx, cy = np.loadtxt(os.path.join(intr_dir, os.path.basename(f).replace(".png", ".pincam")))
        sx, sy = D.shape[1] / w, D.shape[0] / h
        vv, uu = np.mgrid[0:D.shape[0]:4, 0:D.shape[1]:4]
        z = D[vv, uu]
        m = z > 0.2
        X = np.stack([(uu[m] + 0.5 - cx * sx) / (fx * sx) * z[m], (vv[m] + 0.5 - cy * sy) / (fy * sy) * z[m], z[m]], 1)
        T = poses[tp]
        pts.append(X @ T[:3, :3].T + T[:3, 3])
    P = np.concatenate(pts)
    best = None
    for ax in range(3):
        for sg in (1, -1):
            h = sg * P[:, ax]
            hist, _ = np.histogram(h, bins=int((h.max() - h.min()) / 0.02) + 1)
            sc = float((hist.astype(float) ** 2).sum())
            # sign: cameras are above the floor -> most points below camera height on the floor side
            if best is None or sc > best[0]:
                best = (sc, ax, sg)
    ax = best[1]
    # decide the sign: the floor (dense, lowest strong peak) must be BELOW the median camera height
    cam_h = np.median([poses[t][ax, 3] for t in tss])
    below = np.mean(P[:, ax] < cam_h - 0.8)
    above = np.mean(P[:, ax] > cam_h + 0.6)
    sg = 1 if below >= above * 0.6 else -1
    up = np.zeros(3)
    up[ax] = sg
    return up


def to_yup(up: np.ndarray) -> np.ndarray:
    """Rotation taking the dataset world to an ARKit-style world (+Y up)."""
    y = up / np.linalg.norm(up)
    x = np.array([1.0, 0, 0]) if abs(y[0]) < 0.9 else np.array([0, 0, 1.0])
    x = x - y * (x @ y)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z], 0)


def convert_lidar(vdir, vid, out, stride=3):
    poses = read_traj(os.path.join(vdir, "lowres_wide.traj"))
    ddir, rdir, idir = (os.path.join(vdir, d) for d in ("lowres_depth", "lowres_wide", "lowres_wide_intrinsics"))
    up = world_up(poses, ddir, idir)
    A = to_yup(up)
    dst = os.path.join(out, f"{vid}_lidar")
    shutil.rmtree(dst, ignore_errors=True)
    for d in ("frames", "depth", "rgb"):
        os.makedirs(os.path.join(dst, d))
    tss = sorted(poses)
    dfiles = sorted(glob.glob(os.path.join(ddir, "*.png")))
    dts = [float(os.path.basename(f)[:-4].split("_")[-1]) for f in dfiles]
    order = np.argsort(dts)
    dts_sorted = [dts[i] for i in order]
    n = 0
    for tp in tss[::stride]:  # poses come at ~10 Hz, depth at 60 Hz: one key-frame every `stride` poses (~3.3 Hz)
        ts, dt = nearest(dts_sorted, tp)
        f = dfiles[order[dts_sorted.index(ts)]]
        stem = os.path.basename(f)[:-4]
        rgbf = os.path.join(rdir, stem + ".png")
        inf = os.path.join(idir, stem + ".pincam")
        if dt > 0.05 or not os.path.exists(rgbf) or not os.path.exists(inf):
            continue
        w, h, fx, fy, cx, cy = np.loadtxt(inf)
        T = poses[tp]
        Tar = np.eye(4)
        Tar[:3, :3] = A @ T[:3, :3] @ C
        Tar[:3, 3] = A @ T[:3, 3]
        shutil.copy(f, os.path.join(dst, "depth", f"{n:05d}.png"))
        cv2.imwrite(os.path.join(dst, "rgb", f"{n:05d}.jpg"), cv2.imread(rgbf), [cv2.IMWRITE_JPEG_QUALITY, 92])
        json.dump(dict(index=n, timestamp=ts, camera_to_world=Tar.tolist(),
                       intrinsics=dict(fx=fx, fy=fy, cx=cx, cy=cy, width=int(w), height=int(h)),
                       depth_file=f"depth/{n:05d}.png", rgb_file=f"rgb/{n:05d}.jpg"),
                  open(os.path.join(dst, "frames", f"{n:05d}.json"), "w"))
        n += 1
    json.dump(dict(format="floorscan-lidar-v1", tier="lidar", device=dict(model="iPad Pro (ARKitScenes, real)"),
                   source=f"ARKitScenes video {vid}", frames=n, world_up_in_dataset=up.tolist()),
              open(os.path.join(dst, "capture.json"), "w"), indent=1)
    print("lidar frames", n, "up axis", up)
    return dst


ROT = {"Up": None, "Left": cv2.ROTATE_90_CLOCKWISE, "Right": cv2.ROTATE_90_COUNTERCLOCKWISE, "Down": cv2.ROTATE_180}


def convert_video(vdir, vid, out, sky, fps_out=10.0, width=960):
    pc = sorted(glob.glob(os.path.join(vdir, "lowres_wide_intrinsics", "*.pincam")))[0]
    w0, h0, fx0 = np.loadtxt(pc)[:3]
    f35_hint = float(fx0 * DIAG35 / np.hypot(w0, h0))  # this dataset's video is the 1x wide camera, not the 0.5x lens
    src = os.path.join(vdir, f"{vid}.mov")
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps / fps_out)))
    dst = os.path.join(out, f"{vid}_video")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    vw, i, n = None, 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            if ROT[sky] is not None:
                fr = cv2.rotate(fr, ROT[sky])
            h, w = fr.shape[:2]
            s = width / max(w, h)
            fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            if vw is None:
                vw = cv2.VideoWriter(os.path.join(dst, "walkthrough.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / step,
                                     (fr.shape[1], fr.shape[0]))
            vw.write(fr)
            n += 1
        i += 1
    vw.release()
    json.dump(dict(format="floorscan-video-v1", tier="video", device=dict(model="iPad Pro (ARKitScenes, real)"),
                   fps=fps / step, video="walkthrough.mp4", frames=n, lens_f35_hint=round(f35_hint, 2),
                   note="upright-rotated from the dataset .mov; no scale card in this real capture"),
              open(os.path.join(dst, "capture.json"), "w"), indent=1)
    print("video frames", n, "fps", fps / step)
    return dst


def convert_photos(vdir, vid, out, sky, k=8, radius=0.6):
    """Approximate the photo protocol from a walking capture: pick up to k hi-res frames whose camera centres lie within
    `radius` of one point and whose viewing directions are as spread out as possible."""
    poses = read_traj(os.path.join(vdir, "lowres_wide.traj"))
    tss = sorted(poses)
    src = "wide" if os.path.isdir(os.path.join(vdir, "wide")) else "vga_wide"  # 1920x1440 frames, else 640x480
    wdir = os.path.join(vdir, src)
    idir = os.path.join(vdir, src + "_intrinsics")
    files = sorted(glob.glob(os.path.join(wdir, "*.png")))
    cand = []
    for f in files:
        stem = os.path.basename(f)[:-4]
        ts = float(stem.split("_")[-1])
        tp, dt = nearest(tss, ts)
        if dt > 0.05:
            continue
        T = poses[tp]
        cand.append((f, T[:3, 3], T[:3, 2], stem))
    up = world_up(poses, os.path.join(vdir, "lowres_depth"), os.path.join(vdir, "lowres_wide_intrinsics"))
    ax = int(np.argmax(np.abs(up)))
    hor = [i for i in range(3) if i != ax]
    C2 = np.array([c[1][hor] for c in cand])
    # horizontal basis (e1, e2) with e1 x e2 = up: azimuth atan2(d.e2, d.e1) grows counter-clockwise seen from above
    e1 = np.zeros(3)
    e1[hor[0]] = 1.0
    e2 = np.cross(up, e1)

    def azimuth(dvec):
        return float(np.arctan2(dvec @ e2, dvec @ e1))
    best_set, best_cov = None, -1
    for i in range(0, len(cand), max(1, len(cand) // 60)):
        near = [j for j in range(len(cand)) if np.linalg.norm(C2[j] - C2[i]) < radius]
        if len(near) < k:
            continue
        az = {j: azimuth(cand[j][2]) for j in near}
        chosen = [near[0]]
        while len(chosen) < k:
            nxt = max((j for j in near if j not in chosen),
                      key=lambda j: min(abs((az[j] - az[c] + np.pi) % (2 * np.pi) - np.pi) for c in chosen))
            chosen.append(nxt)
        cov = sum(min(abs((az[j] - az[c] + np.pi) % (2 * np.pi) - np.pi) for c in chosen if c != j) for j in chosen)
        if cov > best_cov:
            best_set, best_cov = chosen, cov
    if best_set is None:
        print("no standing spot with enough frames - photo tier skipped")
        return None
    best_set.sort(key=lambda j: -azimuth(cand[j][2]))  # protocol: turn clockwise on the spot (photo order = capture order)
    dst = os.path.join(out, f"{vid}_photos")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(os.path.join(dst, "room"))
    for n, j in enumerate(best_set):
        f, _, _, stem = cand[j]
        im = cv2.imread(f)
        w0, h0, fx, fy, cx, cy = np.loadtxt(os.path.join(idir, stem + ".pincam"))
        if ROT[sky] is not None:
            im = cv2.rotate(im, ROT[sky])
        h, w = im.shape[:2]
        f35 = fx * DIAG35 / np.hypot(w0, h0)
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ex = Image.Exif()
        ex[0x010F] = "Apple"
        ex[0x0110] = "iPad Pro (ARKitScenes)"
        sub = ex.get_ifd(0x8769)
        sub[0xA405] = int(round(f35))
        sub[0x9003] = "2021:01:01 10:%02d:%02d" % (n // 60, n % 60)
        pil.save(os.path.join(dst, "room", f"IMG_{n:04d}.jpg"), quality=92, exif=ex.tobytes())
    json.dump(dict(format="floorscan-photos-v1", tier="photos", device=dict(model="iPad Pro (ARKitScenes, real)"),
                   rooms=["room"], frames_source=src,
                   note="8 frames picked from a walking capture near one standing spot (approximates the protocol)"),
              open(os.path.join(dst, "capture.json"), "w"), indent=1)
    print("photos", len(best_set), "f35", round(f35, 1))
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sky", default="Up")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--photos", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    convert_lidar(a.video_dir, a.video_id, a.out)
    if a.video:
        convert_video(a.video_dir, a.video_id, a.out, a.sky)
    if a.photos:
        convert_photos(a.video_dir, a.video_id, a.out, a.sky)
