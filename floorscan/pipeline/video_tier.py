"""Video tier: a handheld walkthrough clip (no depth, no poses) -> the same plan contract.

  frames -> COLMAP structure-from-motion (poses, sparse points)            [pycolmap, disclosed dependency]
         -> metric monocular depth per key-frame, aligned per frame to the SfM points (removes the per-image
            scale error of the depth model; all frames then share one consistent scale)
         -> gravity from the floor/ceiling height-histogram sharpness
         -> metric scale: printed ArUco card (1 %) > standard door height (~3 %) > ceiling-height prior (~5 %)
         -> the same room segmentation / plane-anchored refinement / door pose graph as the LiDAR tier.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import Config
from ..frames import KIND_HIT, Capture, Frame, RaySet
from ..ingest import photos as ph_mod
from .run import run_global

KEYFRAME_HZ = 3.3  # key-frames per second of video (2 Hz registered only 73 % of a real clip, 3.3 Hz 96 %)
MAX_W = 960
PRIOR_CEIL = 2.5
CEIL_PRIOR_SIGMA = 0.10  # real ceilings span ~2.3-3.2 m (a real ARKitScenes room is 3.07 m)
NET_SCALE_SIGMA = 0.08  # relative scale error of the metric depth network, median over key-frames (dev rooms)
DOOR_H = 2.04
DOOR_SIGMA = 0.04


def find_video(path: str) -> str:
    for f in sorted(os.listdir(path)):
        if f.lower().endswith((".mp4", ".mov", ".m4v")):
            return os.path.join(path, f)
    raise FileNotFoundError("no video file in capture folder")


def _informative(fr: np.ndarray) -> bool:
    """False for frames that cannot be registered: blank wall at arm's length, motion-blurred, lens covered."""
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6)).apply(g)
    kp = cv2.ORB_create(nfeatures=400, fastThreshold=8).detect(g, None)
    return len(kp) >= 40 and cv2.Laplacian(g, cv2.CV_64F).var() > 20


def extract_keyframes(video: str, out_dir: str, hz: float = KEYFRAME_HZ):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / hz)))
    os.makedirs(out_dir, exist_ok=True)
    keys, i, dropped = [], 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = fr.shape[:2]
            if w > MAX_W:
                fr = cv2.resize(fr, (MAX_W, int(round(h * MAX_W / w))), interpolation=cv2.INTER_AREA)
            if _informative(fr):
                name = f"{i:06d}.jpg"
                cv2.imwrite(os.path.join(out_dir, name), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                keys.append((i, name))
            else:
                dropped += 1
        i += 1
    extract_keyframes.dropped = dropped
    return keys, fps, stride


def run_sfm(work: str, f35: float = 13.0, wh: Tuple[int, int] = (640, 360), overlap: int = 24):
    import pycolmap
    try:
        pycolmap.logging.minloglevel = 3
        pycolmap.logging.alsologtostderr = False
    except Exception:
        pass
    pycolmap.set_random_seed(0)
    db = os.path.join(work, "db.db")
    if os.path.exists(db):
        os.remove(db)
    ropt = pycolmap.ImageReaderOptions()
    ropt.camera_model = "SIMPLE_RADIAL"
    f0 = f35 * float(np.hypot(*wh)) / ph_mod.DIAG35
    ropt.camera_params = f"{f0},{wh[0] / 2},{wh[1] / 2},0"
    eopt = pycolmap.FeatureExtractionOptions()
    try:
        eopt.num_threads = 1
        eopt.sift.peak_threshold = 0.004  # painted walls are low-contrast
        eopt.sift.max_num_features = 2048  # keeps exhaustive matching affordable on CPU
    except Exception:
        pass
    pycolmap.extract_features(db, os.path.join(work, "images"), camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ropt, extraction_options=eopt, device=pycolmap.Device.cpu)
    n_img = len(os.listdir(os.path.join(work, "images")))
    if n_img <= 300:
        # every pair: a room walkthrough revisits the same walls, and these loop-closing pairs are what keeps COLMAP in one
        # model (real ARKitScenes clip: 244/255 key-frames registered vs 152 with sequential matching)
        pycolmap.match_exhaustive(db)
    else:
        pycolmap.match_sequential(db, pairing_options=pycolmap.SequentialPairingOptions(overlap=overlap,
                                                                                       loop_detection=False))
    sp = os.path.join(work, "sparse")
    shutil.rmtree(sp, ignore_errors=True)
    os.makedirs(sp)
    opts = pycolmap.IncrementalPipelineOptions()
    try:
        opts.num_threads = 1
        opts.mapper.abs_pose_min_num_inliers = 12
        opts.mapper.abs_pose_min_inlier_ratio = 0.12
    except Exception:
        pass
    maps = pycolmap.incremental_mapping(db, os.path.join(work, "images"), sp, opts)
    if not maps:
        raise RuntimeError("structure-from-motion failed (no model): too little texture/parallax in the clip")
    best = max(maps.values(), key=lambda m: m.num_reg_images())
    return best


def _bilinear(depth: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(depth.astype(np.float32), x.astype(np.float32).reshape(-1, 1), y.astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1)


def estimate_up(pts: np.ndarray, hint: np.ndarray) -> np.ndarray:
    """Vertical direction = the direction along which the point heights are most sharply clustered
    (floor and ceiling planes), searched in a cone around the cameras' mean 'up' vector."""
    if len(pts) > 120000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 120000, replace=False)]
    dist = np.linalg.norm(pts - np.median(pts, 0), axis=1)
    bw = 0.012 * float(np.median(dist))
    a = np.cross(hint, [1.0, 0, 0] if abs(hint[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(hint, a)

    def vec(pol, az):
        return np.cos(pol) * hint + np.sin(pol) * (np.cos(az) * a + np.sin(az) * b)

    def score(u):
        h = pts @ u
        hist, _ = np.histogram(h, bins=int(np.ceil((h.max() - h.min()) / bw)) + 1)
        return float((hist.astype(np.float64) ** 2).sum())

    best = (score(hint), 0.0, 0.0)
    for pol in np.deg2rad(np.arange(2, 31, 3.0)):
        for az in np.deg2rad(np.arange(0, 360, 30.0)):
            sc = score(vec(pol, az))
            if sc > best[0]:
                best = (sc, pol, az)
    sc, pol, az = best
    for step in (np.deg2rad(1.0), np.deg2rad(0.3)):
        imp = True
        while imp:
            imp = False
            for dp, da in [(step, 0), (-step, 0), (0, step * 6), (0, -step * 6)]:
                p2, a2 = max(pol + dp, 0.0), az + da
                s2 = score(vec(p2, a2))
                if s2 > sc:
                    pol, az, sc, imp = p2, a2, s2, True
    return vec(pol, az)


def scale_capture(cap: Capture, f: float):
    cap.org *= f
    cap.rays.rng *= np.float32(f)
    cap.rays.sigma *= np.float32(f)
    for k, fr in enumerate(cap.frames):
        fr.org = cap.org[k].copy()


def _load_best_model(work: str):
    import pycolmap
    sp = os.path.join(work, "sparse")
    best = None
    for m in sorted(os.listdir(sp)):
        r = pycolmap.Reconstruction(os.path.join(sp, m))
        if best is None or r.num_reg_images() > best.num_reg_images():
            best = r
    return best


def run_video(path: str, cfg: Config):
    """SfM results can be cached (FLOORSCAN_CACHE=<dir>): the cache holds the key-frames, the COLMAP database and the
    sparse models, so repeated runs (benchmark ablations, the fix loop) replay *identical* poses.  Without the
    environment variable everything is computed live in a temporary folder."""
    T: Dict[str, float] = {}
    t0 = time.time()
    video = find_video(path)
    cache = os.environ.get("FLOORSCAN_CACHE")
    if cache:
        work = os.path.join(cache, "sfm_" + os.path.basename(os.path.normpath(path)))
        os.makedirs(work, exist_ok=True)
    else:
        work = tempfile.mkdtemp(prefix="floorscan_sfm_")
    try:
        done = os.path.join(work, "done.json")
        if cache and os.path.exists(done):
            import json as _json
            meta = _json.load(open(done))
            keys = [tuple(k) for k in meta["keys"]]
            extract_keyframes.dropped = meta.get("dropped", 0)
            im0 = cv2.imread(os.path.join(work, "images", keys[0][1]))
            H0, W0 = im0.shape[:2]
            rec = _load_best_model(work)
            T["decode"] = T["sfm"] = 0.0
        else:
            keys, fps, stride = extract_keyframes(video, os.path.join(work, "images"))
            T["decode"] = time.time() - t0
            t0 = time.time()
            im0 = cv2.imread(os.path.join(work, "images", keys[0][1]))
            H0, W0 = im0.shape[:2]
            f35 = 13.0  # protocol: 0.5x ultra-wide lens
            try:
                import json as _json
                hint = _json.load(open(os.path.join(path, "capture.json"))).get("lens_f35_hint")
                f35 = float(hint) if hint else f35
            except Exception:
                pass
            rec = run_sfm(work, f35=f35, wh=(W0, H0))
            T["sfm"] = time.time() - t0
            if cache:
                import json as _json
                _json.dump(dict(keys=keys, dropped=getattr(extract_keyframes, "dropped", 0)), open(done, "w"))
        t0 = time.time()
        cap, info = _build_capture(path, work, rec, keys, cfg, (W0, H0))
        T["depth_fusion"] = time.time() - t0
        scale0 = info["scale"]
        sig_rel = info["scale_sigma"]
        s_total, src = scale0, ("card" if info["card"] else "depth network + ceiling prior")
        out, r, places, links = run_global(cap, cfg, "video", "video/colmap+" + info["depth_backend"],
                                           device="iPhone 15+ (video)", extra=dict(frames=len(keys)),
                                           scale_rel=sig_rel)
        if not info["card"]:
            ks = _door_scale(r.rooms)
            if ks:
                w1 = 1 / sig_rel ** 2
                w2 = 1 / (DOOR_SIGMA ** 2 + 0.02 ** 2)
                s_f = (w1 * 1.0 + w2 * ks) / (w1 + w2)
                if abs(s_f - 1) > 0.015:
                    scale_capture(cap, s_f)
                    s_total, src = scale0 * s_f, src + " + door height"
                    out, r, places, links = run_global(cap, cfg, "video", "video/colmap+" + info["depth_backend"],
                                                       device="iPhone 15+ (video)", extra=dict(frames=len(keys)),
                                                       scale_rel=float(np.sqrt(1 / (w1 + w2))))
                    out["quality"]["warnings"].append(f"scale refined by door-height prior (x{s_f:.3f})")
                else:
                    out["quality"]["warnings"].append("scale: depth network + ceiling prior, door-height prior agrees")
            else:
                out["quality"]["warnings"].append(f"scale from the metric depth network (x{info['scale_net']:.3f}) combined with the "
                                                  f"ceiling-height prior (x{info['scale_ceil']:.3f}): +-{100 * sig_rel:.0f} % (no printed card, no door seen)")
        else:
            out["quality"]["warnings"].append(f"scale from printed card ({info['n_marker']} detections, x{scale0:.3f})")
        out["quality"]["scale"] = dict(source=src, scale_net_over_final=round(info["scale_net"] / s_total, 4),
                                       scale_ceiling_prior_over_final=round(info["scale_ceil"] / s_total, 4))
        out["quality"]["warnings"].append(f"sfm: {info['n_reg']}/{len(keys)} informative key-frames registered "
                                          f"({getattr(extract_keyframes, 'dropped', 0)} blank/blurred key-frames skipped)")
        if info["n_reg"] < 0.9 * len(keys):
            out["quality"]["warnings"].append("PARTIAL RECONSTRUCTION: part of the walkthrough could not be registered; "
                                              "rooms seen only there are missing (re-film passing doorways slowly)")
        for k, v in T.items():
            out["timing"][k] = round(v, 3)
        return out, r, places, links
    finally:
        if not cache:
            shutil.rmtree(work, ignore_errors=True)


def _door_scale(rooms) -> Optional[float]:
    hs = []
    for rm in rooms:
        for o in rm.openings:
            if o.kind in ("door", "passage") and o.z0 < 0.05 and (o.s1 - o.s0) < 1.3 and 1.7 < (o.z1 - o.z0) < 2.4:
                hs.append(o.z1 - o.z0)
    if len(hs) < 1:
        return None
    return DOOR_H / float(np.median(hs))


def _build_capture(path: str, work: str, rec, keys, cfg: Config, wh: Tuple[int, int]):
    backend = ph_mod.DepthAnythingV2()
    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    cam = next(iter(rec.cameras.values()))
    fpx, cx, cy = float(cam.params[0]), float(cam.params[1]), float(cam.params[2])
    W, H = wh
    frames_data = []
    for im in imgs:
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R = np.array(cfw.rotation.matrix())
        t = np.array(cfw.translation)
        frames_data.append(dict(im=im, R=R, t=t, C=-R.T @ t, idx=int(os.path.splitext(im.name)[0])))
    # ---- dense depth per key-frame, aligned to the SfM points ------------------------------------
    depth_maps, scales = {}, {}
    for fd in frames_data:
        im = fd["im"]
        img = cv2.cvtColor(cv2.imread(os.path.join(work, "images", im.name)), cv2.COLOR_BGR2RGB)
        d = backend(None, img, f_norm=fpx / max(W, H))
        dh, dw = d.shape
        pts2 = [(p.xy, p.point3D_id) for p in im.points2D if p.has_point3D()]
        if len(pts2) < 12:
            continue
        xy = np.array([p[0] for p in pts2])
        X = np.array([rec.points3D[p[1]].xyz for p in pts2])
        z = (X @ fd["R"].T + fd["t"])[:, 2]
        dp = _bilinear(d, xy[:, 0] * dw / W - 0.5, xy[:, 1] * dh / H - 0.5)
        ok = (z > 0) & (dp > 0.2)
        if ok.sum() < 12:
            continue
        ratio = z[ok] / dp[ok]
        s = float(np.median(ratio))
        inl = np.abs(ratio - s) < 0.15 * s
        if inl.sum() >= 10:
            s = float(np.median(ratio[inl]))
        depth_maps[fd["idx"]] = d
        scales[fd["idx"]] = s
    fd_ok = [fd for fd in frames_data if fd["idx"] in depth_maps]
    if len(fd_ok) < 8:
        raise RuntimeError("too few key-frames with usable depth/SfM overlap")
    # ---- gravity -------------------------------------------------------------------------------------
    up_hint = np.mean([-fd["R"].T @ np.array([0, 1.0, 0]) for fd in fd_ok], axis=0)  # camera -y axis in world
    up_hint /= np.linalg.norm(up_hint)
    samp = []
    for fd in fd_ok[:: max(1, len(fd_ok) // 40)]:
        d = depth_maps[fd["idx"]] * scales[fd["idx"]]
        dh, dw = d.shape
        uu, vv = np.meshgrid(np.arange(0, dw, 3), np.arange(0, dh, 3))
        xn = cam_norm(cam, (uu + 0.5) * W / dw, (vv + 0.5) * H / dh)
        Pc = np.concatenate([xn, np.ones((len(xn), 1))], 1) * d[vv, uu].reshape(-1, 1)
        samp.append((Pc - fd["t"]) @ fd["R"])
    P = np.concatenate(samp)
    u = estimate_up(P, up_hint)
    ax = np.array([1.0, 0, 0]) - u * u[0]
    ax /= np.linalg.norm(ax)
    G = np.stack([ax, np.cross(u, ax), u], 0)  # sfm-world -> gravity frame
    h = P @ u
    from ..geometry.primitives import mode_peak
    lo, hi = float(np.percentile(h, 1)), float(np.percentile(h, 99))
    fl = mode_peak(h, lo, lo + 0.45 * (hi - lo), bin_w=(hi - lo) / 300, min_frac=0.3, which="low", min_count=20)
    ce = mode_peak(h, hi - 0.45 * (hi - lo), hi, bin_w=(hi - lo) / 300, min_frac=0.3, which="high", min_count=20)
    H_sfm = (ce[0] - fl[0]) if (fl and ce) else (hi - lo)
    scale_ceil = PRIOR_CEIL / max(H_sfm, 1e-6)
    # metric depth network: per key-frame ratio SfM depth / network depth -> SfM units to metres = 1 / median ratio
    s_all = np.array([scales[fd["idx"]] for fd in fd_ok])
    scale_net = 1.0 / float(np.median(s_all))
    # combine the two scale sources in log space by inverse variance
    netcal = ph_mod.net_calibration()
    w_c, w_n = 1.0 / CEIL_PRIOR_SIGMA ** 2, 1.0 / (float(netcal["sigma"]) if netcal else NET_SCALE_SIGMA) ** 2
    scale = float(np.exp((w_c * np.log(scale_ceil) + w_n * np.log(scale_net)) / (w_c + w_n)))
    scale_sigma = float(np.sqrt(1.0 / (w_c + w_n)))
    # ---- scale from the printed card -----------------------------------------------------------------
    n_marker, s_cards = 0, []
    for fd in fd_ok:
        img = cv2.cvtColor(cv2.imread(os.path.join(work, "images", fd["im"].name)), cv2.COLOR_BGR2RGB)
        K = np.array([[fpx, 0, cx], [0, fpx, cy], [0, 0, 1.0]])
        det = ph_mod.detect_marker(img, K, cfg.scale_card_mm / 1000.0, ident=7)
        if not det:
            continue
        m = det[0]
        d = depth_maps[fd["idx"]] * scales[fd["idx"]]
        dh, dw = d.shape
        px = m["px"]
        dd = float(np.median(d[max(int(px[1] * dh / H) - 1, 0):int(px[1] * dh / H) + 2,
                               max(int(px[0] * dw / W) - 1, 0):int(px[0] * dw / W) + 2]))
        if dd > 0.05 and m["side_px"] > 10:
            s_cards.append(m["z"] / dd)
    card = False
    if len(s_cards) >= 3 and np.std(s_cards) < 0.08 * np.median(s_cards):
        scale = float(np.median(s_cards))
        scale_sigma = 0.015
        card = True
        n_marker = len(s_cards)
    # ---- capture ---------------------------------------------------------------------------------------
    frames: List[Frame] = []
    parts = []
    for fd in fd_ok:
        Rwc = G @ fd["R"].T
        org = scale * (G @ fd["C"])
        idx = fd["idx"]
        impath = os.path.join(work, "images", fd["im"].name)
        K = np.array([[fpx, 0, cx], [0, fpx, cy], [0, 0, 1.0]])
        frames.append(Frame(len(frames), Rwc, org, K, (W, H), rgb=_lazy(impath), name=fd["im"].name))
        d = depth_maps[idx] * scales[idx] * scale
        dh, dw = d.shape
        uu, vv = np.meshgrid(np.arange(dw), np.arange(dh))
        xn = cam_norm(cam, (uu.reshape(-1) + 0.5) * W / dw, (vv.reshape(-1) + 0.5) * H / dh)
        dc = np.concatenate([xn, np.ones((len(xn), 1))], 1)
        nrm = np.linalg.norm(dc, axis=1)
        dirc = (dc / nrm[:, None]).astype(np.float32)
        rng = (d.reshape(-1) * nrm).astype(np.float32)
        img = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2RGB)
        col = img[np.clip(((vv.reshape(-1) + 0.5) * H / dh).astype(int), 0, H - 1),
                  np.clip(((uu.reshape(-1) + 0.5) * W / dw).astype(int), 0, W - 1)]
        sig = (0.05 * d.reshape(-1)).astype(np.float32)
        parts.append((dirc, rng, np.full(len(rng), KIND_HIT, np.uint8), np.full(len(rng), len(frames) - 1, np.int32),
                      np.stack([uu.reshape(-1), vv.reshape(-1)], 1).astype(np.int16), sig, col.astype(np.uint8)))
    rays = RaySet(*(np.concatenate([q[i] for q in parts]) for i in range(6)))
    rays.rgb = np.concatenate([q[6] for q in parts])
    cap = Capture(frames, rays, dict(tier="video", depth_size=(dw, dh)))
    info = dict(scale=scale, scale_sigma=scale_sigma, card=card, n_marker=n_marker, n_reg=len(frames),
                depth_backend="depth_anything_v2", scale_ceil=scale_ceil, scale_net=scale_net)
    return cap, info


def cam_norm(cam, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pixel -> normalised camera coordinates using COLMAP's camera model (handles radial distortion)."""
    xy = np.stack([np.asarray(x, np.float64).reshape(-1), np.asarray(y, np.float64).reshape(-1)], 1)
    try:
        return np.asarray(cam.cam_from_img(xy))
    except Exception:
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        return np.stack([(xy[:, 0] - cx) / f, (xy[:, 1] - cy) / f], 1)


def _lazy(path):
    def f():
        im = cv2.imread(path)
        return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return f
