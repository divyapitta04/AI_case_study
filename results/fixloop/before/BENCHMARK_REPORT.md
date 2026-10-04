# Benchmark report - real captures (Apple ARKitScenes) vs laser ground truth

> Real iPad Pro captures from Apple's public ARKitScenes dataset (non-commercial research licence): LiDAR depth + ARKit poses (two captures of the same room), the device video, and 8 frames from the walkthrough as the photo set. Ground truth: the FARO laser scan of the same room (`dataprep/room_gt.py`): 3.711 x 4.213 m, ceiling 3.054 m, main wall planes only (fit figure in `data/gt/`). Not an iPhone 15+ capture made with our protocol (no scale card). Openings are not scored (no reliable opening ground truth). The room has no known damage, so every damage detection counts as a false positive. Rebuild the data with `python dataprep/fetch_sample.py`.

## 0. Summary - share of measurements inside each gate

TEST rooms were never used for any design or tuning decision (data/SPLIT.json); DEV rooms were.

| split | tier | captures | walls in gate | ceilings in gate | openings within tolerance (incl. missed/phantom) | footprint in gate | median worst wall error mm |
|---|---|---|---|---|---|---|---|
| DEV | lidar | 2 | 0/8 (0 %) | 1/2 (50 %) | 0/5 (0 %) | 0/2 | 153 |

* DEV repeatability (1 LiDAR session pairs): walls within max(1 cm, 0.5 %) in 0/1 pairs, ceiling spread <= 1 cm in 0/1; median largest wall difference 190 mm.


## 1. Gates at all three tiers

### lidar tier  
Gate: wall <= max(2 cm, 0.5 %); ceiling <= 1.5 cm; opening width <= 2 cm on >= 85 % incl. missed/phantom

| capture | walls ok (max err mm) | ceilings ok (max err mm) | openings | footprint err % | damage detections (all FP) | time s |
|---|---|---|---|---|---|---|
| 42444946_lidar | 0/4 (90) | 0/1 (25) | 0/1 (+2) | +3.1 | 11 | 58 |
| 42444949_lidar | 0/4 (217) | 1/1 (11) | 0/1 (+1) | +7.1 | 9 | 31 |

| capture | wall lengths | ceiling height | openings | footprint |
|---|---|---|---|---|
| 42444946_lidar | FAIL | FAIL | FAIL | FAIL |
| 42444949_lidar | FAIL | PASS | FAIL | FAIL |


## 2. Per-wall errors (measured - laser, mm)

| capture | S | E | N | W | ceiling |
|---|---|---|---|---|---|
| 42444946_lidar | +90 | +28 | +90 | +28 | -25 |
| 42444949_lidar | +68 | +217 | +68 | +217 | -11 |


## 3. Repeatability (same room, same tier, two captures)

| pair | per-wall difference between the two captures | ceiling height a / b | walls <= max(1 cm, 0.5 %) | ceiling spread <= 1 cm | verdict (repeatable-but-biased vs unrepeatable) |
|---|---|---|---|---|---|
| lidar (42444946_lidar vs 42444949_lidar) | S: +22 mm (tol 19), E: -190 mm (tol 21), N: +22 mm (tol 19), W: -190 mm (tol 21) | 3.028 / 3.043 (spread 15 mm, bias -25/-11 mm) | FAIL | FAIL | UNREPEATABLE |


## 4. Drift accountability (pose correction ON vs poses used as-is)

Ablation not run (`--no-ablation`).


Method: ARKit poses are never used as-is. Per room, frames are chunked in time and each chunk gets an (x, y, yaw, z) correction so its points sit on the fitted wall / floor / ceiling planes (joint over chunks with a smoothness prior, 3 iterations); in multi-room captures a door-anchored pose graph then places the rooms. OFF = poses used as delivered.


## 5. Calibration of the confidence intervals

| tier | n measurements | inside 95 % CI | error > 3x half-width ('confident garbage') | by measurement |
|---|---|---|---|---|
| lidar | 18 | 22 % | 44.4 % | area: 0 % (n=2), footprint: 0 % (n=2), height: 50 % (n=2), length: 25 % (n=8), opening_height: 50 % (n=2), opening_width: 0 % (n=2) |


Intervals: analytic sigma + per-tier systematic terms (`floorscan/uncertainty.py`) x the calibration factor in `floorscan/calibration.json` (currently k = 1: no held-out real calibration captures exist; `bench/calibrate.py --data <dir>`).


## 6. Timing (this machine, CPU only)

| capture | total s | stages (s) |
|---|---|---|
| 42444946_lidar | 57.8 | openings 7.5, stitch 0.0, levels_yaw 0.5, segment 17.5, regions 0.0, room_fit 25.4, damage 4.2 |
| 42444949_lidar | 31.0 | openings 3.6, stitch 0.0, levels_yaw 0.5, segment 6.4, regions 0.0, room_fit 12.9, damage 2.4 |
