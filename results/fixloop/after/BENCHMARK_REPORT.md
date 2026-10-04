# Benchmark report - real captures (Apple ARKitScenes) vs laser ground truth

> Real iPad Pro captures from Apple's public ARKitScenes dataset (non-commercial research licence): LiDAR depth + ARKit poses (two captures of the same room), the device video, and 8 frames from the walkthrough as the photo set. Ground truth: the FARO laser scan of the same room (`dataprep/room_gt.py`): 3.711 x 4.213 m, ceiling 3.054 m, main wall planes only (fit figure in `data/gt/`). Not an iPhone 15+ capture made with our protocol (no scale card). Openings are not scored (no reliable opening ground truth). The room has no known damage, so every damage detection counts as a false positive. Rebuild the data with `python dataprep/fetch_sample.py`.

## 0. Summary - share of measurements inside each gate

TEST rooms were never used for any design or tuning decision (data/SPLIT.json); DEV rooms were.

| split | tier | captures | walls in gate | ceilings in gate | openings within tolerance (incl. missed/phantom) | footprint in gate | median worst wall error mm |
|---|---|---|---|---|---|---|---|
| DEV | lidar | 2 | 4/8 (50 %) | 1/2 (50 %) | 0/9 (0 %) | 2/2 | 26 |

* DEV repeatability (1 LiDAR session pairs): walls within max(1 cm, 0.5 %) in 0/1 pairs, ceiling spread <= 1 cm in 0/1; median largest wall difference 44 mm.


## 1. Gates at all three tiers

### lidar tier  
Gate: wall <= max(2 cm, 0.5 %); ceiling <= 1.5 cm; opening width <= 2 cm on >= 85 % incl. missed/phantom

| capture | walls ok (max err mm) | ceilings ok (max err mm) | openings | footprint err % | damage detections (all FP) | time s |
|---|---|---|---|---|---|---|
| 42444946_lidar | 0/4 (38) | 0/1 (26) | 0/1 (+2) | -0.4 | 10 | 34 |
| 42444949_lidar | 4/4 (14) | 1/1 (13) | 0/1 (+5) | -0.2 | 11 | 27 |

| capture | wall lengths | ceiling height | openings | footprint |
|---|---|---|---|---|
| 42444946_lidar | FAIL | FAIL | FAIL | PASS |
| 42444949_lidar | PASS | PASS | FAIL | PASS |


## 2. Per-wall errors (measured - laser, mm)

| capture | S | E | N | W | ceiling |
|---|---|---|---|---|---|
| 42444946_lidar | -38 | +27 | -38 | +27 | -26 |
| 42444949_lidar | +5 | -14 | +5 | -14 | -13 |


## 3. Repeatability (same room, same tier, two captures)

| pair | per-wall difference between the two captures | ceiling height a / b | walls <= max(1 cm, 0.5 %) | ceiling spread <= 1 cm | verdict (repeatable-but-biased vs unrepeatable) |
|---|---|---|---|---|---|
| lidar (42444946_lidar vs 42444949_lidar) | S: -44 mm (tol 19), E: +41 mm (tol 21), N: -44 mm (tol 19), W: +41 mm (tol 21) | 3.028 / 3.040 (spread 13 mm, bias -26/-13 mm) | FAIL | FAIL | UNREPEATABLE |


## 4. Drift accountability (pose correction ON vs poses used as-is)

Ablation not run (`--no-ablation`).


Method: ARKit poses are never used as-is. Per room, frames are chunked in time and each chunk gets an (x, y, yaw, z) correction so its points sit on the fitted wall / floor / ceiling planes (joint over chunks with a smoothness prior, 3 iterations); in multi-room captures a door-anchored pose graph then places the rooms. OFF = poses used as delivered.


## 5. Calibration of the confidence intervals

| tier | n measurements | inside 95 % CI | error > 3x half-width ('confident garbage') | by measurement |
|---|---|---|---|---|
| lidar | 18 | 61 % | 11.1 % | area: 100 % (n=2), footprint: 100 % (n=2), height: 50 % (n=2), length: 75 % (n=8), opening_height: 0 % (n=2), opening_width: 0 % (n=2) |


Intervals: analytic sigma + per-tier systematic terms (`floorscan/uncertainty.py`) x the calibration factor in `floorscan/calibration.json` (currently k = 1: no held-out real calibration captures exist; `bench/calibrate.py --data <dir>`).


## 6. Timing (this machine, CPU only)

| capture | total s | stages (s) |
|---|---|---|
| 42444946_lidar | 34.0 | openings 3.3, stitch 0.0, levels_yaw 0.3, segment 10.5, regions 0.0, room_fit 16.1, damage 2.5 |
| 42444949_lidar | 27.2 | openings 3.3, stitch 0.0, levels_yaw 0.2, segment 7.1, regions 0.0, room_fit 12.5, damage 2.0 |
