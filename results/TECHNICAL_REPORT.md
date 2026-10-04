# floorscan - technical report

**Why the data looks like this.** No iPhone was available, so the benchmark uses Apple's public **ARKitScenes** dataset: 9 real rooms captured with an iPad Pro (same LiDAR + ARKit stack as Pro iPhones) - 2-3 LiDAR sessions, the device video and 8 frames of it as a photo set per room - scored against **FARO laser scans** of the same rooms. Rooms were drawn at random before any was run; 4 DEV rooms were used for fixes and fits, 5 TEST rooms only for the final numbers. Consequences: no scale card in any video or photo set (scale comes from priors and the depth network), no staged damage (damage is evaluated on real photos of damaged walls, BD3), single rooms only (multi-room stitching untested on real data), no consumer-app export (head-to-head NOT RUN), no git history. Details: docs/WHY_THESE_RESULTS.md.

## 1. Architecture

Every tier is reduced to the same object: calibrated camera frames plus a bag of _rays_ (direction, range, sigma, dropout) in a gravity-aligned metric frame. LiDAR rays come from the ARKit depth map; video rays from COLMAP poses (exhaustive matching) plus Depth-Anything-V2 metric depth aligned to the SfM points; photo rays from the same depth network, with gravity from depth normals and yaw from homographies. Shared geometry: free-space carving, persistence watershed and a merge of regions joined wider than any door; rectilinear walls snapped to the innermost full-height plane covering most of the wall (recesses and furniture excluded); floor = lowest real plane, ceiling accepted only if it covers the room and is >= 2 m up (else 'not observed' + prior); plane-anchored drift correction; openings from rays crossing wall planes; door-anchored pose graph for multi-room plans. Damage: candidates from 2 cm/px ortho-textures, verified and classified by DINOv2 features + a linear head trained on real photos; concealed-damage rules and scope lines keyed to surfaces. Every number is {value, ci95, sigma}. Code map: docs/CODE_WALKTHROUGH.md.

## 2. Benchmark data and ground truth

dataprep/fetch_sample.py downloads the pre-registered sample (data/SPLIT.json) and converts each session to our three input layouts. Ground truth (dataprep/room_gt.py, openings_gt.py, no shared code with the pipeline): each session is registered to the venue's laser scans; the laser stations inside the room give floor, ceiling and the four main wall faces (a written convention: innermost full-height face covering >= 40 % of the side; recesses and furniture excluded) and the openings the laser sees as openings; closed flush doors and recesses are ignore regions. Exclusions by rules fixed in advance: one room whose laser does not cover the captured room (replaced by the next random room) and one session that captured a different space. bench/metrics.py aligns each plan to the ground truth (k x 90 deg + translation) and scores it.

## 3. Results: share of measurements inside each case-study gate

| split | tier | captures | walls in gate | ceilings in gate | openings within tol. | footprint in gate | median worst wall err |
|---|---|---|---|---|---|---|---|
| TEST | lidar | 15 | 6/60 (10 %) | 8/14 (57 %) | 0/57 (0 %) | 1/15 | 286 mm |
| TEST | video | 3 | 0/12 (0 %) | 0/2 (0 %) | 0/9 (0 %) | 0/3 | 1391 mm |
| TEST | photos | 5 | 6/20 (30 %) | 3/5 (60 %) | 0/15 (0 %) | 1/5 | 1244 mm |
| DEV | lidar | 11 | 18/44 (41 %) | 7/10 (70 %) | 1/31 (3 %) | 8/11 | 36 mm |
| DEV | video | 4 | 0/16 (0 %) | 1/2 (50 %) | 0/8 (0 %) | 0/4 | 760 mm |
| DEV | photos | 4 | 2/16 (12 %) | 0/4 (0 %) | 0/8 (0 %) | 0/4 | 848 mm |

Gates (case study): LiDAR walls within max(2 cm, 0.5 %), ceiling within 1.5 cm, openings within 2 cm (missed and phantom openings count as misses); video within +-3 %; photos within +-8 %. TEST rows are the held-out result. Per-capture tables, per-wall errors and timing: results/benchmark/BENCHMARK_REPORT.md.

## 4. Repeatability and drift handling

**TEST**: 15 pairs of LiDAR sessions of the same room - walls within max(1 cm, 0.5 %) in 0/15 pairs, ceiling spread <= 1 cm in 6/15, median largest wall difference 2080 mm.

**DEV**: 10 pairs of LiDAR sessions of the same room - walls within max(1 cm, 0.5 %) in 1/10 pairs, ceiling spread <= 1 cm in 5/10, median largest wall difference 295 mm.

| capture (TEST) | drift handling | walls in gate | max wall err | ceiling err | footprint err |
|---|---|---|---|---|---|
| 48018382_lidar | OFF (poses as delivered) | 0/4 | 1205 mm | 4 mm | -36.6 % |
| 48018382_lidar | ON | 0/4 | 59 mm | 13 mm | -2.4 % |
| 47332915_lidar | OFF (poses as delivered) | 1/4 | 1233 mm | 7 mm | -2.9 % |
| 47332915_lidar | ON | 2/4 | 41 mm | 2 mm | -1.1 % |
| 47334107_lidar | OFF (poses as delivered) | 0/4 | 569 mm | 18 mm | +15.2 % |
| 47334107_lidar | ON | 0/4 | 585 mm | 137 mm | +24.8 % |
| 42897501_lidar | OFF (poses as delivered) | 0/4 | 1705 mm | 9 mm | -53.6 % |
| 42897501_lidar | ON | 0/2 | 2263 mm | 19 mm | -57.0 % |
| 47204552_lidar | OFF (poses as delivered) | 2/4 | 155 mm | 9 mm | -4.5 % |
| 47204552_lidar | ON | 2/4 | 109 mm | 2 mm | -3.1 % |

ARKit poses are never used as-is: frames are chunked and each chunk gets an (x, y, yaw, z) correction putting its points on the fitted planes (joint over chunks, smoothness prior, 3 iterations); multi-room captures add a door-anchored pose graph.

## 5. The fix loop (on real data, DEV room)

Fix loop on REAL data (ARKitScenes room 421337, FARO laser ground truth). Worst gate: LiDAR wall lengths - 0 of 8 walls within max(2 cm, 0.5 %), worst error 217 mm. Diagnosis (fixloop/diagnose.py, our wall-band points registered onto the laser scan) showed that 6 of 8 wall planes sit within 2.1 cm of the laser's main planes; the error comes from two planes placed in recesses - the bay window (+10.6 cm, capture 46) and the chimney alcove (+24.1 cm, capture 49) - because the wall plane was the highest histogram peak and the recess back collected more points than the main wall. Fix: among full-height candidate planes that cover >= 60 % of the wall length with >= 25 % of the top support, take the innermost. Predicted worst error 70 mm (the bay-window plane of capture 49 has no main-plane candidate); measured after: 38 mm.

|  | before | predicted | after |
|---|---|---|---|
| worst wall error (both captures) | 217 mm | 70 mm | 38 mm |
| walls in the 2 cm gate | 0/8 | 0-2/8 | 4/8 |

Declaration, laser-registered evidence and post-mortem: fixloop/FIX_DECLARATION.md; diff: fixloop/fix.diff. Further fixes found on the other DEV rooms (room merge, floor/ceiling rules, height and depth-scale fits) are listed in docs/DISCLOSURES.md.

## 6. Calibration of the intervals

| split | tier | n | inside 95 % CI | error > 3 half-widths |
|---|---|---|---|---|
| TEST | lidar | 151 | 53 % | 27 % |
| TEST | video | 15 | 60 % | 20 % |
| TEST | photos | 33 | 91 % | 0 % |
| DEV | lidar | 94 | 48 % | 26 % |
| DEV | video | 17 | 65 % | 35 % |
| DEV | photos | 28 | 89 % | 7 % |

value +- 1.96 k sqrt(sigma_fit^2 + floor^2 + (rel x value)^2); k fitted per tier and kind on the DEV captures (bench/calibrate.py --from-results; DEV rows above were produced with k = 1). The TEST rows show whether that calibration transfers to unseen rooms.

## 7. Damage

Classifier on **793 real photos** it never saw (BD3 test split): class accuracy **94.7 %** over water_stain, mold, hole, peeling, crack, none; damage-vs-clean precision 100.0 %, recall 99.9 % (results/damage/BD3_EVAL.md).

On the undamaged benchmark rooms every detection is a false positive: TEST LiDAR captures report 5.3 detections on average (median 5). Metric extent of real damage cannot be scored without staged damage and measured ground truth, which no public dataset provides.

## 8. Known limits

Video and photos have no scale card in this dataset, so their scale rests on the depth network (15 % spread across DEV captures) and the ceiling/door priors - the main reason those tiers miss their gates; with the protocol's card the scale error is 1-2 %. Sessions that never look at the ceiling report a prior with a wide interval. Multi-room stitching, mirrors, wet floors and low light are not exercised by this sample. Full list: docs/KNOWN_FAILURES.md; what is missing: docs/REAL_CAPTURE_TODO.md.
