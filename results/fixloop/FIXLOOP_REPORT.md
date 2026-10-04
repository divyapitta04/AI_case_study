# Fix loop on real data: before / after

Gate: **LiDAR wall lengths within max(2 cm, 0.5 %) of the laser ground truth (real ARKitScenes captures 42444946 + 42444949)**.  
Worst wall-length error over both captures: **before 217 mm -> after 38 mm** (predicted 70 mm; required <= 20 mm). Walls in the gate: 0/8 -> 4/8.

| capture | worst wall error mm | walls in gate | footprint error | ceiling error mm |
|---|---|---|---|---|
| 42444946_lidar | 90 -> **38** | 0/4 -> 0/4 | +3.1 % -> -0.4 % | 25 -> 26 |
| 42444949_lidar | 217 -> **14** | 0/4 -> 4/4 | +7.1 % -> -0.2 % | 11 -> 13 |

Repeatability (largest per-wall difference between the two captures): 190 mm -> 44 mm (UNREPEATABLE -> UNREPEATABLE).

Plans and metrics of each run: `before/`, `after/`. Root-cause evidence: `diagnosis_before.md` / `diagnosis_after.md`. Declaration and post-mortem: `fixloop/FIX_DECLARATION.md`.
