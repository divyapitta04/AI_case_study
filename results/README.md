# results/ — every generated output

All generated from `../data/` by `scripts/reproduce.ps1` (or `.sh`); nothing here is hand-edited.

| what | file |
|---|---|
| **Technical report** (<= 6 pages) | `TECHNICAL_REPORT.pdf` (+ `.md`, `architecture.png`) |
| **Benchmark report**: section 0 = share of measurements inside each gate, TEST vs DEV; per-capture gates, per-wall errors, repeatability, drift ablation, calibration, timing | `benchmark/BENCHMARK_REPORT.md`, `benchmark/metrics.json` |
| Plans of every capture | `benchmark/runs/<capture>/plan.{json,png,svg}` (`*_driftoff` = ablation runs) |
| Drift ablation figures (TEST rooms) | `benchmark/drift_ablation_*.png` |
| Head-to-head vs a consumer app | `benchmark/HEAD_TO_HEAD.md` — **NOT RUN** (no app export) |
| **Fix loop** | `fixloop/FIXLOOP_REPORT.md`, `fixloop/summary.json`, `fixloop/before/`, `fixloop/after/`, `fixloop/diagnosis_*` |
| Damage classifier on real photos (BD3 test split) | `damage/BD3_EVAL.md`, `damage/bd3_metrics.json` |
| Real iPhone 15 Pro photos → plan (no ground truth exists) | `iphone15pro_photos/plan.{json,png,svg}` |

DEV-room plans were produced with interval factor k = 1 (they are the calibration set); TEST plans with the DEV-fitted k.
Two TEST videos (42897501, 47204552) were not processed for CPU-time reasons.
