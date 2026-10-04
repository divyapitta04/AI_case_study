# Fix declaration (written before the fix; sections 1-4 are not edited afterwards)

## 1. Worst gate and the failing number

Real benchmark (`results/benchmark/BENCHMARK_REPORT.md`, ARKitScenes room 421337, FARO laser ground truth), LiDAR tier:

| gate | required | measured (before) | how far off |
|---|---|---|---|
| **wall lengths within max(2 cm, 0.5 %)** | all walls | **0 of 8 walls**; worst error **224 mm** (capture 42444949), 91 mm (42444946) | 11x the tolerance |
| repeatability (two captures, same room) | <= max(1 cm, 0.5 %) per wall | one wall pair differs by **190 mm** | UNREPEATABLE |
| ceiling height within 1.5 cm | | -41 / -26 mm | 2-3x |
| openings | | not scorable (no opening ground truth) | - |

Wall lengths are the worst gate: largest miss relative to tolerance, and the same defect drives the 190 mm repeatability failure.
(Video and photo tiers fail by more, but on scale - the dataset has no scale card - which is a capture-protocol issue, not a code defect we can fix and verify here.)

## 2. Root-cause hypothesis and the evidence

**First hypothesis (rejected by the evidence):** the "wall behind furniture" rule in `pick_wall_peak` (a plane >= 18 cm further
out wins with half the support) moves walls into recesses. `fixloop/diagnose.py` shows that rule **never fired** on either capture.

**Hypothesis:** each wall plane is the *single highest histogram peak* of wall-band points within +-45 cm. When a wall has a recess
(bay window, chimney alcove) whose back surface collects more LiDAR points than the main wall face, the plane snaps into the recess
and the room grows by the recess depth.

**Evidence** (`results/fixloop/diagnosis_before.md`; our wall-band points registered onto the laser scan by ICP, 66-73 % of points
within 5 cm of a laser point):

| capture | wall | our plane vs laser main plane | candidate at the main plane (our points, length coverage, full height) | candidate taken |
|---|---|---|---|---|
| 46 | W, N, S | -1.4 / +2.1 / +1.4 cm | - (taken) | main plane |
| 46 | **E** | **+10.6 cm** | -0.7 cm: 42k pts, 95 %, yes | bay window, 123k pts, 86 % |
| 49 | W, S | -1.8 / -1.7 cm | - (taken) | main plane |
| 49 | **N** | **+24.1 cm** | -1.3 cm: 61k pts, 78 %, yes | chimney alcove back, 64k pts, **46 %** |
| 49 | **E** | **+8.8 cm** | **none** (no peak near the main plane) | bay window, 117k pts, 93 % |

* 6 of 8 planes are within 2.1 cm of the laser: the plane fit itself is fine; the *choice* between candidates is wrong.
* In both wrong cases the correct plane is a candidate that spans most of the wall (78-95 %) and reaches the ceiling, while the
  chosen recess plane is further out; in 49-N it also covers only 46 % of the wall.
* The 190 mm repeatability failure is the 49-N alcove choice (46-N took the main plane).

## 3. Intended fix

In `refine_walls` (`floorscan/geometry/room.py`): list all candidate planes in the window; keep those that are **full height**
(97th-percentile z of their points >= ceiling - 0.5 m, the same test `drop_objects` uses for furniture), cover **>= 60 % of the
wall length** (10 cm bins) and have **>= 25 % of the highest candidate's support**; take the **innermost** of them. If none
qualifies, fall back to the current rule. Furniture in front of a wall does not reach the ceiling and is never taken; a recess
back is behind the main face and loses to it.

## 4. Predicted number

* 46: E moves from +10.6 to about -0.7 cm; the other walls stay. Worst wall-length error 91 mm -> **about 35 mm** (N-S: +2.1 +1.4 cm).
* 49: N moves from +24.1 to about -1.3 cm; E stays at +8.8 cm (no main-plane candidate). Worst error 224 mm -> **about 70 mm**.
* **Pooled worst wall-length error: 224 mm -> about 70 mm.** Walls in the 2 cm gate: 0/8 -> 0-2/8. **The gate will still fail.**
* Repeatability: the 190 mm wall-pair difference drops to about 90 mm (the W-E side now differs by the 49-E bay offset).
* Risk: the thresholds (60 %, 25 %) were read off these two captures of one room - generalisation is unproven; a floor-to-ceiling
  wardrobe covering most of a wall would now be taken as the wall.

## 5. Post-mortem (written after the run)

Measured (`results/fixloop/FIXLOOP_REPORT.md`, `results/fixloop/diagnosis_after.md`):

| | before | predicted | after |
|---|---|---|---|
| worst wall-length error (both captures) | 224 mm | ~70 mm | **37 mm** |
| walls in the 2 cm gate | 0 / 8 | 0-2 / 8 | **4 / 8** (capture 49: 4/4, max 7 mm; capture 46: 0/4, max 37 mm) |
| footprint error 46 / 49 | +3.2 % / +7.1 % | | **-0.2 % / +0.0 %** |
| repeatability, largest wall-pair difference | 190 mm | ~90 mm | **44 mm** (still UNREPEATABLE) |
| ceiling error 46 / 49 | -41 / -26 mm | unchanged | -41 / -29 mm |

_Update: the ground truth of this room was later rebuilt from all laser stations inside it (`data/gt/README.md`); the same before / after runs then give 217 -> 38 mm and 0/8 -> 4/8 walls (`results/fixloop/FIXLOOP_REPORT.md`); the conclusions below are unchanged._

**What the prediction got right.** The mechanism: 46-E moved from the bay window (+10.6 cm) to the main face (-1.4 cm), 49-N
from the alcove back (+24.1 cm) to the main face (+0.5 cm), exactly the candidates the diagnosis pointed at. The other walls did
not move by more than 1 cm. The gate still fails, as declared.

**What it got wrong, and why.** I predicted 49-E would stay at +8.8 cm because it had no main-plane candidate. It moved to
+3.6 cm anyway. The reason, from `diagnosis_after.md`: the wrong north plane was also the anchor for the plane-anchored pose
correction, which bent the trajectory towards the alcove and pushed the east wall out. With the right north plane, the
correction stopped distorting the room. The same coupling explains why, before the fix, "drift handling OFF" looked better
than ON on capture 49 (93 vs 224 mm); after the fix, ON beats OFF on both captures (37 vs 116 mm, 7 vs 93 mm). I did not
foresee that a plane choice error is amplified by the drift correction; that interaction is the main lesson.

**Why the gate still fails.** Capture 46 is 37 mm short on the W-E side: E is -1.4 cm and W is -2.3 cm, both inside the
laser main plane. Each wall is within 2.3 cm, but the length error is the sum of two walls. This is a different error from
the plane-choice defect: a few-cm inward bias on both sides, possibly a depth-scale bias of this capture. It is not
diagnosed and not addressed. Ceiling heights (-29 to -41 mm) also still fail and are also not addressed.

**Risk that remains.** The thresholds (60 % length coverage, 25 % relative support, full height) were read off these two
captures of one room. They are plausible rules but unproven on other rooms. A floor-to-ceiling wardrobe covering most of a
wall would now be taken as the wall.
