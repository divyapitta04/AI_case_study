# Known failure modes (and what the pipeline does about them)

Real properties contain mirrors, glass, wet-look surfaces and low light. Each is handled explicitly, and where it is not
handled the output says so (warnings / widened intervals) instead of returning a confident number.

| Hazard / failure | What happens | Mitigation in code | Residual risk |
|---|---|---|---|
| **Mirror** (LiDAR) | ToF returns the unfolded path -> a phantom "room" behind the wall, or dropouts | candidate openings need *jamb evidence or a visited neighbouring room*; dropout-type candidates must look like sky/glass (blue tint or brighter than the wall) – a mirror reflects the room and is rejected, listed in `room.notes` ("suspect_mirror") | a mirror on an exterior wall facing a bright window can pass as a window; mirror mounted flush in a doorway |
| **Window glass** (LiDAR) | depth dropouts over the pane | dropout-dominated regions with sky colour -> `window`; masked out of damage textures (depth-dropout fraction > 50 %) | tinted / blinds-down windows with partial returns |
| **Windows in the photo / video tiers** | monocular depth sees the pane as the wall plane | glass-like colour evidence (blue tint + bright) | night-time photos: dark panes are not detected; frosted glass |
| **Wet-look / glossy floors** | specular highlights, saturated pixels | texels with a channel > 248 are never classified as damage; floor mould class disabled (plank seams look like speckle) | large glare patches hide real damage |
| **Low light** | noise, no features, depth-model degradation | luminance check -> damage confidence x 0.7 and a warning; photo registration falls back from SIFT to *Manhattan + range-profile* matching (works without texture); video drops blank/blurred key-frames | very dark rooms may not register at all (partial reconstruction is reported, rooms listed as missing) |
| **Blank painted walls** | too few features for SfM / photo registration | CLAHE + low-contrast SIFT; geometry-only yaw registration for photos; SfM key-frames without features are skipped | long corridors of blank wall break SfM into several models; only the largest model is used and the output carries `PARTIAL RECONSTRUCTION` |
| **Narrow corridors (< ~1.1 m)** | room segmentation by distance-transform persistence may merge/split; openings seen only at grazing angles | rays weighted by cos^3 of the incidence angle for opening evidence; through-openings extended to the floor when only the upper part is seen | hall doors are the weakest measurements in every tier |
| **Furniture in corners** | free-space mask contains a notch | wall candidates that do not reach the ceiling are treated as objects and removed; corners are closed by the neighbouring walls | floor-to-ceiling cabinets look like walls |
| **Bay windows, alcoves (recesses)** | the recess back can collect more points than the main wall face, so the plane snaps into the recess (real capture 49: +24 cm) | recess-aware plane choice: innermost full-height plane covering >= 60 % of the wall (`pick_wall_plane`, the fix loop) | thresholds set on one room; a recess covering most of a wall is taken as the wall |
| **Non-rectilinear rooms** | the room model is a rectilinear polygon (Manhattan assumption) | rectilinear polygons incl. L-shapes are supported by the contour fit | curved / diagonal walls are snapped to the dominant axes |
| **VIO drift** | poses of the LiDAR tier accumulate cm-level error | plane-anchored per-room refinement + door pose graph; on the real captures it lowers the worst wall error from 116 / 93 mm (poses as delivered) to 37 / 7 mm | tracking *loss* (not modelled) |
| **Scale in photo / video** | monocular depth is metric only to ~5 % | scale card (1-2 %) > standard door height (~3 %) > ceiling prior (10 %), always reported in the interval | houses with non-standard door/ceiling heights and no card |
| **Damage detector** | classical colour/shape/depth rules, no learned model | decorations rejected (axis-aligned rectangles, busy texture), thin slivers at occlusion edges rejected, depth inset used for holes when depth exists | paint that looks like a stain; dark pictures; real mould that is not speckled |
| **Concealed damage** | rule base, not physics | every flag names the rule and the triggering regions; rules are data (`floorscan/reasoning.py: RULES`) | rules are heuristics, not code citations |
| **Wall thickness** | unobservable in photos | 12 cm default (CI widened); measured from door jambs in LiDAR when visible | thick masonry walls |

What the pipeline never does: invent a room it did not see. Rooms with no adjacency evidence are placed beside the plan and flagged
"not a real position"; a wall seen in no photo is reported `observed: false` with a +-0.9 m prior interval.

## Observed on the real benchmark (9 ARKitScenes rooms vs FARO laser; `results/benchmark/`)

| Failure | Seen in | What was done | Remaining |
|---|---|---|---|
| wall plane in a recess (bay window, alcove) | DEV 421337: 224 mm | recess-aware plane choice (fix loop) | recesses covering most of a wall |
| furniture splits one room into 2-3 rooms | DEV 467293: walls off by 2.2 m | merge regions joined by > 1.3 m of free space | TEST cluttered rooms (kitchen with partitions, bathroom) still mis-segmented |
| bed / wardrobe top taken as floor / ceiling | DEV 434650 | floor = lowest real plane; ceiling must be >= 2 m up and cover the room, else "not observed" + prior | narrow furnished rooms with the floor hidden |
| session never looks at the ceiling | DEV 434650, TEST 469782 | reported as not observed, prior 2.5 m ± 0.25 m | needs the protocol's "look at the ceiling edge" |
| iPad heights ~1 % short | 8 of 9 DEV sessions | DEV-fitted height factor 1.0107 | per-device factor unknown for iPhones |
| no room reconstructed (capture held low / too short) | DEV 42898817, TEST 42897504 | empty plan with a clear warning instead of a crash | recapture |
| video / photo scale without a card | every video / photo set | depth network + priors with honest intervals | ±15 % between rooms; the card fixes it |
| COLMAP track loss | DEV 42444946: 60 % registered | exhaustive matching (96 %) | clips with > 300 key-frames fall back to sequential |
| openings: phantoms and width errors | all tiers | ignore regions for recesses in the ground truth | 0 % of TEST openings within 2 cm |
| damage false positives (furniture, pictures, wood floors) | all rooms | learned verifier (DINOv2 + head on real photos) | several detections per undamaged room remain |
