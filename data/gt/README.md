# Ground truth (laser-derived) — how it is made

Source: the FARO terrestrial laser scans that Apple's ARKitScenes provides for each visit (mm-level ranging), sampled with
`dataprep/fetch_sample.py` (1/8 of every scan's columns, all directions). Built by `dataprep/room_gt.py` and
`dataprep/openings_gt.py`, which share no code with the pipeline. The conventions below were written down before any
TEST room was run through the pipeline.

## Rooms, walls, ceiling (`dataprep/room_gt.py`)
1. Every LiDAR session of the room is registered to the laser scans from its raw depth + ARKit poses only
   (`dataprep/register.py`: Manhattan yaw hypotheses x 4 x 90 deg x correlation peaks, refined by ICP; result per session in
   `registration/`). A visit's laser stations can come from separate scanning sessions with their own frames; the session is
   registered to each and the best fit is used.
2. The laser stations standing in the captured room (within 1.5 m of the best-registered session's camera path) are used,
   cropped to the room.
3. Floor = lowest strong horizontal plane, ceiling = highest strong horizontal plane (a bed / table top is never the floor).
4. Room axes = the yaw with the sharpest wall histograms.
5. **Wall convention:** each of the four walls is the **main inner wall face** on that side: the innermost full-height
   wall surface (not furniture, not window glass or sills) that covers at least 40 % of the side's length (e.g. a chimney
   breast across 46 % of the wall); if none does, the dominant face. Recesses behind it (bay windows, alcoves, door reveals)
   are excluded. A person picks a search window around that face on the slice figure (`<visit>_laser_fit.png`, candidate
   planes drawn); the plane is then fitted automatically (range²-weighted histogram peak, 5 mm bins). Windows and notes per
   room: `windows.json`.

## Openings (`dataprep/openings_gt.py`)
* A person marks each opening roughly on laser elevation images of the four walls (`openings/<visit>_walls.png`: wall face in
  its laser colour, what the laser saw *through* the wall plane in blue). Edges are then snapped automatically to the
  laser: where the share of solid points (at or in front of the plane) per 1 cm column / row crosses 50 %, persisting for
  3 cm. Edges that are a room corner are kept as marked. Marks: `openings_annot.json`.
* Only openings the laser shows as openings (seen through, or recessed) are ground truth. A closed door flush with the wall
  has no geometric evidence; it and recesses that are part of the room (bay windows, alcoves, shower enclosures, kitchen
  units) are **ignore regions**: an opening reported there is neither credited nor counted as phantom (`bench/metrics.py`).

## Exclusions (rules fixed before any TEST result)
* A room whose sessions register to its laser scans with < 50 % inliers has no usable ground truth (the laser stations do
  not cover the captured room): visit 471948 -> replaced by the next visit in the pre-registered order (471952).
* A session whose floor-to-ceiling (raw depth) is more than 0.2 m *higher* than the laser room's captured a different space:
  session 42445021 (3.12 m vs 2.32 m).
Both are recorded in `../SPLIT.json`.

## Files
`<visit>.json` (dimensions, openings, ignore regions), `<visit>_laser_fit.png` (slice + fitted walls), `<visit>_laser_slice.npz`
(wall-band slice + planes, for diagnosis), `openings/<visit>_walls.png`, `index.json` (capture id -> ground-truth file),
`registration/<video>.json`, `windows.json`, `openings_annot.json`.

Ground truth is laser-derived, not tape-measured: the wall convention and the opening marks involve human judgement; the
plane fits and edge snapping are automatic. Repeating a whole room's fit from different laser stations moved walls by
<= 7 mm and the ceiling by 1.5 cm (room 421337), which bounds how precise the reference itself is.
