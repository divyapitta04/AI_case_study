# How to capture a property (iPhone 15 or newer) — one page

You do **not** need to be an engineer. Follow this page literally; if something is unclear, do the safer option.

**You need:** your iPhone · the printed **scale card** (`scale_card_A4.pdf` / `scale_card_Letter.pdf`, print at *100 % / Actual size*, then check the bar measures **10.0 cm** with a ruler) · tape · 10 minutes per 3 rooms.

## Before you start (every option)
1. Open **every** interior door and the front door. Switch **all lights on**. Move people and pets out.
2. Tape the **scale card** flat on a wall at chest height, in plain view (first room for video/LiDAR; every room for photos if you can).
3. Settings → Camera → **Formats → Most Compatible** (JPEG / H.264 – the pipeline cannot read HEIC/HEVC).
4. Note anything shiny: big **mirrors**, glass doors, wet/glossy floors. Cover a mirror with a towel if you can. Don't film the sun through a window.

## Option A – PHOTOS (any iPhone 15+, the always-works fallback)
Camera app → tap **.5** (ultra-wide) → hold the phone **level, landscape**.
* One **folder per room**, named by the room: `living`, `kitchen`, `hall`, `bedroom`.
* Stand **in the middle** of the room. **Turn clockwise on the spot**, one photo about every 45° (each photo must overlap the last by about half). 6–8 photos. *Never more than 8 per room, doorway photos included.*
* For each doorway, aim straight at it, take the photo, then **rename it** `…_to_<neighbour>` e.g. `IMG_0412_to_hall.JPG` (it still counts as one of your 8).
* Zip the folders → send to us.

## Option B – VIDEO (any iPhone 15+)
Camera → Video → tap **.5** → 1080p. Hold the phone at chest height, **level**.
1. Start recording **facing the scale card** from ~1 m for 3 seconds.
2. Walk slowly (one step per second). In each room walk along the walls and **slowly tilt up to the ceiling edge and down to the floor edge**; look at every wall, door and window once.
3. Walk **through** every doorway. Finish **where you started** (the loop helps accuracy). 2–3 minutes for 3 rooms.
4. Send the `.mp4`/`.mov` file (in a folder by itself).

## Option C – LiDAR (iPhone 12 Pro or newer, Pro / Pro Max only) — best accuracy
Install **Polycam** (App Store). New capture → **LiDAR** mode. Walk exactly as in Option B (card first, every wall, every doorway, finish at the start). Stop → Share/Export → **Raw Data** → save the `.zip` → send it to us. *(Raw-data export may need a paid Polycam plan; check on the day before the visit.)*

## What to avoid
| Don't | Why |
|---|---|
| fast turning, shaking, blurry frames | tracking breaks |
| covering the lens / flash | LiDAR and depth need a clean view |
| long looks at mirrors, glass doors, sun in windows | phantom rooms / dropouts |
| dark rooms | turn the lights on |
| only filming a blank wall | nothing for the software to lock onto – include doors, frames, furniture |
| moving people / pets | they become "walls" |

## Hand-off to the software
Put each capture in its own folder and run: `python -m floorscan run <folder> --out <result_folder>` — the tier (photos / video / LiDAR) is detected automatically. Open `<result_folder>/plan.png`.
