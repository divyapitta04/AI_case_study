"""The main use case: a user's own iPhone 15 Pro photos of a room go in, a valid plan comes out.

Fixture: 4 real iPhone 15 Pro photos of one empty room (0.5x and 1x lens, EXIF intact) from Wikimedia Commons, series
"20250310 Alfter Kronenstrasse 16" (CC BY-SA 4.0, see data/iphone15pro_photos/ATTRIBUTION.md).  No measured dimensions
exist for this room, so the test checks that the photo tier runs and reports honest uncertainty, not accuracy.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = os.path.join(ROOT, "data", "iphone15pro_photos")


def test_iphone15pro_photos_run_end_to_end(tmp_path):
    from floorscan.cli import run, validate
    from floorscan.ingest import detect_tier
    from floorscan.ingest.photos import list_photos
    assert detect_tier(CAP) == "photos"
    photos = [p for ps in list_photos(CAP).values() for p in ps]
    assert len(photos) == 4 and all(p.f35_from_exif for p in photos)  # focal length read from the iPhone EXIF
    out = run(CAP, str(tmp_path / "o"), quiet=True)
    assert validate(out) == []
    assert len(out["rooms"]) == 1
    r = out["rooms"][0]
    assert 5.0 < r["floor_area"]["value"] < 40.0  # a plausible room, not a degenerate one
    lo, hi = r["ceiling_height"]["ci95"]
    assert hi - lo > 0.3  # no scale card in these photos: the interval must be wide, not falsely precise
    assert os.path.getsize(tmp_path / "o" / "plan.png") > 1000
