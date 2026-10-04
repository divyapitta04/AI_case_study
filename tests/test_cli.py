import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_tier_detection(real_lidar, real_photos, real_video):
    from floorscan.ingest import detect_tier
    assert detect_tier(real_lidar) == "lidar"
    assert detect_tier(real_photos) == "photos"
    assert detect_tier(real_video) == "video"


def test_one_command_per_capture(real_photos, tmp_path):
    """`python -m floorscan run <capture> --out <dir>` is the whole interface."""
    out = tmp_path / "o"
    r = subprocess.run([sys.executable, "-m", "floorscan", "run", real_photos, "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-500:]
    plan = json.load(open(out / "plan.json"))
    assert plan["schema_version"] == "floorscan-1.0"
    assert (out / "plan.png").exists() and (out / "plan.svg").exists()
    v = subprocess.run([sys.executable, "-m", "floorscan", "validate", str(out / "plan.json")], cwd=ROOT,
                       capture_output=True, text=True)
    assert v.stdout.strip() == "OK"
