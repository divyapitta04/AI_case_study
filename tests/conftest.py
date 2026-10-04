"""Shared fixtures: the real ARKitScenes captures in data/ (rebuild them with `python dataprep/fetch_sample.py`)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
GT = os.path.join(DATA, "gt", "421337.json")


def _need(path):
    if not os.path.exists(os.path.join(path, "capture.json")):
        pytest.skip(f"real capture missing: {path} (run python dataprep/fetch_sample.py)")
    return path


@pytest.fixture(scope="session")
def gt():
    return json.load(open(GT))


@pytest.fixture(scope="session")
def real_lidar():
    return _need(os.path.join(DATA, "42444946_lidar"))


@pytest.fixture(scope="session")
def real_photos():
    return _need(os.path.join(DATA, "42444946_photos"))


@pytest.fixture(scope="session")
def real_video():
    return _need(os.path.join(DATA, "42444946_video"))


@pytest.fixture(scope="session")
def lidar_plan(real_lidar, tmp_path_factory):
    """One pipeline run on the real LiDAR capture, shared by the tests that only read the plan."""
    from floorscan.cli import run
    d = tmp_path_factory.mktemp("lidar_plan")
    return run(real_lidar, str(d), quiet=True), str(d)
