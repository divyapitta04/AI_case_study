import numpy as np

from floorscan.geometry import room as R
from floorscan.geometry import stitch as S
from floorscan.geometry.primitives import Grid, _watershed_persistence, mode_peak


def test_mode_peak_finds_plane_and_rejects_outliers():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(2.5, 0.004, 5000), rng.uniform(0, 4, 800)])
    p = mode_peak(v, 0.0, 4.0, bin_w=0.01)
    assert abs(p[0] - 2.5) < 0.005


def test_rectilinear_runs_from_mask():
    mask = np.zeros((120, 160), bool)
    mask[20:100, 30:140] = True
    g = Grid(np.array([0.0, 0.0]), np.array([8.0, 6.0]), 0.05)
    g.lo = np.zeros(2)
    runs = R.runs_from_mask(mask, g)
    walls = R.walls_from_runs(runs)
    assert len(walls) == 4
    L = sorted(w.length for w in walls)  # contour pixels are inside the mask by ~1 pixel; walls are refined later
    assert np.allclose(L, [4.0, 4.0, 5.5, 5.5], atol=0.15)


def test_watershed_separates_rooms_joined_by_a_door():
    free = np.zeros((60, 140), bool)
    free[5:55, 5:60] = True
    free[5:55, 80:135] = True
    free[27:33, 60:80] = True  # doorway 6 cells wide
    from scipy import ndimage as ndi
    D = ndi.distance_transform_edt(free) * 0.05
    lab = _watershed_persistence(D, free, 0.12)
    assert len(np.unique(lab[lab > 0])) == 2


def test_pose_graph_closes_a_door_link():
    """Two rooms joined by a door, second placed with a (0.3, 0.25) m drift -> door residual drops below 1 cm."""
    def room(w, d, doors):
        walls = R.walls_from_runs([["H", 0.0], ["V", w], ["H", d], ["V", 0.0]])
        rm = R.RoomModel("r", walls, 0.0, 2.5)
        for wi, s0, s1 in doors:
            rm.openings.append(R.Opening(wi, s0, s1, 0.0, 2.0, "door"))
        return rm

    A = room(4, 3, [(1, 1.0, 1.9)])  # east wall door
    B = room(3, 3, [(3, 1.0, 1.9)])  # west wall door
    places = [S.Placement(), S.Placement()]
    places[1].p = np.array([4.12 + 0.3, 0.25])
    links = [S.Link(0, 0, 1, 0)]
    S.solve_translations([A, B], places, links, t=0.12)
    assert links[0].residual < 0.01
