import cv2
import numpy as np

from floorscan.ingest.photos import detect_marker


def test_marker_pnp_gives_metric_distance():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    bm = cv2.aruco.generateImageMarker(d, 7, 200)
    img = np.full((480, 640), 255, np.uint8)
    f, side = 480.0, 0.10 * 480 / 1.5  # a 10 cm marker at 1.5 m is ~32 px wide
    small = cv2.resize(bm, (int(side), int(side)), interpolation=cv2.INTER_AREA)
    img[200:200 + small.shape[0], 300:300 + small.shape[1]] = small
    rgb = np.stack([img] * 3, -1)
    K = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1.0]])
    det = detect_marker(rgb, K, 0.10, ident=7)
    assert det and abs(det[0]["z"] - 1.5) / 1.5 < 0.04
