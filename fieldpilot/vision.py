"""Simple OpenCV preprocessing for robot camera images (no learned vision models).

- crop_to_panel: find the saturated-blue panel cells, take their bounding box, add margins
  (extra margin below, where ground vegetation would be), and crop. Falls back to the full
  image if no panel is found.
- encode_data_url: resize + JPEG + base64 data URL for the Decisions API.
- green_fraction: share of saturated-green pixels; used ONLY by the offline mock classifier.
"""
import base64

import cv2
import numpy as np

MAX_SIDE = 768


def _hsv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def panel_mask(rgb: np.ndarray) -> np.ndarray:
    h, s, v = cv2.split(_hsv(rgb))
    mask = ((h >= 95) & (h <= 130) & (s >= 140) & (v >= 25)).astype(np.uint8)
    # Close the thin grid lines so the cells merge into one panel blob.
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def crop_to_panel(rgb: np.ndarray, side=0.20, top=0.15, bottom=0.50, min_area_frac=0.01):
    """Return (crop, (x0, y0, x1, y1)). bbox is the full image when no panel blob is found."""
    H, W = rgb.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats(panel_mask(rgb))
    if n <= 1:
        return rgb, (0, 0, W, H)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[i]
    if area < min_area_frac * W * H:
        return rgb, (0, 0, W, H)
    x0, x1 = max(0, int(x - side * w)), min(W, int(x + w + side * w))
    y0, y1 = max(0, int(y - top * h)), min(H, int(y + h + bottom * h))
    return rgb[y0:y1, x0:x1], (x0, y0, x1, y1)


def encode_data_url(rgb: np.ndarray, max_side: int = MAX_SIDE, quality: int = 90) -> str:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def green_fraction(rgb: np.ndarray) -> float:
    h, s, v = cv2.split(_hsv(rgb))
    return float(((h >= 35) & (h <= 85) & (s >= 100) & (v >= 40)).mean())
