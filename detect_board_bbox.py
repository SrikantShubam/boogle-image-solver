import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def find_candidate_tiles(image_bgr):
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, (0, 0, 170), (179, 80, 255))
    cyan_mask = cv2.inRange(hsv, (70, 45, 75), (105, 255, 255))
    mask = cv2.bitwise_or(white_mask, cyan_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    img_area = h * w
    # Tiles are large; require at least 0.3% of image area to skip small UI noise.
    min_area = max(800, int(img_area * 0.003))
    max_area = int(img_area * 0.04)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = bw / float(bh)
        if circularity < 0.55 or not (0.7 <= aspect <= 1.3):
            continue

        cx = int(x + bw / 2)
        cy = int(y + bh / 2)
        if not (h * 0.25 <= cy <= h * 0.88):
            continue
        if not (w * 0.05 <= cx <= w * 0.95):
            continue

        radius = int(0.5 * (bw + bh) / 2)
        candidates.append((cx, cy, radius))

    # NMS: each tile has an inner white circle AND an outer blue border ring,
    # both detected separately. Keep only the largest of any near-duplicate pair.
    candidates.sort(key=lambda c: -c[2])
    nms = []
    for cx, cy, r in candidates:
        if not any(
            (cx - kx) ** 2 + (cy - ky) ** 2 < (max(r, kr) * 0.55) ** 2
            for kx, ky, kr in nms
        ):
            nms.append((cx, cy, r))
    return nms


def select_grid_points(candidates):
    if len(candidates) < 14:
        return None

    pts = np.array([[c[0], c[1], c[2]] for c in candidates], dtype=np.float32)
    centers = pts[:, :2]
    radii = pts[:, 2]

    best = None
    for n in (4, 5):
        if len(candidates) < n * n:
            continue

        x_vals = centers[:, 0].reshape(-1, 1).astype(np.float32)
        y_vals = centers[:, 1].reshape(-1, 1).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)

        _, x_labels, x_centroids = cv2.kmeans(
            x_vals, n, None, crit, 8, cv2.KMEANS_PP_CENTERS
        )
        _, y_labels, y_centroids = cv2.kmeans(
            y_vals, n, None, crit, 8, cv2.KMEANS_PP_CENTERS
        )

        x_order = np.argsort(x_centroids[:, 0])
        y_order = np.argsort(y_centroids[:, 0])
        x_rank = {int(lbl): rank for rank, lbl in enumerate(x_order)}
        y_rank = {int(lbl): rank for rank, lbl in enumerate(y_order)}

        cells = {}
        for i in range(len(candidates)):
            col = x_rank[int(x_labels[i][0])]
            row = y_rank[int(y_labels[i][0])]
            key = (row, col)

            cx = centers[i, 0]
            cy = centers[i, 1]
            tx = x_centroids[x_order[col], 0]
            ty = y_centroids[y_order[row], 0]
            d2 = float((cx - tx) ** 2 + (cy - ty) ** 2)

            if key not in cells or d2 < cells[key][0]:
                cells[key] = (d2, i)

        coverage = len(cells) / float(n * n)
        if coverage < 0.92:
            continue

        selected = []
        for row in range(n):
            for col in range(n):
                if (row, col) not in cells:
                    selected = None
                    break
                idx = cells[(row, col)][1]
                selected.append((int(centers[idx, 0]), int(centers[idx, 1]), int(radii[idx])))
            if selected is None:
                break

        if selected is None:
            continue

        # Slight penalty for larger grids so 4x4 wins when coverage ties.
        score = coverage - n * 0.001
        if best is None or score > best[0]:
            best = (score, n, selected)

    return best


def detect_bbox(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
      raise RuntimeError(f"Could not read image: {image_path}")

    result = select_grid_points(find_candidate_tiles(image))
    if result is None:
        raise RuntimeError("Could not detect a 4x4 or 5x5 board.")

    _, n, points = result
    xs = [x for x, _, r in points for _ in [0]]
    ys = [y for _, y, r in points for _ in [0]]
    rs = [r for _, _, r in points]
    pad = int(max(rs) * 1.15)

    h, w = image.shape[:2]
    left = max(0, min(xs) - pad)
    top = max(0, min(ys) - pad)
    right = min(w, max(xs) + pad)
    bottom = min(h, max(ys) + pad)

    return {
        "grid_size": n,
        "left": int(left),
        "top": int(top),
        "width": int(right - left),
        "height": int(bottom - top),
        "points": [{"x": int(x), "y": int(y), "r": int(r)} for x, y, r in points],
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: detect_board_bbox.py <image_path>")
    data = detect_bbox(Path(sys.argv[1]))
    print(json.dumps(data))


if __name__ == "__main__":
    main()
