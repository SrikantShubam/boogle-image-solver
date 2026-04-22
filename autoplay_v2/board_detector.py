"""Auto-detect the Plato Crosswords board from a screenshot.

Uses HSV colour masking to find white/cyan tile circles, NMS to remove
duplicates, and K-means clustering to arrange them into a 4×4 or 5×5 grid.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from autoplay_v2.models import DetectedBoard, DetectedTile

# HSV bounds for game tile circles.
_CYAN_LOWER = (70, 45, 75)
_CYAN_UPPER = (105, 255, 255)
_WHITE_LOWER = (0, 0, 170)
_WHITE_UPPER = (179, 80, 255)


def _find_candidate_circles(
    image_bgr: np.ndarray,
    debug_dir: Optional[Path] = None,
) -> List[Tuple[int, int, int]]:
    """Detect white and cyan circles that could be game tiles.

    Returns a list of (centre_x, centre_y, radius).
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, _WHITE_LOWER, _WHITE_UPPER)
    cyan_mask = cv2.inRange(hsv, _CYAN_LOWER, _CYAN_UPPER)
    mask = cv2.bitwise_or(white_mask, cyan_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "01_hsv_mask.png"), mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = h * w
    min_area = max(400, int(img_area * 0.001))
    max_area = int(img_area * 0.04)

    # Adaptive y-bound: tall portrait screenshots (h/w > 1.5) are full device
    # captures with avatars/nav UI → clip top 22% and bottom 12%. Short or
    # square images are pre-cropped to the board → keep lenient bounds.
    if h / max(1, w) > 1.5:
        y_lo_frac, y_hi_frac = 0.22, 0.88
    else:
        y_lo_frac, y_hi_frac = 0.05, 0.95
    candidates: List[Tuple[int, int, int]] = []
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

        # Only consider tiles in the plausible game area (avoid status bar,
        # avatars at top, input box / nav at bottom).
        if not (h * y_lo_frac <= cy <= h * y_hi_frac):
            continue
        if not (w * 0.02 <= cx <= w * 0.98):
            continue

        radius = int(0.5 * (bw + bh) / 2)
        candidates.append((cx, cy, radius))

    hsv_count = len(candidates)
    print(f"[board_detector] HSV candidates: {hsv_count}")

    # Supplemental Fallback: Structural Hough Circles
    # This prevents total failure if the user's phone has a screen filter (e.g. night light)
    # that drastically alters the HSV profile of the "white/cyan" tiles.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    _min_r = int(min(w, h) * 0.04)
    _max_r = int(min(w, h) * 0.18)
    hough = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=_min_r * 2,
        param1=50, param2=30, minRadius=_min_r, maxRadius=_max_r
    )
    if hough is not None:
        for _x, _y, _r in np.round(hough[0, :]).astype(int).tolist():
            if h * y_lo_frac <= _y <= h * y_hi_frac and w * 0.02 <= _x <= w * 0.98:
                candidates.append((_x, _y, _r))

    hough_additions = len(candidates) - hsv_count
    print(f"[board_detector] Hough additions: {hough_additions}")
    if debug_dir is not None and hough is not None:
        hough_vis = image_bgr.copy()
        for _x, _y, _r in np.round(hough[0, :]).astype(int).tolist():
            if h * y_lo_frac <= _y <= h * y_hi_frac and w * 0.02 <= _x <= w * 0.98:
                cv2.circle(hough_vis, (_x, _y), _r, (0, 255, 0), 2)
                cv2.circle(hough_vis, (_x, _y), 2, (0, 0, 255), 3)
        cv2.imwrite(str(debug_dir / "03_hough.png"), hough_vis)

    # Non-maximum suppression: keep only the larger of concentric pairs.
    candidates.sort(key=lambda c: -c[2])  # largest radius first
    nms: List[Tuple[int, int, int]] = []
    for cx, cy, r in candidates:
        if not any(
            (cx - kx) ** 2 + (cy - ky) ** 2 < (max(r, kr) * 0.55) ** 2
            for kx, ky, kr in nms
        ):
            nms.append((cx, cy, r))

    print(f"[board_detector] After NMS: {len(nms)} candidates")
    if debug_dir is not None:
        cand_vis = image_bgr.copy()
        for cx, cy, r in nms:
            cv2.circle(cand_vis, (cx, cy), r, (255, 0, 0), 2)
            cv2.circle(cand_vis, (cx, cy), 3, (0, 0, 255), -1)
        cv2.imwrite(str(debug_dir / "02_candidates.png"), cand_vis)

    return nms


def _cluster_into_grid(
    candidates: List[Tuple[int, int, int]],
    force_grid_size: Optional[int] = None,
) -> Optional[Tuple[float, int, List[Tuple[int, int, int]]]]:
    """Cluster candidate circles into a 4×4 or 5×5 grid using K-means.

    Prefers the **larger** grid when both have full coverage (e.g. 25
    candidates → 5×5 wins over 4×4).

    Returns ``(confidence, grid_size, row_major_points)`` or ``None``.
    """
    if len(candidates) < 14 and not force_grid_size:
        return None

    pts = np.array(candidates, dtype=np.float32)
    centres = pts[:, :2]
    radii = pts[:, 2]

    best = None
    sizes_to_try = [force_grid_size] if force_grid_size else (5, 4)
    for n in sizes_to_try:
        if len(candidates) < n * n and not force_grid_size:
            continue

        # If not enough candidates for forced grid, just duplicate/fail gracefully
        actual_n = min(len(candidates), n)
        if actual_n < 2:
            continue

        x_vals = centres[:, 0].reshape(-1, 1).astype(np.float32)
        y_vals = centres[:, 1].reshape(-1, 1).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)

        # Catch kmeans failure when trying to force too many clusters (n > len(candidates))
        try:
            _, x_labels, x_centroids = cv2.kmeans(
                x_vals, min(n, len(x_vals)), None, crit, 8, cv2.KMEANS_PP_CENTERS,
            )
            _, y_labels, y_centroids = cv2.kmeans(
                y_vals, min(n, len(y_vals)), None, crit, 8, cv2.KMEANS_PP_CENTERS,
            )
        except Exception:
            continue

        x_order = np.argsort(x_centroids[:, 0])
        y_order = np.argsort(y_centroids[:, 0])
        x_rank = {int(lbl): rank for rank, lbl in enumerate(x_order)}
        y_rank = {int(lbl): rank for rank, lbl in enumerate(y_order)}

        # Map every candidate into the nearest cell.
        cells: dict[tuple[int, int], tuple[float, int]] = {}
        for i in range(len(candidates)):
            col = x_rank[int(x_labels[i][0])]
            row = y_rank[int(y_labels[i][0])]
            key = (row, col)
            cx, cy = centres[i, 0], centres[i, 1]
            tx = x_centroids[x_order[min(col, len(x_order)-1)], 0]
            ty = y_centroids[y_order[min(row, len(y_order)-1)], 0]
            d2 = float((cx - tx) ** 2 + (cy - ty) ** 2)
            if key not in cells or d2 < cells[key][0]:
                cells[key] = (d2, i)

        coverage = len(cells) / float(n * n)
        if coverage < 0.92 and not force_grid_size:
            print(f"[board_detector] {n}×{n} grid: coverage={coverage:.1%} (need ≥92%) — skipped")
            continue

        selected: Optional[List[Tuple[int, int, int]]] = []
        for row in range(n):
            for col in range(n):
                if (row, col) not in cells:
                    selected = None
                    break
                idx = cells[(row, col)][1]
                selected.append((
                    int(centres[idx, 0]),
                    int(centres[idx, 1]),
                    int(radii[idx]),
                ))
            if selected is None:
                break

        if selected is None and not force_grid_size:
            continue
            
        # If forcing grid, but we missed some nodes, just duplicate nearest ones so it doesn't crash the script later
        if selected is None and force_grid_size:
            selected = []
            for r in range(n):
                for c in range(n):
                    if (r, c) in cells:
                        idx = cells[(r, c)][1]
                        selected.append((int(centres[idx, 0]), int(centres[idx, 1]), int(radii[idx])))
                    else:
                        selected.append((0, 0, 10))

        tiles_used = n * n
        score = coverage + tiles_used * 0.01
        print(f"[board_detector] {n}×{n} grid: coverage={coverage:.1%} — accepted")
        if best is None or score > best[0]:
            best = (score, n, selected)

    return best


def detect_board(
    image_bgr: np.ndarray,
    debug_dir: Optional[Path] = None,
    force_grid_size: Optional[int] = None,
) -> Optional[DetectedBoard]:
    """Detect the Plato Crosswords board in *image_bgr*.

    Returns a :class:`DetectedBoard` with tile centres in the image's pixel
    coordinate system, or ``None`` when no board is found.
    """
    candidates = _find_candidate_circles(image_bgr, debug_dir=debug_dir)
    result = _cluster_into_grid(candidates, force_grid_size=force_grid_size)
    if result is None:
        return None

    _, grid_size, points = result

    # Bounding box with padding.
    xs = [x for x, _, _ in points]
    ys = [y for _, y, _ in points]
    rs = [r for _, _, r in points]
    pad = int(max(rs) * 1.15), int(max(rs) * 1.15)

    h, w = image_bgr.shape[:2]
    roi_left = max(0, min(xs) - pad[0])
    roi_top = max(0, min(ys) - pad[1])
    roi_right = min(w, max(xs) + pad[0])
    roi_bottom = min(h, max(ys) + pad[1])

    tiles: List[DetectedTile] = []
    for idx, (cx, cy, radius) in enumerate(points):
        row, col = divmod(idx, grid_size)
        tiles.append(DetectedTile(
            index=idx, row=row, col=col,
            cx=cx, cy=cy, radius=radius,
        ))

    return DetectedBoard(
        grid_size=grid_size,
        tiles=tiles,
        roi_left=roi_left,
        roi_top=roi_top,
        roi_width=roi_right - roi_left,
        roi_height=roi_bottom - roi_top,
    )


def is_bonus_tile(
    image_bgr: np.ndarray,
    tile: DetectedTile,
) -> bool:
    """Check whether *tile* is the cyan/green bonus tile."""
    r = int(tile.radius * 0.6)
    h, w = image_bgr.shape[:2]
    y0 = max(0, tile.cy - r)
    y1 = min(h, tile.cy + r)
    x0 = max(0, tile.cx - r)
    x1 = min(w, tile.cx + r)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _CYAN_LOWER, _CYAN_UPPER)
    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    return ratio >= 0.12
