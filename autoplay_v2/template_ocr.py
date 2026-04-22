"""Local per-character template-matching OCR for Plato Crosswords.

Pipeline:
  1. On first call, synthesise 26 uppercase + 26 lowercase templates by
     rendering Poppins SemiBold (identified as the game's font).
  2. For each detected tile crop, binarise via Otsu, find ink bbox, decide
     whether it's a single letter or digraph by bbox aspect ratio.
  3. Normalise crop (or each half for digraphs) to a fixed canvas and pick
     the best-matching template via SSIM.

Zero network calls. Runs in ~5 ms per tile on CPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from autoplay_v2.models import DetectedBoard, OCRBoardResult, OCRTileResult


def _binary_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 - normalised Hamming distance on binary images (stand-in for SSIM)."""
    return 1.0 - float(np.count_nonzero((a > 0) ^ (b > 0))) / float(a.size)

CANVAS = 96
UPPER = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
LOWER = [chr(c) for c in range(ord("a"), ord("z") + 1)]

_FONT_PATH = Path(__file__).resolve().parents[1] / "tools" / "font_id" / "fonts" / "Poppins-SemiBold.ttf"

# Known digraphs from the game (post-classification sanity filter only)
_DIGRAPHS = {
    "TH", "HE", "QU", "IN", "ER", "AN", "RE", "ON", "AT", "EN",
    "ND", "TI", "ES", "OR", "TE", "OF", "ED", "IS", "IT", "AL",
    "AR", "ST", "TO", "NT", "NG", "SE", "HA", "AS", "OU", "IO",
    "LE", "VE", "CO", "ME", "DE", "HI", "RI", "RO", "IC", "NE",
    "EA", "RA", "CE", "LI", "CH", "LL", "BE", "MA", "SI", "OM",
    "UR",
}
_GAME_DIGRAPHS = {"TH", "HE", "QU", "IN", "ER", "AN"}


# ---------------------------------------------------------------------------
# Template generation (lazy, cached)
# ---------------------------------------------------------------------------

_UPPER_TEMPLATES: Optional[Dict[str, np.ndarray]] = None
_LOWER_TEMPLATES: Optional[Dict[str, np.ndarray]] = None


def _binarize_and_center(gray: np.ndarray, canvas: int = CANVAS, already_binary: bool = False) -> np.ndarray:
    """Threshold dark letter pixels (glyph=255), crop to bbox, resize to ~75%, center."""
    if gray.size == 0:
        return np.zeros((canvas, canvas), dtype=np.uint8)
    if already_binary:
        th = gray
    else:
        # Fixed threshold: letters are dark (< 110) on white tiles; orange bg is ~180.
        th = (gray < 140).astype(np.uint8) * 255

    ys, xs = np.where(th > 0)
    if len(xs) == 0:
        return np.zeros((canvas, canvas), dtype=np.uint8)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    crop = th[y0:y1, x0:x1]

    h, w = crop.shape
    target = int(canvas * 0.75)
    scale = target / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    out = np.zeros((canvas, canvas), dtype=np.uint8)
    oy = (canvas - new_h) // 2
    ox = (canvas - new_w) // 2
    out[oy:oy + new_h, ox:ox + new_w] = resized
    return out


def _render(letter: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Render a single glyph, pre-binarise (glyph=255), thicken, then normalise."""
    img = Image.new("L", (CANVAS * 3, CANVAS * 3), 0)
    draw = ImageDraw.Draw(img)
    draw.text((CANVAS * 1.5, CANVAS * 1.5), letter, fill=255, font=font, anchor="mm")
    arr = np.array(img)
    binary = (arr > 127).astype(np.uint8) * 255
    return _binarize_and_center(binary, CANVAS, already_binary=True)


_EMPIRICAL_DIR = Path(__file__).resolve().parents[1] / "tools" / "empirical_templates"


def _load_empirical(case_dir: str, letters: list[str]) -> Dict[str, np.ndarray]:
    """Load empirical templates (averaged from real game tiles)."""
    out: Dict[str, np.ndarray] = {}
    d = _EMPIRICAL_DIR / case_dir
    if not d.exists():
        return out
    for L in letters:
        name = L if case_dir == "upper" else L.lower()
        p = d / f"{name}.png"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                out[L] = img
    return out


def _build_templates() -> None:
    global _UPPER_TEMPLATES, _LOWER_TEMPLATES
    if _UPPER_TEMPLATES is not None and _LOWER_TEMPLATES is not None:
        return
    if not _FONT_PATH.exists():
        raise RuntimeError(f"Poppins SemiBold font not found at {_FONT_PATH}")
    font = ImageFont.truetype(str(_FONT_PATH), 130)
    # Synthetic Poppins fallback for letters without empirical samples.
    synth_upper = {L: _render(L, font) for L in UPPER}
    synth_lower = {L: _render(L, font) for L in LOWER}
    # Prefer empirical templates (averaged from real game screenshots).
    emp_upper = _load_empirical("upper", UPPER)
    emp_lower = _load_empirical("lower", LOWER)
    _UPPER_TEMPLATES = {**synth_upper, **emp_upper}
    _LOWER_TEMPLATES = {**synth_lower, **emp_lower}


# ---------------------------------------------------------------------------
# Tile processing
# ---------------------------------------------------------------------------

def _regularise_board(board: DetectedBoard) -> DetectedBoard:
    """Reconstruct a regular n×n grid from raw tile detections.

    Discards outliers (wrong size = non-tile UI circles like avatars), then
    uses least-squares on ALL remaining (cx, row), (cy, col) pairs re-derived
    from actual tile positions — ignoring any wrong row/col labels from the
    k-means step inside board_detector.
    """
    n = board.grid_size
    # 1. Pull raw centres, discarding (0,0) placeholders
    raw = [(t.cx, t.cy, t.radius) for t in board.tiles if not (t.cx == 0 and t.cy == 0)]
    if len(raw) < n * n // 2:
        # too few — keep original labels
        return board

    # 2. Filter by radius — avatars, nav buttons have different sizes
    radii = np.array([r for _, _, r in raw], dtype=np.float64)
    med_r = float(np.median(radii))
    mad_r = float(np.median(np.abs(radii - med_r))) + 1e-6
    filt = [(cx, cy, r) for cx, cy, r in raw if abs(r - med_r) <= 4.0 * mad_r]
    if len(filt) < n * n // 2:
        filt = raw  # relax if too strict

    # 3. Re-cluster into n rows via k-means on y, and n cols on x, IGNORING labels.
    xs_arr = np.array([[cx] for cx, _, _ in filt], dtype=np.float32)
    ys_arr = np.array([[cy] for _, cy, _ in filt], dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.1)
    _, _, x_cent = cv2.kmeans(xs_arr, n, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    _, _, y_cent = cv2.kmeans(ys_arr, n, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    col_centres = sorted(int(round(c[0])) for c in x_cent)
    row_centres = sorted(int(round(c[0])) for c in y_cent)

    # 4. Robust spacing: for each axis, use k-means centres' MEDIAN gap as
    # the "true" grid pitch, then re-fit a rigid grid using all raw tile
    # positions that are close to any predicted grid line.
    def refit_axis(centres: List[int], all_vals: List[float]) -> List[int]:
        if len(centres) < 2:
            return centres
        gaps = [centres[i + 1] - centres[i] for i in range(len(centres) - 1)]
        med = float(np.median(gaps))
        # Find offset: project each raw value onto grid with spacing=med,
        # anchored at centres[0]; collect residuals → median-offset gives origin.
        anchor = float(centres[0])
        residuals = [(v - anchor) - round((v - anchor) / med) * med for v in all_vals]
        offset = float(np.median(residuals))
        origin = anchor + offset
        return [int(round(origin + i * med)) for i in range(len(centres))]

    col_centres = refit_axis(col_centres, [cx for cx, _, _ in filt])
    row_centres = refit_axis(row_centres, [cy for _, cy, _ in filt])

    # 4. Rebuild tiles as the intersection grid
    from autoplay_v2.models import DetectedTile
    new_tiles = []
    for r in range(n):
        for c in range(n):
            idx = r * n + c
            new_tiles.append(DetectedTile(
                index=idx, row=r, col=c,
                cx=col_centres[c], cy=row_centres[r],
                radius=int(med_r),
            ))

    pad = int(med_r * 1.15)
    return DetectedBoard(
        grid_size=n, tiles=new_tiles,
        roi_left=max(0, col_centres[0] - pad),
        roi_top=max(0, row_centres[0] - pad),
        roi_width=col_centres[-1] - col_centres[0] + 2 * pad,
        roi_height=row_centres[-1] - row_centres[0] + 2 * pad,
    )


def _regularise_board_OLD(board: DetectedBoard) -> DetectedBoard:
    """DEPRECATED — kept for reference."""
    n = board.grid_size
    col_xs: List[List[int]] = [[] for _ in range(n)]
    row_ys: List[List[int]] = [[] for _ in range(n)]
    for t in board.tiles:
        if t.cx == 0 and t.cy == 0:
            continue
        col_xs[t.col].append(t.cx)
        row_ys[t.row].append(t.cy)

    # Fit a global rigid grid via least squares on ALL valid detections:
    #   cx = origin_x + col * spacing_x
    #   cy = origin_y + row * spacing_y
    cols = np.array([t.col for t in board.tiles if not (t.cx == 0 and t.cy == 0)], dtype=np.float64)
    rows = np.array([t.row for t in board.tiles if not (t.cx == 0 and t.cy == 0)], dtype=np.float64)
    cxs = np.array([t.cx for t in board.tiles if not (t.cx == 0 and t.cy == 0)], dtype=np.float64)
    cys = np.array([t.cy for t in board.tiles if not (t.cx == 0 and t.cy == 0)], dtype=np.float64)

    if len(cxs) < 4:
        # fall back to medians
        col_centres = [int(np.median(v)) if v else 0 for v in col_xs]
        row_centres = [int(np.median(v)) if v else 0 for v in row_ys]
    else:
        # Robust fit: trim outliers via two-pass least squares
        def fit_axis(indices: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
            slope, intercept = np.polyfit(indices, values, 1)
            residuals = values - (slope * indices + intercept)
            mad = np.median(np.abs(residuals - np.median(residuals))) + 1e-6
            keep = np.abs(residuals - np.median(residuals)) < 3.0 * mad
            if keep.sum() >= 4:
                slope, intercept = np.polyfit(indices[keep], values[keep], 1)
            return float(slope), float(intercept)

        sx, ox = fit_axis(cols, cxs)
        sy, oy = fit_axis(rows, cys)
        col_centres = [int(round(ox + i * sx)) for i in range(n)]
        row_centres = [int(round(oy + i * sy)) for i in range(n)]

    # Rebuild tiles, preserving (row, col, index), replacing cx/cy with grid intersection.
    # Reuse median radius for consistency.
    radii = [t.radius for t in board.tiles if t.radius > 15]
    med_r = int(np.median(radii)) if radii else 40
    from autoplay_v2.models import DetectedTile  # local import to avoid cycle
    new_tiles = []
    for t in board.tiles:
        new_tiles.append(DetectedTile(
            index=t.index, row=t.row, col=t.col,
            cx=col_centres[t.col], cy=row_centres[t.row],
            radius=med_r,
        ))
    # Build a replacement board
    xs = [c for c in col_centres]
    ys = [c for c in row_centres]
    pad = int(med_r * 1.15)
    h_w = max(xs) - min(xs) + 2 * pad
    h_h = max(ys) - min(ys) + 2 * pad
    return DetectedBoard(
        grid_size=n, tiles=new_tiles,
        roi_left=max(0, min(xs) - pad), roi_top=max(0, min(ys) - pad),
        roi_width=h_w, roi_height=h_h,
    )


def _board_letter_half(board: DetectedBoard) -> int:
    """Compute half-width of a letter-sized crop from board tile spacing."""
    if board.grid_size < 2 or len(board.tiles) < 2:
        return max(15, int(np.median([t.radius for t in board.tiles]) * 0.6))
    xs = np.array([t.cx for t in board.tiles])
    ys = np.array([t.cy for t in board.tiles])
    spacing_x = (xs.max() - xs.min()) / (board.grid_size - 1)
    spacing_y = (ys.max() - ys.min()) / (board.grid_size - 1)
    spacing = (spacing_x + spacing_y) / 2.0
    # Letter fills ~35-45% of tile spacing; pad to ~45% for safety (digraphs wider)
    return max(15, int(spacing * 0.30))


def _extract_tile_gray(
    image_bgr: np.ndarray,
    tile,
    half: int,
    *,
    search_scale: float = 1.8,
    use_ink_centroid: bool = True,
    center_dx: int = 0,
    center_dy: int = 0,
) -> np.ndarray:
    """Crop a tight region around the letter.

    Uses a larger SEARCH window (half * 1.6) around the nominal tile centre to
    find dark pixels (letter ink), computes their centroid, then returns a
    half-sized crop centred on that centroid. Makes the pipeline robust to
    noisy board-detection centres.
    """
    h, w = image_bgr.shape[:2]
    search = max(1, int(half * search_scale))
    sy0 = max(0, tile.cy - search); sy1 = min(h, tile.cy + search)
    sx0 = max(0, tile.cx - search); sx1 = min(w, tile.cx + search)
    search_crop = image_bgr[sy0:sy1, sx0:sx1]
    if search_crop.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    gray_search = cv2.cvtColor(search_crop, cv2.COLOR_BGR2GRAY)
    # Only consider dark pixels that are also surrounded by bright (tile interior).
    dark = (gray_search < 110).astype(np.uint8)
    bright = (gray_search > 200).astype(np.uint8)
    # Dilate bright to form "tile interior" and require dark pixel to be near it.
    bright_dilated = cv2.dilate(bright, np.ones((9, 9), np.uint8))
    ink = dark & bright_dilated
    ys, xs = np.where(ink > 0)
    if use_ink_centroid and len(xs) > 5:
        # BBOX-center (not median): preserves asymmetric glyphs like J (hook),
        # Q (tail), X (thin diagonals). Median centroid gets pulled toward
        # the densest stroke and clips the opposite end of the glyph.
        cy_global = sy0 + int(np.median(ys))
        cx_global = sx0 + int(np.median(xs))
    else:
        cx_global, cy_global = tile.cx, tile.cy

    cy_global += int(center_dy)
    cx_global += int(center_dx)
    y0 = max(0, cy_global - half); y1 = min(h, cy_global + half)
    x0 = max(0, cx_global - half); x1 = min(w, cx_global + half)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _classify(slot: np.ndarray, templates: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """Pick the template with highest shape-match score.

    Combines pixel IoU (shape overlap — strong discriminator for binary glyphs)
    with SSIM (structural — handles small misalignments). Weighted 0.7/0.3.
    """
    # Shape-based matching via edge/contour chamfer distance.
    # Stroke-thickness invariant: we extract the letter's outline edges,
    # then compute the symmetric chamfer distance between slot-edges and
    # template-edges. This penalises shape differences (e.g. E's middle bar
    # vs C's right opening) without being fooled by thickness mismatches.
    slot_edge = cv2.Canny(slot, 50, 150)
    slot_ink_bool = slot > 0
    slot_inv = (255 - slot).astype(np.uint8)
    slot_dt = cv2.distanceTransform(slot_inv, cv2.DIST_L2, 3)
    best_letter = "?"
    best_score = -1e9
    e_a = slot_edge > 0
    for letter, tpl in templates.items():
        tpl_edge = cv2.Canny(tpl, 50, 150)
        tpl_inv = (255 - tpl).astype(np.uint8)
        tpl_dt = cv2.distanceTransform(tpl_inv, cv2.DIST_L2, 3)
        e_b = tpl_edge > 0
        if e_a.any() and e_b.any():
            chamfer = (float(tpl_dt[e_a].mean()) + float(slot_dt[e_b].mean())) / 2.0
        else:
            chamfer = 50.0
        b_bool = tpl > 0
        inter = int(np.logical_and(slot_ink_bool, b_bool).sum())
        union = int(slot_ink_bool.sum()) + int(b_bool.sum()) - inter
        iou = inter / max(1, union)
        score = -chamfer + 1.0 * iou
        if score > best_score:
            best_score = score
            best_letter = letter
    return best_letter, max(0.0, min(1.0, (best_score + 10) / 12))


def _classify_scores(slot: np.ndarray, templates: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Return per-template confidence scores for a normalized slot."""
    slot_edge = cv2.Canny(slot, 50, 150)
    slot_ink_bool = slot > 0
    slot_inv = (255 - slot).astype(np.uint8)
    slot_dt = cv2.distanceTransform(slot_inv, cv2.DIST_L2, 3)
    e_a = slot_edge > 0
    out: Dict[str, float] = {}
    for letter, tpl in templates.items():
        tpl_edge = cv2.Canny(tpl, 50, 150)
        tpl_inv = (255 - tpl).astype(np.uint8)
        tpl_dt = cv2.distanceTransform(tpl_inv, cv2.DIST_L2, 3)
        e_b = tpl_edge > 0
        if e_a.any() and e_b.any():
            chamfer = (float(tpl_dt[e_a].mean()) + float(slot_dt[e_b].mean())) / 2.0
        else:
            chamfer = 50.0
        b_bool = tpl > 0
        inter = int(np.logical_and(slot_ink_bool, b_bool).sum())
        union = int(slot_ink_bool.sum()) + int(b_bool.sum()) - inter
        iou = inter / max(1, union)
        raw = -chamfer + 1.0 * iou
        out[letter] = max(0.0, min(1.0, (raw + 10) / 12))
    return out


def _ink_bbox(gray: np.ndarray) -> Optional[Tuple[int, int, int, int, np.ndarray]]:
    """Return (x0, y0, x1, y1, binary) for the ink region, or None."""
    if gray.size == 0:
        return None
    th = (gray < 140).astype(np.uint8) * 255
    ys, xs = np.where(th > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, th


def _glyph_features(gray: np.ndarray) -> Dict[str, float]:
    """Simple structural features for ambiguity tie-breaks."""
    bb = _ink_bbox(gray)
    if bb is None:
        return {
            "components": 0.0,
            "holes": 0.0,
            "top_width": 0.0,
            "bot_width": 0.0,
            "top_ink": 0.0,
            "bot_ink": 0.0,
        }
    x0, y0, x1, y1, th = bb
    strip = th[y0:y1, x0:x1]
    if strip.size == 0:
        return {
            "components": 0.0,
            "holes": 0.0,
            "top_width": 0.0,
            "bot_width": 0.0,
            "top_ink": 0.0,
            "bot_ink": 0.0,
        }
    bin_img = (strip > 0).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(bin_img)
    components = max(0, int(num_labels) - 1)

    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None and len(contours) > 0:
        for i in range(len(contours)):
            parent = hierarchy[0][i][3]
            if parent >= 0:
                holes += 1

    h = strip.shape[0]
    top = strip[: max(1, int(h * 0.25)), :]
    bot = strip[max(0, int(h * 0.75)) :, :]
    top_width = float(np.count_nonzero(top.sum(axis=0) > 0))
    bot_width = float(np.count_nonzero(bot.sum(axis=0) > 0))
    top_ink = float(np.count_nonzero(top > 0))
    bot_ink = float(np.count_nonzero(bot > 0))
    return {
        "components": float(components),
        "holes": float(holes),
        "top_width": top_width,
        "bot_width": bot_width,
        "top_ink": top_ink,
        "bot_ink": bot_ink,
    }


def _split_slots(gray: np.ndarray) -> List[np.ndarray]:
    """Return list of already-binarised slot crops (glyph=255, bg=0).

    A tile is a digraph if EITHER:
      (a) aspect ratio > 1.15 (uppercase dominant tiles), OR
      (b) the ink bbox contains >=2 horizontally-separated connected
          components (catches narrow digraphs like "In" where aspect ≈ 1.1).
    """
    bb = _ink_bbox(gray)
    if bb is None:
        return [np.zeros((1, 1), dtype=np.uint8)]
    x0, y0, x1, y1, th = bb
    w = x1 - x0
    h = y1 - y0
    aspect = w / max(1, h)
    strip = th[y0:y1, x0:x1]

    # Connected-components check: project ink horizontally, count horizontal gaps.
    if aspect < 1.15:
        # Look for 2+ distinct vertical-strokes via horizontal projection.
        # If the column sum drops to 0 somewhere in the middle third, split.
        col_sum = strip.sum(axis=0)
        mid_lo = int(len(col_sum) * 0.30)
        mid_hi = int(len(col_sum) * 0.70)
        if mid_hi > mid_lo and col_sum[mid_lo:mid_hi].min() == 0:
            zeros = np.where(col_sum[mid_lo:mid_hi] == 0)[0]
            mid = mid_lo + int(zeros[len(zeros) // 2])
            left, right = strip[:, :mid], strip[:, mid:]
            # Only split if both sides have substantial ink (avoid splitting serifs).
            if (left > 0).sum() > 20 and (right > 0).sum() > 20:
                return [left, right]
        return [strip]

    col_sum = strip.sum(axis=0)
    lo = int(len(col_sum) * 0.40)
    hi = int(len(col_sum) * 0.68)
    if hi <= lo + 1:
        mid = len(col_sum) // 2
    else:
        mid = lo + int(np.argmin(col_sum[lo:hi]))
    return [strip[:, :mid], strip[:, mid:]]


def classify_tile(
    image_bgr: np.ndarray,
    tile,
    half: Optional[int] = None,
    *,
    search_scale: float = 1.8,
    use_ink_centroid: bool = True,
) -> Tuple[str, float]:
    """Classify a single detected tile. Returns (token, confidence).

    Token is either one uppercase letter or an uppercase+lowercase digraph
    (game displays digraphs as e.g. 'An', but we return them as 'AN').
    """
    _build_templates()
    if half is None:
        half = max(15, int(tile.radius * 0.6))
    gray = _extract_tile_gray(
        image_bgr,
        tile,
        half,
        search_scale=search_scale,
        use_ink_centroid=use_ink_centroid,
    )
    slots = _split_slots(gray)

    if len(slots) == 1:
        norm = _binarize_and_center(slots[0], already_binary=True)
        letter, score = _classify(norm, _UPPER_TEMPLATES)  # type: ignore[arg-type]
        return letter, score

    # Digraph
    upper_norm = _binarize_and_center(slots[0], already_binary=True)
    lower_norm = _binarize_and_center(slots[1], already_binary=True)
    up_letter, up_score = _classify(upper_norm, _UPPER_TEMPLATES)  # type: ignore[arg-type]
    # Constrain lowercase search to letters that form valid digraphs with up_letter.
    valid_seconds = {d[1] for d in _GAME_DIGRAPHS if d[0] == up_letter}
    if valid_seconds:
        restricted = {L: _LOWER_TEMPLATES[L.lower()] for L in valid_seconds if L.lower() in _LOWER_TEMPLATES}  # type: ignore[index]
        if restricted:
            lo_letter, lo_score = _classify(lower_norm, restricted)
            lo_letter = lo_letter.lower()
        else:
            lo_letter, lo_score = _classify(lower_norm, _LOWER_TEMPLATES)  # type: ignore[arg-type]
    else:
        lo_letter, lo_score = _classify(lower_norm, _LOWER_TEMPLATES)  # type: ignore[arg-type]
    token = (up_letter + lo_letter.upper())
    score = (up_score + lo_score) / 2
    if token not in _GAME_DIGRAPHS:
        norm = _binarize_and_center(gray)
        s_letter, s_score = _classify(norm, _UPPER_TEMPLATES)  # type: ignore[arg-type]
        if s_score > score:
            return s_letter, s_score
    return token, score


_AMBIGUOUS_ALTS: Dict[str, List[str]] = {
    "RI": ["IN"],
    "BE": ["IN"],
    "I": ["Q", "J"],
    "N": ["Q", "IN"],
    "B": ["U"],
    "A": ["Z", "X"],
    "F": ["V"],
    "Q": ["I", "N"],
    "U": ["B"],
    "J": ["I"],
    "Z": ["A"],
    "X": ["A"],
    "V": ["F"],
    "M": ["IN"],
    "IN": ["M"],
}


def _score_token_on_gray(gray: np.ndarray, token: str) -> float:
    """Score a specific token against a gray tile crop."""
    token = token.upper()
    bb = _ink_bbox(gray)
    if bb is None:
        return -1.0
    x0, y0, x1, y1, th = bb
    strip = th[y0:y1, x0:x1]
    if strip.size == 0:
        return -1.0

    # Single-letter candidate
    if len(token) == 1:
        norm = _binarize_and_center(strip, already_binary=True)
        scores = _classify_scores(norm, _UPPER_TEMPLATES)  # type: ignore[arg-type]
        return float(scores.get(token, -1.0))

    # Digraph candidate
    if len(token) == 2:
        slots = _split_slots(gray)
        if len(slots) < 2:
            col_sum = strip.sum(axis=0)
            mid = len(col_sum) // 2
            slots = [strip[:, :mid], strip[:, mid:]]
        if len(slots) < 2:
            return -1.0
        up = _binarize_and_center(slots[0], already_binary=True)
        lo = _binarize_and_center(slots[1], already_binary=True)
        up_scores = _classify_scores(up, _UPPER_TEMPLATES)  # type: ignore[arg-type]
        lo_scores = _classify_scores(lo, _LOWER_TEMPLATES)  # type: ignore[arg-type]
        u = token[0]
        l = token[1].lower()
        if u not in up_scores or l not in lo_scores:
            return -1.0
        dig_penalty = 0.0 if token in _GAME_DIGRAPHS else 0.25
        return float((up_scores[u] + lo_scores[l]) / 2.0 - dig_penalty)
    return -1.0


def _maybe_refine_ambiguous_tile(
    image_bgr: np.ndarray,
    tile,
    *,
    base_token: str,
    base_score: float,
    half: int,
    search_scale: float,
    use_ink_centroid: bool,
) -> Tuple[str, float, bool]:
    tok = base_token.upper()
    if tok not in _AMBIGUOUS_ALTS and base_score >= 0.80:
        return base_token, base_score, False

    candidate_tokens = [tok]
    candidate_tokens.extend(_AMBIGUOUS_ALTS.get(tok, []))
    # Digraph-heavy tie-breakers seen in benchmark.
    if tok == "IN":
        candidate_tokens.extend(["RI", "BE", "M"])
    if tok in {"RI", "BE"}:
        candidate_tokens.append("IN")
    candidate_tokens = [
        c for c in candidate_tokens
        if len(c) == 1 or c in _GAME_DIGRAPHS
    ]
    candidate_tokens = list(dict.fromkeys(candidate_tokens))

    best_token = tok
    best_score = float(base_score)
    jitters = [(0, 0), (-3, 0), (3, 0), (0, -3), (0, 3), (-5, 0), (5, 0), (0, -5), (0, 5)]
    for hs in (1.0, 1.12):
        half2 = max(15, int(round(half * hs)))
        for dx, dy in jitters:
            gray = _extract_tile_gray(
                image_bgr,
                tile,
                half2,
                search_scale=search_scale,
                use_ink_centroid=use_ink_centroid,
                center_dx=dx,
                center_dy=dy,
            )
            for cand in candidate_tokens:
                s = _score_token_on_gray(gray, cand)
                if s > best_score:
                    best_score = float(s)
                    best_token = cand

    ref_gray = _extract_tile_gray(
        image_bgr,
        tile,
        half,
        search_scale=search_scale,
        use_ink_centroid=use_ink_centroid,
    )
    f = _glyph_features(ref_gray)
    if best_token == "M" and "IN" in candidate_tokens and f["components"] >= 2:
        best_token = "IN"
        best_score = max(best_score, base_score + 0.01)
    if best_token in {"N", "I"} and "Q" in candidate_tokens and f["holes"] >= 1:
        best_token = "Q"
        best_score = max(best_score, base_score + 0.01)
    if best_token == "B" and "U" in candidate_tokens and f["holes"] < 1:
        best_token = "U"
        best_score = max(best_score, base_score + 0.01)
    if (
        best_token == "I"
        and "J" in candidate_tokens
        and f["bot_width"] > f["top_width"] * 1.08
        and f["bot_ink"] > f["top_ink"] * 1.20
    ):
        best_token = "J"
        best_score = max(best_score, base_score + 0.01)

    if best_token != tok and best_score > base_score + 0.005:
        return best_token, best_score, True
    return base_token, base_score, False


_GEOMETRY_MODES: Dict[str, Dict[str, Any]] = {
    "base": {"regularizer": "new", "half_scale": 1.0, "search_scale": 1.8, "use_ink_centroid": True},
    "retry1": {"regularizer": "old", "half_scale": 1.0, "search_scale": 2.0, "use_ink_centroid": True},
    "retry2": {"regularizer": "new", "half_scale": 1.15, "search_scale": 2.2, "use_ink_centroid": False},
}


def _token_is_plausible(token: str) -> bool:
    t = token.upper()
    return t in _GAME_DIGRAPHS or (len(t) == 1 and "A" <= t <= "Z")


def _cv(values: List[int]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if mean <= 1e-6:
        return 0.0
    return float(np.std(arr) / mean)


def _geometry_metrics(board: DetectedBoard) -> Dict[str, float]:
    n = board.grid_size
    row_centers = [int(round(np.mean([t.cy for t in board.tiles if t.row == r]))) for r in range(n)]
    col_centers = [int(round(np.mean([t.cx for t in board.tiles if t.col == c]))) for c in range(n)]
    row_gaps = [row_centers[i + 1] - row_centers[i] for i in range(max(0, n - 1))]
    col_gaps = [col_centers[i + 1] - col_centers[i] for i in range(max(0, n - 1))]
    row_cv = _cv(row_gaps)
    col_cv = _cv(col_gaps)
    coverage = 0.0
    if board.roi_width > 0 and board.roi_height > 0:
        coverage = ((col_centers[-1] - col_centers[0]) * (row_centers[-1] - row_centers[0])) / max(
            1.0,
            float(board.roi_width * board.roi_height),
        )
    return {
        "row_spacing_cv": row_cv,
        "col_spacing_cv": col_cv,
        "grid_coverage_ratio": float(coverage),
    }


def _failure_bucket(geometry_low: bool, invalid_count: int, low_conf_ratio: float) -> str:
    if geometry_low:
        return "geometry"
    if invalid_count > 0:
        return "segmentation"
    if low_conf_ratio > 0.20:
        return "classification"
    return "ok"


def _score_candidate(tiles: List[OCRTileResult], geom: Dict[str, float], low_conf_threshold: float) -> float:
    if not tiles:
        return -1e9
    conf_mean = float(np.mean([t.confidence for t in tiles]))
    low_conf_ratio = float(np.mean([1.0 if t.confidence < low_conf_threshold else 0.0 for t in tiles]))
    invalid_ratio = float(np.mean([0.0 if _token_is_plausible(t.normalized_token) else 1.0 for t in tiles]))
    geom_penalty = 2.8 * geom["row_spacing_cv"] + 2.8 * geom["col_spacing_cv"]
    return conf_mean - 0.90 * low_conf_ratio - 0.80 * invalid_ratio - geom_penalty


def _run_mode(
    image_bgr: np.ndarray,
    board: DetectedBoard,
    *,
    mode: str,
    low_conf_threshold: float,
) -> Tuple[DetectedBoard, List[OCRTileResult], List[List[str]], Dict[str, Any], float]:
    cfg = _GEOMETRY_MODES[mode]
    candidate_board = _regularise_board_OLD(board) if cfg["regularizer"] == "old" else _regularise_board(board)
    n = candidate_board.grid_size
    base_half = _board_letter_half(candidate_board)
    half = max(15, int(round(base_half * float(cfg["half_scale"]))))

    tiles_out: List[OCRTileResult] = []
    grid_out = [["?" for _ in range(n)] for _ in range(n)]
    ambiguous_tile_count = 0
    second_pass_used = 0
    for tile in candidate_board.tiles:
        token, score = classify_tile(
            image_bgr,
            tile,
            half=half,
            search_scale=float(cfg["search_scale"]),
            use_ink_centroid=bool(cfg["use_ink_centroid"]),
        )
        token, score, refined = _maybe_refine_ambiguous_tile(
            image_bgr,
            tile,
            base_token=token,
            base_score=float(score),
            half=half,
            search_scale=float(cfg["search_scale"]),
            use_ink_centroid=bool(cfg["use_ink_centroid"]),
        )
        if refined:
            second_pass_used += 1
        if (token.upper() in _AMBIGUOUS_ALTS and score < 0.86) or score < low_conf_threshold:
            ambiguous_tile_count += 1
        low = score < low_conf_threshold
        row = tile.row
        col = tile.col
        tiles_out.append(
            OCRTileResult(
                index=tile.index,
                row=row,
                col=col,
                raw_token=token,
                normalized_token=token,
                confidence=float(score),
                low_confidence=low,
                source_method="template_match",
            )
        )
        grid_out[row][col] = token

    geom = _geometry_metrics(candidate_board)
    low_conf_count = sum(1 for t in tiles_out if t.low_confidence)
    invalid_count = sum(1 for t in tiles_out if not _token_is_plausible(t.normalized_token))
    low_conf_ratio = low_conf_count / max(1, len(tiles_out))
    geometry_low = (
        geom["row_spacing_cv"] > 0.02
        or geom["col_spacing_cv"] > 0.02
        or geom["grid_coverage_ratio"] < 0.58
    )
    diag: Dict[str, Any] = {
        "mode": mode,
        "row_spacing_cv": round(geom["row_spacing_cv"], 6),
        "col_spacing_cv": round(geom["col_spacing_cv"], 6),
        "grid_coverage_ratio": round(geom["grid_coverage_ratio"], 6),
        "low_conf_count": low_conf_count,
        "invalid_token_count": invalid_count,
        "low_conf_ratio": round(low_conf_ratio, 6),
        "geometry_low_quality": geometry_low,
        "failure_bucket": _failure_bucket(geometry_low, invalid_count, low_conf_ratio),
        "ambiguous_tile_count": ambiguous_tile_count,
        "second_pass_used": bool(second_pass_used > 0),
        "second_pass_tile_count": int(second_pass_used),
    }
    candidate_score = _score_candidate(tiles_out, geom, low_conf_threshold)
    recapture_recommended = (
        ambiguous_tile_count >= 2
        or low_conf_ratio > 0.12
        or invalid_count > 0
        or candidate_score < 0.80
    )
    diag["recapture_recommended"] = bool(recapture_recommended)
    diag["candidate_score"] = round(candidate_score, 6)
    return candidate_board, tiles_out, grid_out, diag, candidate_score


# ---------------------------------------------------------------------------
# Public API (matches nvidia_ocr_board signature)
# ---------------------------------------------------------------------------

def template_ocr_board(
    image_bgr: np.ndarray,
    board: DetectedBoard,
    *,
    debug_dir: Optional[Path] = None,
    low_conf_threshold: float = 0.55,
) -> OCRBoardResult:
    candidates: List[Tuple[str, DetectedBoard, List[OCRTileResult], List[List[str]], Dict[str, Any], float]] = []
    base_board, base_tiles, base_grid, base_diag, base_score = _run_mode(
        image_bgr,
        board,
        mode="base",
        low_conf_threshold=low_conf_threshold,
    )
    candidates.append(("base", base_board, base_tiles, base_grid, base_diag, base_score))

    needs_retry = (
        float(base_score) < 0.82
        or int(base_diag.get("invalid_token_count", 0)) > 0
        or float(base_diag.get("low_conf_ratio", 0.0)) > 0.12
    )
    if needs_retry:
        for mode in ("retry1", "retry2"):
            c_board, c_tiles, c_grid, c_diag, c_score = _run_mode(
                image_bgr,
                board,
                mode=mode,
                low_conf_threshold=low_conf_threshold,
            )
            candidates.append((mode, c_board, c_tiles, c_grid, c_diag, c_score))

    def _grid_agreement(g1: List[List[str]], g2: List[List[str]]) -> float:
        n = len(g1)
        m = len(g1[0]) if n else 0
        if n == 0 or m == 0:
            return 0.0
        same = 0
        total = n * m
        for r in range(n):
            for c in range(m):
                if g1[r][c].upper() == g2[r][c].upper():
                    same += 1
        return same / total

    rescored: List[Tuple[str, DetectedBoard, List[OCRTileResult], List[List[str]], Dict[str, Any], float]] = []
    for i, cand in enumerate(candidates):
        agreements = []
        for j, other in enumerate(candidates):
            if i == j:
                continue
            agreements.append(_grid_agreement(cand[3], other[3]))
        agreement_bonus = (sum(agreements) / len(agreements)) if agreements else 0.0
        final_score = cand[5] + 0.22 * agreement_bonus
        cand[4]["agreement_bonus"] = round(agreement_bonus, 6)
        cand[4]["final_score"] = round(final_score, 6)
        rescored.append((cand[0], cand[1], cand[2], cand[3], cand[4], final_score))

    mode, chosen_board, tiles_out, grid_out, diag, _ = max(rescored, key=lambda it: it[5])
    diag["final_selection_reason"] = "second_pass" if bool(diag.get("second_pass_used")) else "base"
    any_low = any(t.low_confidence for t in tiles_out)
    retry_used = mode != "base"

    debug_path = None
    if debug_dir is not None:
        debug_path = str(_save_debug_overlay(image_bgr, chosen_board, tiles_out, Path(debug_dir)))

    return OCRBoardResult(
        calibration_id="auto_template",
        grid_size=chosen_board.grid_size,
        tiles=tiles_out,
        normalized_grid=grid_out,
        has_low_confidence=any_low,
        debug_overlay_path=debug_path,
        template_match_count=len(tiles_out),
        local_ocr_count=len(tiles_out),
        selected_geometry_mode=mode,
        geometry_retry_used=retry_used,
        diagnostics=diag,
    )


def _save_debug_overlay(image_bgr, board, tile_results, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    idx_map = {t.index: t for t in tile_results}
    for tile in board.tiles:
        ocr = idx_map.get(tile.index)
        r = tile.radius
        colour = "red" if (ocr and ocr.low_confidence) else "lime"
        draw.ellipse(
            [(tile.cx - r, tile.cy - r), (tile.cx + r, tile.cy + r)],
            outline=colour, width=3,
        )
        label = f"{ocr.normalized_token if ocr else '?'} {ocr.confidence:.2f}" if ocr else "?"
        draw.text((tile.cx - r, tile.cy - r - 18), label, fill=colour)
    out_path = out_dir / f"ocr_template_{board.grid_size}x{board.grid_size}.png"
    pil.save(out_path)
    return out_path
