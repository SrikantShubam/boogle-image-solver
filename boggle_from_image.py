import argparse
import json
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractNotFoundError

from boggle_winner import (
    DIGRAPH_TOKENS,
    WORDS_PATH,
    build_neighbors,
    ensure_word_list,
    load_words_from_file,
    prepare_solver_resources,
    serialize_board,
    solve_exact_trie_dfs,
    validate_candidates,
)


# HSV bounds tuned for this game's cyan-highlighted tile style.
CYAN_LOWER = (70, 45, 75)
CYAN_UPPER = (105, 255, 255)


def _normalize_token(text):
    letters = re.sub(r"[^A-Z]", "", text.upper())
    if not letters:
        return None

    if len(letters) >= 2:
        two = letters[:2]
        if two in DIGRAPH_TOKENS:
            return two
        # Accept generic two-letter tiles if OCR strongly returns 2 chars.
        return two

    return letters[0]


def _find_candidate_tiles(image_bgr):
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # White circles + cyan highlighted circle.
    white_mask = cv2.inRange(hsv, (0, 0, 170), (179, 80, 255))
    cyan_mask = cv2.inRange(hsv, CYAN_LOWER, CYAN_UPPER)
    mask = cv2.bitwise_or(white_mask, cyan_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    img_area = h * w
    # Require tiles to be at least 0.3% of image area — filters tiny UI noise.
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

    # Non-maximum suppression: each tile has an inner white circle AND an outer
    # blue-border ring, both detected separately.  Keep only the larger of any
    # two circles whose centres are very close (concentric duplicates).
    candidates.sort(key=lambda c: -c[2])  # largest radius first
    nms = []
    for cx, cy, r in candidates:
        if not any(
            (cx - kx) ** 2 + (cy - ky) ** 2 < (max(r, kr) * 0.55) ** 2
            for kx, ky, kr in nms
        ):
            nms.append((cx, cy, r))
    return nms


def _select_grid_points(candidates):
    if len(candidates) < 14:
        raise RuntimeError(f"Not enough tile candidates found ({len(candidates)}).")

    pts = np.array([[c[0], c[1], c[2]] for c in candidates], dtype=np.float32)
    centers = pts[:, :2]
    radii = pts[:, 2]

    best = None
    for n in (5, 4):
        if len(candidates) < n * n:
            continue

        # Cluster X and Y into n groups; map points into n x n cells.
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

            # Keep closest point to cell centroid.
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

        # Slight penalty for larger grids so 4x4 wins ties over 5x5.
        score = coverage - n * 0.001
        if best is None or score > best[0]:
            best = (score, n, selected)

    if best is None:
        raise RuntimeError("Could not map detected circles into a 4x4 or 5x5 grid.")

    _, n, selected_points = best
    return n, selected_points


def _ocr_tile(tile_bgr):
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Keep dark letters on white background — Tesseract expects black-on-white.
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    )
    bw = cv2.resize(bw, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    config_word = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    config_char = "--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    text = pytesseract.image_to_string(bw, config=config_word)
    token = _normalize_token(text)
    if token:
        return token

    text = pytesseract.image_to_string(bw, config=config_char)
    token = _normalize_token(text)
    if token:
        return token

    return None


def _cyan_ratio(tile_hsv):
    mask = cv2.inRange(tile_hsv, CYAN_LOWER, CYAN_UPPER)
    return float(np.count_nonzero(mask)) / float(mask.size)


def extract_board_from_image(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    candidates = _find_candidate_tiles(image)
    n, points = _select_grid_points(candidates)

    grid = [[None for _ in range(n)] for _ in range(n)]
    cyan_scores = []
    digraph_indices = set()

    # Row-major points; decode each tile.
    for idx, (cx, cy, radius) in enumerate(points):
        row, col = divmod(idx, n)
        r = max(18, int(radius * 0.92))
        y0 = max(0, cy - r)
        y1 = min(image.shape[0], cy + r)
        x0 = max(0, cx - r)
        x1 = min(image.shape[1], cx + r)

        tile = image[y0:y1, x0:x1].copy()
        if tile.size == 0:
            raise RuntimeError("Detected empty tile crop while parsing board.")

        token = _ocr_tile(tile)
        if not token:
            raise RuntimeError(f"OCR failed at tile ({row}, {col}).")

        grid[row][col] = token
        if len(token) > 1:
            digraph_indices.add(idx)

        tile_hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        cyan_scores.append((idx, _cyan_ratio(tile_hsv)))

    # At most 2 multi-letter nodes.
    if len(digraph_indices) > 2:
        digraph_indices = set(sorted(digraph_indices)[:2])

    # Special bonus node: strongest cyan-highlighted tile.
    bonus_index = None
    if cyan_scores:
        best_idx, best_ratio = max(cyan_scores, key=lambda x: x[1])
        if best_ratio >= 0.12:
            bonus_index = best_idx

    return {
        "grid": grid,
        "digraph_indices": digraph_indices,
        "bonus_index": bonus_index,
    }


def solve_words_from_image(image_path, words_path=WORDS_PATH, time_budget=10.0):
    board = extract_board_from_image(image_path)

    words_path = Path(words_path)
    resources = prepare_solver_resources(load_words_from_file(ensure_word_list(words_path)))
    resources["neighbors"] = build_neighbors(len(board["grid"]))

    deadline = time.perf_counter() + float(time_budget)
    candidates = solve_exact_trie_dfs(board["grid"], deadline, resources)
    report = validate_candidates(
        candidates,
        board["grid"],
        resources["all_words"],
        resources["neighbors"],
        digraph_indices=board["digraph_indices"],
        bonus_index=board["bonus_index"],
    )

    words = sorted(report["valid_words"], key=lambda w: (-len(w), w))
    return board, report, words


def _configure_tesseract(tesseract_cmd):
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)

    cmd = pytesseract.pytesseract.tesseract_cmd
    if cmd and Path(str(cmd)).exists():
        return

    if shutil.which("tesseract"):
        return

    raise RuntimeError(
        "Tesseract executable not found. Install Tesseract OCR and/or pass --tesseract-cmd "
        "with the full executable path (for example: "
        r"C:\Program Files\Tesseract-OCR\tesseract.exe)."
    )


def _print_board(board):
    print("Detected board:")
    n = len(board["grid"])
    digraph = set(board.get("digraph_indices", set()))
    bonus = board.get("bonus_index")

    for r in range(n):
        row_tokens = []
        for c in range(n):
            idx = r * n + c
            token = board["grid"][r][c]
            suffix = ""
            if idx in digraph:
                suffix += "~"
            if bonus is not None and idx == bonus:
                suffix += "*"
            row_tokens.append(f"{token}{suffix}")
        print(" ".join(row_tokens))

    print(f"Board key: {serialize_board(board)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Boggle grid from screenshot and solve with Exact_Trie_DFS."
    )
    parser.add_argument("--image", required=True, help="Path to screenshot image.")
    parser.add_argument("--words", default=str(WORDS_PATH), help="Path to dictionary word file.")
    parser.add_argument("--time-budget", type=float, default=10.0, help="Solver time budget in seconds.")
    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help="Optional full path to tesseract executable.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write JSON output with board, score and words.",
    )
    args = parser.parse_args()

    _configure_tesseract(args.tesseract_cmd)

    try:
        board, report, words = solve_words_from_image(
            image_path=args.image,
            words_path=args.words,
            time_budget=args.time_budget,
        )
    except TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract executable not available. Install it and retry with --tesseract-cmd if needed."
        )

    _print_board(board)
    print(f"Valid unique words: {report['valid_count']}")
    print(f"Bonus points: {report['bonus_points']:.2f}")
    print(f"Total score: {report['score']:.2f}")
    print("")
    print("Words:")
    for w in words:
        print(w)

    if args.json_out:
        board_json = {
            "grid": board["grid"],
            "digraph_indices": sorted(list(board.get("digraph_indices", set()))),
            "bonus_index": board.get("bonus_index"),
        }
        payload = {
            "board": board_json,
            "valid_count": report["valid_count"],
            "bonus_points": report["bonus_points"],
            "score": report["score"],
            "words": words,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON output: {args.json_out}")


if __name__ == "__main__":
    main()
