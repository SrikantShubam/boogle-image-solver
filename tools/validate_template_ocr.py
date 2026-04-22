"""Validate template OCR against ground_truth.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import template_ocr_board

SCREENSHOTS = REPO / "images screenshots"
GT_PATH = SCREENSHOTS / "ground_truth.json"


def main() -> int:
    with GT_PATH.open("r", encoding="utf-8") as f:
        gt = json.load(f)

    total_tiles = 0
    correct = 0
    per_board = []
    mismatches = []

    for fname, truth_grid in gt.items():
        img_path = SCREENSHOTS / fname
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        n = len(truth_grid)
        board = detect_board(img, force_grid_size=n)
        if board is None or board.grid_size != n:
            print(f"[skip] {fname}: board detection failed")
            continue
        result = template_ocr_board(img, board)
        got = result.normalized_grid

        board_total = n * n
        board_correct = 0
        for r in range(n):
            for c in range(n):
                t = truth_grid[r][c].upper()
                g = got[r][c].upper()
                total_tiles += 1
                if g == t:
                    correct += 1
                    board_correct += 1
                else:
                    mismatches.append((fname, r, c, t, g))
        per_board.append((fname, board_correct, board_total))

    print("\n=== Per-board accuracy ===")
    for fname, c, t in sorted(per_board):
        pct = 100.0 * c / t
        marker = " " if pct == 100.0 else ("~" if pct >= 80 else "!")
        print(f"  {marker} {fname:12s}  {c:3d}/{t:3d}  {pct:5.1f}%")

    overall = 100.0 * correct / max(1, total_tiles)
    print(f"\n=== Overall: {correct}/{total_tiles}  {overall:.2f}% ===")

    if mismatches:
        print(f"\nFirst 40 mismatches (file [r,c]: truth -> got):")
        for fname, r, c, t, g in mismatches[:40]:
            print(f"  {fname:12s} [{r},{c}]: {t:3s} -> {g:3s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
