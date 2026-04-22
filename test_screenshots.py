"""Run detect_board + ocr_board_auto on every screenshot in 'images screenshots/'
and print the detected grid side-by-side with pass/fail info.
Debug images saved to dbg/<stem>/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

repo = Path(__file__).parent
sys.path.insert(0, str(repo))

from autoplay_v2.board_detector import detect_board
from autoplay_v2.ocr import ocr_board_auto

SCREENSHOTS_DIR = repo / "images screenshots"
DEBUG_ROOT = repo / "dbg"

GROUND_TRUTH: dict[str, list[list[str]]] = {
    "WhatsApp Image 2026-04-19 at 1.39.09 PM.jpeg": [
        ["I", "E", "T",  "N",  "T"],
        ["O", "D", "I",  "N",  "J"],
        ["R", "P", "A",  "X",  "A"],
        ["S", "N", "A",  "AN", "O"],
        ["E", "K", "A",  "L",  "X"],
    ],
    "WhatsApp Image 2026-04-19 at 1.39.09 PM (1).jpeg": [
        ["U", "O",  "U", "S", "B"],
        ["T", "HE", "N", "E", "C"],
        ["S", "R",  "E", "A", "A"],
        ["I", "R",  "Y", "H", "L"],
        ["A", "K",  "N", "V", "O"],
    ],
    "WhatsApp Image 2026-04-19 at 1.39.10 PM.jpeg": [
        ["T", "E",  "A", "ER", "E"],
        ["R", "Q",  "V", "R",  "R"],
        ["A", "E",  "S", "S",  "R"],
        ["H", "E",  "E", "X",  "L"],
        ["S", "U",  "A", "O",  "R"],
    ],
    "WhatsApp Image 2026-04-19 at 1.39.11 PM.jpeg": [
        ["M", "Z",  "R", "P", "I"],
        ["N", "U",  "Z", "U", "L"],
        ["E", "O",  "E", "O", "IN"],
        ["T", "T",  "S", "K", "E"],
        ["E", "M",  "T", "V", "E"],
    ],
}


def _fmt_grid(grid: list[list[str]]) -> str:
    return "\n".join("  ".join(tok.ljust(2) for tok in row) for row in grid)


def run_one(path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"FILE: {path.name}")

    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        print("  ERROR: could not load image")
        return

    debug_dir = DEBUG_ROOT / path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    board = detect_board(image_bgr, debug_dir=debug_dir)
    if board is None:
        print("  ❌ Board not detected")
        return

    print(f"  ✅ Detected {board.grid_size}×{board.grid_size} board")

    ocr = ocr_board_auto(image_bgr, board, debug_dir=debug_dir)
    detected = ocr.normalized_grid

    print("\n  DETECTED GRID:")
    for line in _fmt_grid(detected).splitlines():
        print(f"    {line}")

    truth = GROUND_TRUTH.get(path.name)
    if truth:
        correct = 0
        total = 0
        mismatches = []
        for r, row in enumerate(truth):
            for c, expected in enumerate(row):
                total += 1
                got = detected[r][c] if r < len(detected) and c < len(detected[r]) else "?"
                if got == expected:
                    correct += 1
                else:
                    mismatches.append(f"    ({r},{c}) expected={expected!r} got={got!r}")
        pct = 100 * correct / total
        print(f"\n  ACCURACY: {correct}/{total} = {pct:.0f}%")
        if mismatches:
            print("  MISMATCHES:")
            for m in mismatches:
                print(m)
    else:
        print("  (no ground truth for this file)")


def main() -> None:
    images = sorted(SCREENSHOTS_DIR.glob("*.jpeg")) + sorted(SCREENSHOTS_DIR.glob("*.png"))
    if not images:
        print("No images found in 'images screenshots/'")
        return

    for img_path in images:
        run_one(img_path)

    print(f"\nDebug images saved to: {DEBUG_ROOT}/")


if __name__ == "__main__":
    main()
