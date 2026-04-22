"""Build per-letter empirical templates by averaging correctly-extracted
game tiles from `new images/` using ground truth. Saves to
`tools/empirical_templates/{UPPER,LOWER}/<letter>.png`.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import (
    _regularise_board, _board_letter_half, _extract_tile_gray,
    _binarize_and_center, _ink_bbox, _DIGRAPHS,
)

DIR = REPO / "new images"
GT = json.loads((DIR / "ground_truth.json").read_text())
OUT = REPO / "tools" / "empirical_templates"
(OUT / "upper").mkdir(parents=True, exist_ok=True)
(OUT / "lower").mkdir(parents=True, exist_ok=True)

upper_acc: dict[str, list[np.ndarray]] = defaultdict(list)
lower_acc: dict[str, list[np.ndarray]] = defaultdict(list)

for fname, truth in GT.items():
    img = cv2.imread(str(DIR / fname))
    board = detect_board(img, force_grid_size=len(truth))
    board = _regularise_board(board)
    half = _board_letter_half(board)
    for tile in board.tiles:
        t = truth[tile.row][tile.col].upper()
        gray = _extract_tile_gray(img, tile, half)
        bb = _ink_bbox(gray)
        if bb is None:
            continue
        x0, y0, x1, y1, th = bb
        w, h = x1 - x0, y1 - y0
        strip = th[y0:y1, x0:x1]
        if len(t) == 1:
            norm = _binarize_and_center(strip, already_binary=True)
            upper_acc[t].append(norm)
        elif len(t) == 2:
            aspect = w / max(1, h)
            if aspect < 1.15:
                continue  # couldn't reliably split, skip
            col_sum = strip.sum(axis=0)
            lo = int(len(col_sum)*0.40); hi = int(len(col_sum)*0.68)
            mid = lo + int(np.argmin(col_sum[lo:hi])) if hi > lo+1 else len(col_sum)//2
            up = _binarize_and_center(strip[:, :mid], already_binary=True)
            dn = _binarize_and_center(strip[:, mid:], already_binary=True)
            upper_acc[t[0]].append(up)
            lower_acc[t[1].lower()].append(dn)

print("Sample counts:")
for L in sorted(upper_acc): print(f"  upper {L}: {len(upper_acc[L])}")
for L in sorted(lower_acc): print(f"  lower {L}: {len(lower_acc[L])}")

# Average & binarise
for L, ss in upper_acc.items():
    avg = np.mean(ss, axis=0).astype(np.uint8)
    _, binar = cv2.threshold(avg, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(OUT / "upper" / f"{L}.png"), binar)
for L, ss in lower_acc.items():
    avg = np.mean(ss, axis=0).astype(np.uint8)
    _, binar = cv2.threshold(avg, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(OUT / "lower" / f"{L}.png"), binar)

print("Written to", OUT)
