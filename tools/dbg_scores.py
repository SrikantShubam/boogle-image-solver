"""Dump per-letter scores for specific failing tiles."""
import sys
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import (
    _regularise_board, _board_letter_half, _extract_tile_gray,
    _binarize_and_center, _binary_similarity, _build_templates,
)
from autoplay_v2 import template_ocr as t

_build_templates()
OUT = REPO / "tools" / "dbg_scores"
OUT.mkdir(parents=True, exist_ok=True)

# Known C tiles per ground truth
CASES = [
    ("8.jpeg", [(0,2),(2,3)]),
    ("9.jpeg", [(1,0),(4,4),(4,2)]),   # (4,2) is Z
    ("10.jpeg", [(2,2),(3,2)]),
    ("4.jpeg", [(4,3)]),                # G->B
]

for fname, tiles in CASES:
    img = cv2.imread(str(REPO / "new images" / fname))
    board = detect_board(img, force_grid_size=5)
    board = _regularise_board(board)
    half = _board_letter_half(board)
    for rr, cc in tiles:
        tile = next(t_ for t_ in board.tiles if t_.row == rr and t_.col == cc)
        gray = _extract_tile_gray(img, tile, half)
        norm = _binarize_and_center(gray)
        cv2.imwrite(str(OUT / f"{fname}_{rr}_{cc}.png"), norm)
        scores = []
        for L, tpl in t._UPPER_TEMPLATES.items():
            a = norm > 0; b = tpl > 0
            inter = int(np.logical_and(a,b).sum())
            union = int(a.sum()) + int(b.sum()) - inter
            iou = inter / max(1, union)
            sim = _binary_similarity(norm, tpl)
            scores.append((0.7*iou + 0.3*sim, L, iou, sim))
        scores.sort(reverse=True)
        print(f"\n{fname} ({rr},{cc}) — top 5:")
        for sc, L, iou, sim in scores[:5]:
            print(f"  {L}  score={sc:.3f}  iou={iou:.3f}  sim={sim:.3f}")
