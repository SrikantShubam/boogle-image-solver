import sys
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import (
    _regularise_board, _board_letter_half, _extract_tile_gray,
    _ink_bbox,
)

OUT = REPO / "tools" / "dbg_digraphs"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    ("1.jpeg", 0, 1, "IN"),
    ("7.jpeg", 3, 0, "IN"),
    ("9.jpeg", 0, 1, "IN"),
    ("6.jpeg", 3, 0, "HE"),
    ("8.jpeg", 1, 0, "HE"),
    ("6.jpeg", 0, 0, "Q"),
    ("5.jpeg", 2, 4, "QU"),
    ("3.jpeg", 1, 2, "AN"),
    ("11.jpeg", 3, 0, "AN"),
    ("2.jpeg", 4, 4, "TH"),
]
for fname, r, c, truth in CASES:
    img = cv2.imread(str(REPO / "new images" / fname))
    board = detect_board(img, force_grid_size=5)
    board = _regularise_board(board)
    half = _board_letter_half(board)
    tile = next(t_ for t_ in board.tiles if t_.row == r and t_.col == c)
    gray = _extract_tile_gray(img, tile, half)
    cv2.imwrite(str(OUT / f"{fname}_{r}_{c}_{truth}.png"), gray)
    bb = _ink_bbox(gray)
    if bb:
        x0, y0, x1, y1, _ = bb
        w = x1-x0; h = y1-y0
        print(f"{fname} ({r},{c}) truth={truth}  w={w} h={h} aspect={w/max(1,h):.2f}")
