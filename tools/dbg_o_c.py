"""Inspect why O classifies as C."""
import sys
from pathlib import Path
import cv2
from skimage.metrics import structural_similarity as ssim

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import (
    _regularise_board, _board_letter_half, _extract_tile_gray,
    _binarize_and_center, _build_templates,
)
from autoplay_v2 import template_ocr as t

_build_templates()
OUT = REPO / "tools" / "font_id" / "dbg_o"
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(REPO / "new images" / "2.jpeg"))
board = detect_board(img, force_grid_size=5)
board = _regularise_board(board)
half = _board_letter_half(board)

# GT board 2: row 2 = O S A O U (mispredicted O->C)
# Get tile at (2,0) and (2,3)
for tile in board.tiles:
    if (tile.row, tile.col) in [(2, 0), (2, 3), (1, 2), (3, 2)]:
        gray = _extract_tile_gray(img, tile, half)
        norm = _binarize_and_center(gray)
        cv2.imwrite(str(OUT / f"o_tile_{tile.row}_{tile.col}_gray.png"), gray)
        cv2.imwrite(str(OUT / f"o_tile_{tile.row}_{tile.col}_norm.png"), norm)
        # Score O and C
        o_score = ssim(norm, t._UPPER_TEMPLATES['O'], data_range=255)
        c_score = ssim(norm, t._UPPER_TEMPLATES['C'], data_range=255)
        print(f"tile ({tile.row},{tile.col}) O={o_score:.4f}  C={c_score:.4f}")
