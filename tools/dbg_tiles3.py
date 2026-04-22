"""Dump tiles with the new spacing-based cropping."""
import sys
from pathlib import Path
import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import _board_letter_half, _extract_tile_gray, _binarize_and_center, _ink_bbox

OUT = REPO / "tools" / "font_id" / "dbg_b2_v2"
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(REPO / "images screenshots" / "2.jpeg"))
board = detect_board(img, force_grid_size=5)
half = _board_letter_half(board)
print(f"half={half}")

for tile in board.tiles:
    gray = _extract_tile_gray(img, tile, half)
    cv2.imwrite(str(OUT / f"t{tile.index:02d}_0gray.png"), gray)
    norm = _binarize_and_center(gray)
    cv2.imwrite(str(OUT / f"t{tile.index:02d}_1norm.png"), norm)
    bb = _ink_bbox(gray)
    if bb:
        print(f"t{tile.index:2d} ({tile.row},{tile.col}) gray={gray.shape} bbox={bb[2]-bb[0]}x{bb[3]-bb[1]}")
