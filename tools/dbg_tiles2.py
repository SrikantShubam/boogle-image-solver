"""Dump tiles from board 2 to see why so many classify as 'I'."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import _extract_tile_gray, _binarize_and_center, _ink_bbox

OUT = REPO / "tools" / "font_id" / "dbg_b2"
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(REPO / "images screenshots" / "2.jpeg"))
board = detect_board(img, force_grid_size=5)

radii = [t.radius for t in board.tiles]
print(f"radii: min={min(radii)} max={max(radii)} median={sorted(radii)[len(radii)//2]}")

for tile in board.tiles:
    gray = _extract_tile_gray(img, tile)
    bb = _ink_bbox(gray)
    if bb:
        x0,y0,x1,y1,_ = bb
        aspect = (x1-x0)/max(1,y1-y0)
        print(f"tile {tile.index:2d} r={tile.radius} crop={gray.shape} bbox={x1-x0}x{y1-y0} aspect={aspect:.2f}")
    norm = _binarize_and_center(gray)
    cv2.imwrite(str(OUT / f"t{tile.index:02d}_norm.png"), norm)
