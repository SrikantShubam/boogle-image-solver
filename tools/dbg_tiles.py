"""Dump the first row of board 1 — raw crop, binarised, normalised — for inspection."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import _binarize_and_center, _extract_tile_gray, _ink_bbox

OUT = REPO / "tools" / "font_id" / "dbg_tiles"
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(REPO / "images screenshots" / "1.jpeg"))
board = detect_board(img, force_grid_size=5)
assert board is not None

for tile in board.tiles[:10]:
    gray = _extract_tile_gray(img, tile)
    cv2.imwrite(str(OUT / f"tile_{tile.index:02d}_0_gray.png"), gray)
    bb = _ink_bbox(gray)
    if bb:
        x0, y0, x1, y1, th = bb
        cv2.imwrite(str(OUT / f"tile_{tile.index:02d}_1_bin.png"), th)
        cropped = gray[y0:y1, x0:x1]
        cv2.imwrite(str(OUT / f"tile_{tile.index:02d}_2_crop.png"), cropped)
        print(f"tile {tile.index}: bbox {x1-x0}x{y1-y0}, aspect={(x1-x0)/max(1,y1-y0):.2f}")
    norm = _binarize_and_center(gray)
    cv2.imwrite(str(OUT / f"tile_{tile.index:02d}_3_norm.png"), norm)

print(f"Wrote to {OUT}")
