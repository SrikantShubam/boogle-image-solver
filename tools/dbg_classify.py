"""Classify tile 0 and print top 5 template matches."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
from skimage.metrics import structural_similarity as ssim

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import _build_templates, _binarize_and_center, _extract_tile_gray
from autoplay_v2 import template_ocr as t

_build_templates()
OUT = REPO / "tools" / "font_id" / "templates"
OUT.mkdir(parents=True, exist_ok=True)
for L, tpl in t._UPPER_TEMPLATES.items():
    cv2.imwrite(str(OUT / f"U_{L}.png"), tpl)
print(f"Saved {len(t._UPPER_TEMPLATES)} templates to {OUT}")

img = cv2.imread(str(REPO / "images screenshots" / "1.jpeg"))
board = detect_board(img, force_grid_size=5)

for tile in board.tiles[:5]:
    gray = _extract_tile_gray(img, tile)
    norm = _binarize_and_center(gray)
    scores = []
    for L, tpl in t._UPPER_TEMPLATES.items():
        s = float(ssim(norm, tpl, data_range=255))
        scores.append((L, s))
    scores.sort(key=lambda x: -x[1])
    print(f"tile {tile.index} ({tile.row},{tile.col}) top5: {scores[:5]}")
