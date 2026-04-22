import sys
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import (
    _regularise_board, _board_letter_half, _extract_tile_gray,
    _binarize_and_center, _build_templates,
)
from autoplay_v2 import template_ocr as t

_build_templates()
OUT = REPO / "tools" / "dbg_compare"
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(REPO / "new images" / "8.jpeg"))
board = detect_board(img, force_grid_size=5)
board = _regularise_board(board)
half = _board_letter_half(board)
tile = next(t_ for t_ in board.tiles if t_.row == 0 and t_.col == 2)
gray = _extract_tile_gray(img, tile, half)
norm = _binarize_and_center(gray)
cv2.imwrite(str(OUT / "slot_C.png"), norm)
cv2.imwrite(str(OUT / "slot_C_gray.png"), gray)
cv2.imwrite(str(OUT / "tpl_C.png"), t._UPPER_TEMPLATES['C'])
cv2.imwrite(str(OUT / "tpl_E.png"), t._UPPER_TEMPLATES['E'])

# Side by side: slot | tpl_C | tpl_E | overlay(slot,tpl_C) | overlay(slot,tpl_E)
slot = norm
tC = t._UPPER_TEMPLATES['C']
tE = t._UPPER_TEMPLATES['E']
def overlay(s, tp):
    r = np.zeros((s.shape[0], s.shape[1], 3), dtype=np.uint8)
    r[..., 2] = s         # slot red
    r[..., 1] = tp        # tpl green
    return r
row = np.hstack([
    cv2.cvtColor(slot, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(tC, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(tE, cv2.COLOR_GRAY2BGR),
    overlay(slot, tC),
    overlay(slot, tE),
])
cv2.imwrite(str(OUT / "compare.png"), row)
print("slot ink:", int((slot>0).sum()))
print("tpl_C ink:", int((tC>0).sum()))
print("tpl_E ink:", int((tE>0).sum()))
print("inter_C:", int(((slot>0) & (tC>0)).sum()))
print("inter_E:", int(((slot>0) & (tE>0)).sum()))
print("slot-not-C:", int(((slot>0) & (tC==0)).sum()))
print("slot-not-E:", int(((slot>0) & (tE==0)).sum()))
print("C-not-slot:", int(((slot==0) & (tC>0)).sum()))
print("E-not-slot:", int(((slot==0) & (tE>0)).sum()))
