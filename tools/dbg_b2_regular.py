import sys
from pathlib import Path
import cv2
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import _regularise_board, _board_letter_half, _extract_tile_gray, _binarize_and_center

OUT = REPO / "tools" / "font_id" / "dbg_b2_reg"
OUT.mkdir(parents=True, exist_ok=True)
img = cv2.imread(str(REPO / "images screenshots" / "2.jpeg"))
board = detect_board(img, force_grid_size=5)
board = _regularise_board(board)
half = _board_letter_half(board)
print(f"half={half}")
for t in board.tiles:
    print(f"  t{t.index:2d} ({t.row},{t.col}) cx={t.cx} cy={t.cy} r={t.radius}")
for t in board.tiles[:6]:
    gray = _extract_tile_gray(img, t, half)
    cv2.imwrite(str(OUT / f"t{t.index:02d}.png"), gray)
    nm = _binarize_and_center(gray)
    cv2.imwrite(str(OUT / f"t{t.index:02d}_n.png"), nm)
