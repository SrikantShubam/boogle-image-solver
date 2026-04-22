import sys
from pathlib import Path
import cv2
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import _regularise_board, _board_letter_half, _extract_tile_gray

img = cv2.imread(str(REPO / "new images" / "5.jpeg"))
board = detect_board(img, force_grid_size=5)
board = _regularise_board(board)
half = _board_letter_half(board)
print("half=", half)
# Draw circles on full image for visualization
canvas = img.copy()
for tile in board.tiles:
    colour = (0, 255, 0)
    cv2.circle(canvas, (tile.cx, tile.cy), tile.radius, colour, 2)
    cv2.putText(canvas, f"{tile.row},{tile.col}", (tile.cx-20, tile.cy-tile.radius-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
cv2.imwrite(str(REPO / "tools" / "dbg_board5.png"), canvas)

# Print column centers and row centers
xs = sorted(set(t.cx for t in board.tiles))
ys = sorted(set(t.cy for t in board.tiles))
print("cols:", xs)
print("rows:", ys)
print("col diffs:", [xs[i+1]-xs[i] for i in range(len(xs)-1)])
print("row diffs:", [ys[i+1]-ys[i] for i in range(len(ys)-1)])
