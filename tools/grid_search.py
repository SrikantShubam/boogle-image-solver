"""Grid-search scoring/morph params for template_ocr. Prints best combo."""
from __future__ import annotations
import json, sys, itertools, os
from pathlib import Path
import cv2
import numpy as np

os.environ["AUTOPLAY_V2_QUIET"] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Silence board_detector prints
import builtins
_orig_print = builtins.print
def _qprint(*a, **k):
    s = " ".join(str(x) for x in a)
    if s.startswith("[board_detector]"):
        return
    _orig_print(*a, **k, flush=True)
builtins.print = _qprint

from autoplay_v2.board_detector import detect_board
from autoplay_v2 import template_ocr as t
from autoplay_v2.template_ocr import (
    _regularise_board, _board_letter_half, _extract_tile_gray,
    _binarize_and_center, _ink_bbox, _DIGRAPHS,
)
from PIL import ImageFont, Image, ImageDraw

DIR = REPO / "new images"
GT = json.loads((DIR / "ground_truth.json").read_text())

# ---- Preload boards ----
BOARDS = {}
for fname in GT:
    img = cv2.imread(str(DIR / fname))
    board = detect_board(img, force_grid_size=len(GT[fname]))
    board = _regularise_board(board)
    BOARDS[fname] = (img, board, _board_letter_half(board))

# ---- Cache extracted tile grays (independent of scoring params) ----
TILE_GRAYS = {}
for fname, (img, board, half) in BOARDS.items():
    for tile in board.tiles:
        TILE_GRAYS[(fname, tile.row, tile.col)] = _extract_tile_gray(img, tile, half)


def build_templates(font_size, dilate_k, erode_k):
    font = ImageFont.truetype(str(t._FONT_PATH), font_size)
    def render(letter):
        img = Image.new("L", (t.CANVAS*3, t.CANVAS*3), 0)
        ImageDraw.Draw(img).text((t.CANVAS*1.5, t.CANVAS*1.5), letter, fill=255, font=font, anchor="mm")
        b = (np.array(img) > 127).astype(np.uint8) * 255
        if dilate_k > 1: b = cv2.dilate(b, np.ones((dilate_k, dilate_k), np.uint8), 1)
        if erode_k > 1:  b = cv2.erode(b, np.ones((erode_k, erode_k), np.uint8), 1)
        return _binarize_and_center(b, t.CANVAS, already_binary=True)
    U = {L: render(L) for L in t.UPPER}
    L_ = {L: render(L) for L in t.LOWER}
    return U, L_


def tpl_features(tmpls):
    out = {}
    for L, tpl in tmpls.items():
        edge = cv2.Canny(tpl, 50, 150)
        dt = cv2.distanceTransform((255 - tpl).astype(np.uint8), cv2.DIST_L2, 3)
        out[L] = (tpl, edge, dt, int((tpl > 0).sum()))
    return out


def classify(slot, feats, iou_w, chamfer_w):
    slot_edge = cv2.Canny(slot, 50, 150)
    slot_ink = slot > 0
    slot_dt = cv2.distanceTransform((255 - slot).astype(np.uint8), cv2.DIST_L2, 3)
    slot_sum = int(slot_ink.sum())
    best_L, best_s = "?", -1e9
    ea = slot_edge > 0
    for L, (tpl, tpl_edge, tpl_dt, tpl_sum) in feats.items():
        eb = tpl_edge > 0
        if ea.any() and eb.any():
            chamfer = (float(tpl_dt[ea].mean()) + float(slot_dt[eb].mean())) / 2.0
        else:
            chamfer = 50.0
        b = tpl > 0
        inter = int((slot_ink & b).sum())
        union = slot_sum + tpl_sum - inter
        iou = inter / max(1, union)
        s = iou_w * iou - chamfer_w * chamfer
        if s > best_s:
            best_s, best_L = s, L
    return best_L


def classify_tile(fname, tile, U_feats, L_feats, iou_w, chamfer_w, asp_th):
    gray = TILE_GRAYS[(fname, tile.row, tile.col)]
    bb = _ink_bbox(gray)
    if bb is None:
        return "?"
    x0, y0, x1, y1, th = bb
    w, h = x1 - x0, y1 - y0
    if w / max(1, h) < asp_th:
        norm = _binarize_and_center(th[y0:y1, x0:x1], already_binary=True)
        return classify(norm, U_feats, iou_w, chamfer_w)
    strip = th[y0:y1, x0:x1]
    col_sum = strip.sum(axis=0)
    lo = int(len(col_sum) * 0.40); hi = int(len(col_sum) * 0.68)
    mid = lo + int(np.argmin(col_sum[lo:hi])) if hi > lo + 1 else len(col_sum) // 2
    up = _binarize_and_center(strip[:, :mid], already_binary=True)
    dn = _binarize_and_center(strip[:, mid:], already_binary=True)
    up_L = classify(up, U_feats, iou_w, chamfer_w)
    valid = {d[1] for d in _DIGRAPHS if d[0] == up_L}
    if valid:
        restricted = {v: L_feats[v.lower()] for v in valid if v.lower() in L_feats}
        lo_L = classify(dn, restricted, iou_w, chamfer_w) if restricted else classify(dn, L_feats, iou_w, chamfer_w)
    else:
        lo_L = classify(dn, L_feats, iou_w, chamfer_w)
    return up_L + lo_L.upper()


def eval_combo(U_feats, L_feats, iou_w, chamfer_w, asp_th):
    total = correct = 0
    boards_total = boards_exact = 0
    for fname, truth in GT.items():
        img, board, _ = BOARDS[fname]
        n = len(truth)
        per_board_ok = 0
        for tile in board.tiles:
            got = classify_tile(fname, tile, U_feats, L_feats, iou_w, chamfer_w, asp_th)
            total += 1
            if got.upper() == truth[tile.row][tile.col].upper():
                correct += 1
                per_board_ok += 1
        boards_total += 1
        if per_board_ok == n * n:
            boards_exact += 1
    return correct, total, boards_exact, boards_total


grid = list(itertools.product(
    [90, 110, 130],           # font_size
    [0, 3, 5],                 # dilate
    [0, 3],                    # erode
    [0.0, 0.5, 1.0, 2.0],      # iou_w
    [1.0],                     # chamfer_w (normalized)
    [1.05, 1.10, 1.15, 1.25],  # asp_th
))
print(f"Combos: {len(grid)}")

tmpl_cache = {}
results = []
for i, (fs, dk, ek, iw, cw, at) in enumerate(grid):
    key = (fs, dk, ek)
    if key not in tmpl_cache:
        U, L = build_templates(fs, dk, ek)
        tmpl_cache[key] = (tpl_features(U), tpl_features(L))
    Uf, Lf = tmpl_cache[key]
    c, t_, b_exact, b_total = eval_combo(Uf, Lf, iw, cw, at)
    acc = 100.0 * c / t_
    b_acc = 100.0 * b_exact / max(1, b_total)
    results.append((b_acc, acc, fs, dk, ek, iw, cw, at))
    print(
        f"[{i+1}/{len(grid)}] fs={fs} d={dk} e={ek} iou={iw} ch={cw} asp={at} "
        f"-> tiles {c}/{t_} {acc:.2f}% | boards {b_exact}/{b_total} {b_acc:.2f}%"
    )

results.sort(reverse=True)
print("\n=== TOP 15 ===")
print("board% tile%  font  dil  ero  iou   ch   asp")
for b_acc, acc, fs, dk, ek, iw, cw, at in results[:15]:
    print(f"{b_acc:6.2f} {acc:6.2f}  {fs:4d} {dk:4d} {ek:4d}  {iw:4.1f} {cw:4.1f}  {at:.2f}")
