"""Identify the font Plato Crosswords uses on its tiles.

Extracts reference glyphs from labeled screenshots, renders candidate fonts,
scores each via SSIM over all 26 uppercase letters, and emits a leaderboard
plus a side-by-side visual report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from autoplay_v2.board_detector import detect_board  # noqa: E402

SCREENSHOTS_DIR = REPO / "images screenshots"
GROUND_TRUTH = SCREENSHOTS_DIR / "ground_truth.json"
OUT_DIR = Path(__file__).parent
REFS_DIR = OUT_DIR / "refs"
FONTS_DIR = OUT_DIR / "fonts"
REPORT_PNG = OUT_DIR / "report.png"

CANVAS = 128
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

FONT_URLS: Dict[str, str] = {
    # All Poppins bold-ish weights
    "Poppins-Medium":     "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Medium.ttf",
    "Poppins-SemiBold":   "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-SemiBold.ttf",
    "Poppins-Bold":       "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf",
    "Poppins-ExtraBold":  "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-ExtraBold.ttf",
    "Poppins-Black":      "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Black.ttf",
}


def _binarize_and_center(gray: np.ndarray, canvas: int = CANVAS) -> np.ndarray:
    """Threshold, find glyph bbox, recenter in a canvas×canvas square."""
    if gray.size == 0:
        return np.zeros((canvas, canvas), dtype=np.uint8)
    # Otsu — invert so glyph is white (255) on black (0) regardless of source.
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() > 127:
        th = 255 - th

    ys, xs = np.where(th > 0)
    if len(xs) == 0:
        return np.zeros((canvas, canvas), dtype=np.uint8)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    crop = th[y0:y1, x0:x1]

    h, w = crop.shape
    target = int(canvas * 0.75)
    scale = target / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    out = np.zeros((canvas, canvas), dtype=np.uint8)
    oy = (canvas - new_h) // 2
    ox = (canvas - new_w) // 2
    out[oy:oy + new_h, ox:ox + new_w] = resized
    return out


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_references() -> Dict[str, np.ndarray]:
    """For each single-letter ground-truth tile, collect a crop. Keep the sharpest per letter."""
    with GROUND_TRUTH.open("r", encoding="utf-8") as f:
        gt: Dict[str, List[List[str]]] = json.load(f)

    best: Dict[str, Tuple[float, np.ndarray]] = {}

    for fname, grid in gt.items():
        img_path = SCREENSHOTS_DIR / fname
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        n = len(grid)
        board = detect_board(img_bgr, force_grid_size=n)
        if board is None or board.grid_size != n:
            print(f"  [skip] {fname}: board detection failed")
            continue

        for tile in board.tiles:
            token = grid[tile.row][tile.col]
            if len(token) != 1 or not token.isalpha():
                continue  # only single uppercase letters
            r = int(tile.radius * 0.65)
            h, w = img_bgr.shape[:2]
            y0 = max(0, tile.cy - r)
            y1 = min(h, tile.cy + r)
            x0 = max(0, tile.cx - r)
            x1 = min(w, tile.cx + r)
            crop = img_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            s = _sharpness(gray)
            if token not in best or s > best[token][0]:
                best[token] = (s, gray)

    refs: Dict[str, np.ndarray] = {}
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    for letter, (_, gray) in best.items():
        norm = _binarize_and_center(gray)
        refs[letter] = norm
        cv2.imwrite(str(REFS_DIR / f"{letter}.png"), norm)

    missing = [L for L in LETTERS if L not in refs]
    if missing:
        print(f"  [warn] no reference crops extracted for: {''.join(missing)}")
    print(f"  extracted refs for {len(refs)}/26 letters")
    return refs


def download_fonts() -> Dict[str, Path]:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, url in FONT_URLS.items():
        dest = FONTS_DIR / f"{name}.ttf"
        if not dest.exists():
            try:
                print(f"  downloading {name}…")
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                dest.write_bytes(r.content)
            except Exception as e:
                print(f"  [skip] {name}: {e}")
                continue
        paths[name] = dest
    return paths


def render_glyph(font_path: Path, letter: str, canvas: int = CANVAS) -> np.ndarray:
    """Render a single letter onto a canvas×canvas binarised image, centered."""
    best_arr: Optional[np.ndarray] = None
    # Find a size that roughly fills ~75% of canvas
    for size in (140, 130, 120, 110, 100, 90, 80):
        try:
            font = ImageFont.truetype(str(font_path), size)
        except Exception:
            return np.zeros((canvas, canvas), dtype=np.uint8)
        img = Image.new("L", (canvas * 2, canvas * 2), 0)
        draw = ImageDraw.Draw(img)
        draw.text((canvas, canvas), letter, fill=255, font=font, anchor="mm")
        arr = np.array(img)
        ys, xs = np.where(arr > 127)
        if len(xs) == 0:
            continue
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if max(h, w) <= canvas:
            # This size fits — use it via the normaliser for consistent centering
            best_arr = arr
            break
        best_arr = arr  # keep last, will be shrunk by normaliser
    if best_arr is None:
        return np.zeros((canvas, canvas), dtype=np.uint8)
    return _binarize_and_center(best_arr, canvas)


def score_font(refs: Dict[str, np.ndarray], font_path: Path) -> Tuple[float, float, Dict[str, float]]:
    per_letter: Dict[str, float] = {}
    scores: List[float] = []
    for letter, ref in refs.items():
        rendered = render_glyph(font_path, letter)
        s = float(ssim(ref, rendered, data_range=255))
        per_letter[letter] = s
        scores.append(s)
    if not scores:
        return 0.0, 0.0, per_letter
    return float(np.mean(scores)), float(np.min(scores)), per_letter


def build_report(refs: Dict[str, np.ndarray], fonts: Dict[str, Path], top: List[str]) -> None:
    rows = 1 + len(top)  # refs row + one per top font
    cols = 26
    cell = CANVAS
    pad = 4
    label_w = 140
    W = label_w + cols * (cell + pad)
    H = rows * (cell + pad) + 20

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        label_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        label_font = ImageFont.load_default()

    # Row 0 = refs
    draw.text((6, 6), "refs", fill="black", font=label_font)
    for i, letter in enumerate(LETTERS):
        x = label_w + i * (cell + pad)
        y = 0
        if letter in refs:
            pil = Image.fromarray(refs[letter]).convert("RGB")
            canvas.paste(pil, (x, y))
        draw.text((x + 4, y + cell - 16), letter, fill="red", font=label_font)

    # Subsequent rows = top fonts
    for r, font_name in enumerate(top, start=1):
        y = r * (cell + pad)
        draw.text((6, y + 6), font_name[:18], fill="black", font=label_font)
        font_path = fonts[font_name]
        for i, letter in enumerate(LETTERS):
            rendered = render_glyph(font_path, letter)
            pil = Image.fromarray(rendered).convert("RGB")
            canvas.paste(pil, (label_w + i * (cell + pad), y))

    canvas.save(REPORT_PNG)
    print(f"  report -> {REPORT_PNG}")


def main() -> int:
    print("1) Extracting reference glyphs from labeled boards…")
    refs = extract_references()
    if len(refs) < 10:
        print("  not enough refs to score fonts — aborting")
        return 2

    print("\n2) Downloading candidate fonts…")
    fonts = download_fonts()
    if not fonts:
        print("  no fonts available — aborting")
        return 2

    print("\n3) Scoring each font…")
    results: List[Tuple[str, float, float, Dict[str, float]]] = []
    for name, path in fonts.items():
        mean_s, worst_s, per = score_font(refs, path)
        results.append((name, mean_s, worst_s, per))
        print(f"  {name:14s}  mean_ssim={mean_s:.3f}  worst={worst_s:.3f}")

    results.sort(key=lambda r: -r[1])

    print("\n=== Leaderboard ===")
    print(f"{'rank':<5}{'font':<16}{'mean_ssim':<12}{'worst':<8}worst_letter")
    for rank, (name, mean_s, worst_s, per) in enumerate(results, start=1):
        worst_letter = min(per, key=per.get) if per else "-"
        print(f"{rank:<5}{name:<16}{mean_s:<12.3f}{worst_s:<8.3f}{worst_letter}")

    top_names = [r[0] for r in results[:3]]
    print("\n4) Building visual report for top 3…")
    build_report(refs, fonts, top_names)
    print("\nDone. Eyeball report.png to confirm the winner.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
