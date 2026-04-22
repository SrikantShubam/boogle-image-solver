from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
import os

# Set tesseract path on Windows if it's available
_TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = _TESS_PATH
from PIL import Image, ImageDraw

from autoplay_v2.calibration import generate_tile_crop_rects
from autoplay_v2.config import (
    DEFAULT_LOCAL_OCR_CONFIDENCE_THRESHOLD,
    DEFAULT_TEMPLATE_MATCH_THRESHOLD,
    OCR_TEMPLATE_LIBRARY_PATH,
    load_json_file,
)
from autoplay_v2.models import (
    CalibrationConfig,
    DetectedBoard,
    DetectedTile,
    OCRBoardResult,
    OCRTileResult,
)

TileReader = Callable[[np.ndarray, int, int, int], Tuple[str, float]]
_GLYPH_CANVAS_SIZE = 40
_DIGRAPHS = {
    "TH",
    "HE",
    "QU",
    "IN",
    "ER",
    "AN",
    "RE",
    "ON",
    "AT",
    "EN",
    "ND",
    "TI",
    "ES",
    "OR",
    "TE",
    "OF",
    "ED",
    "IS",
    "IT",
    "AL",
    "AR",
    "ST",
    "TO",
    "NT",
    "NG",
    "SE",
    "HA",
    "AS",
    "OU",
    "IO",
    "LE",
    "VE",
    "CO",
    "ME",
    "DE",
    "HI",
    "RI",
    "RO",
    "IC",
    "NE",
    "EA",
    "RA",
    "CE",
    "LI",
    "CH",
    "LL",
    "BE",
    "MA",
    "SI",
    "OM",
    "UR",
}

_OCR_CHAR_SUBSTITUTIONS = str.maketrans({
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "G",
})


@dataclass(frozen=True)
class TemplateLibrary:
    labels: np.ndarray
    canvases: np.ndarray
    min_score: float = DEFAULT_TEMPLATE_MATCH_THRESHOLD

    @classmethod
    def empty(cls, min_score: float = DEFAULT_TEMPLATE_MATCH_THRESHOLD) -> "TemplateLibrary":
        return cls(
            labels=np.array([], dtype="<U2"),
            canvases=np.zeros((0, _GLYPH_CANVAS_SIZE * _GLYPH_CANVAS_SIZE), dtype=np.uint8),
            min_score=min_score,
        )

    @classmethod
    def from_tile_images(
        cls,
        entries: Iterable[Tuple[str, np.ndarray]],
        min_score: float = DEFAULT_TEMPLATE_MATCH_THRESHOLD,
    ) -> "TemplateLibrary":
        labels: List[str] = []
        canvases: List[np.ndarray] = []
        for token, tile_image in entries:
            normalized = normalize_token(token)
            if not normalized:
                continue
            canvas = _prepare_token_canvas(tile_image)
            if canvas is None:
                continue
            labels.append(normalized)
            canvases.append(canvas.reshape(-1))
        if not labels:
            return cls.empty(min_score=min_score)
        return cls(
            labels=np.array(labels),
            canvases=np.stack(canvases).astype(np.uint8),
            min_score=min_score,
        )

    @property
    def is_empty(self) -> bool:
        return self.canvases.size == 0 or self.labels.size == 0

    def match_tile(self, tile_image: np.ndarray) -> Tuple[str, float, float]:
        canvas = _prepare_token_canvas(tile_image)
        if canvas is None:
            return "", 0.0, 0.0
        return self.match_canvas(canvas)

    def match_canvas(self, canvas: np.ndarray) -> Tuple[str, float, float]:
        if self.is_empty:
            return "", 0.0, 0.0
        glyph = canvas.reshape(-1).astype(np.uint8)
        intersection = np.minimum(self.canvases, glyph).sum(axis=1).astype(np.float32)
        union = np.maximum(self.canvases, glyph).sum(axis=1).astype(np.float32) + 1e-6
        scores = intersection / union
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if scores.size > 1:
            second_idx = int(np.argpartition(scores, -2)[-2])
            second_score = float(scores[second_idx])
        else:
            second_score = 0.0
        return str(self.labels[best_idx]), best_score, max(0.0, best_score - second_score)

    def to_dict(self) -> dict:
        return {
            "min_score": float(self.min_score),
            "templates": [
                {"label": str(label), "canvas": canvas.astype(int).tolist()}
                for label, canvas in zip(self.labels.tolist(), self.canvases)
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TemplateLibrary":
        templates = payload.get("templates", [])
        labels: List[str] = []
        canvases: List[np.ndarray] = []
        for item in templates:
            label = normalize_token(str(item.get("label", "")))
            canvas_values = item.get("canvas")
            if not label or not isinstance(canvas_values, list):
                continue
            canvas = np.array(canvas_values, dtype=np.uint8)
            if canvas.size != _GLYPH_CANVAS_SIZE * _GLYPH_CANVAS_SIZE:
                continue
            labels.append(label)
            canvases.append(canvas.reshape(-1))
        if not labels:
            return cls.empty(min_score=float(payload.get("min_score", DEFAULT_TEMPLATE_MATCH_THRESHOLD)))
        return cls(
            labels=np.array(labels),
            canvases=np.stack(canvases).astype(np.uint8),
            min_score=float(payload.get("min_score", DEFAULT_TEMPLATE_MATCH_THRESHOLD)),
        )


@dataclass(frozen=True)
class TileKNNClassifier:
    labels: np.ndarray
    canvases: np.ndarray
    k: int = 5

    @classmethod
    def empty(cls, k: int = 5) -> "TileKNNClassifier":
        return cls(
            labels=np.array([], dtype="<U2"),
            canvases=np.zeros((0, _GLYPH_CANVAS_SIZE * _GLYPH_CANVAS_SIZE), dtype=np.float32),
            k=k,
        )

    @classmethod
    def from_template_library(cls, library: TemplateLibrary, k: int = 5) -> "TileKNNClassifier":
        if library.is_empty:
            return cls.empty(k=k)
        return cls(
            labels=library.labels.copy(),
            canvases=library.canvases.astype(np.float32),
            k=k,
        )

    @property
    def is_empty(self) -> bool:
        return self.canvases.size == 0 or self.labels.size == 0

    def predict_tile(self, tile_image: np.ndarray) -> Tuple[str, float]:
        canvas = _prepare_token_canvas(tile_image)
        if canvas is None:
            return "", 0.0
        return self.predict_canvas(canvas)

    def predict_canvas(self, canvas: np.ndarray) -> Tuple[str, float]:
        if self.is_empty:
            return "", 0.0

        vec = canvas.reshape(1, -1).astype(np.float32)
        distances = ((self.canvases - vec) ** 2).sum(axis=1)
        neighbor_count = max(1, min(self.k, distances.shape[0]))
        neighbor_idx = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]

        weights: dict[str, float] = {}
        for idx in neighbor_idx.tolist():
            label = str(self.labels[idx])
            weight = 1.0 / (float(distances[idx]) + 1.0)
            weights[label] = weights.get(label, 0.0) + weight

        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        best_label, best_weight = ranked[0]
        second_weight = ranked[1][1] if len(ranked) > 1 else 0.0
        total_weight = sum(weights.values()) + 1e-6
        confidence = min(1.0, (best_weight / total_weight) + max(0.0, best_weight - second_weight) / total_weight)
        return best_label, float(confidence)


def normalize_token(raw_token: str) -> str:
    letters = "".join(ch for ch in (raw_token or "").translate(_OCR_CHAR_SUBSTITUTIONS) if ch.isalpha())
    if not letters:
        return ""
    if len(letters) == 1:
        return letters.upper()
    return letters[:2].upper()


def _normalize_with_quality(raw_token: str) -> Tuple[str, bool]:
    letters = "".join(ch for ch in (raw_token or "").translate(_OCR_CHAR_SUBSTITUTIONS) if ch.isalpha())
    if not letters:
        return "", True
    if len(letters) == 1:
        return letters.upper(), False
    normalized = letters[:2].upper()
    degraded = len(letters) > 2
    return normalized, degraded


def extract_tile_images(frame: np.ndarray, calibration: CalibrationConfig) -> List[np.ndarray]:
    rects = generate_tile_crop_rects(calibration, relative_to_roi=True)
    tiles: List[np.ndarray] = []
    height, width = frame.shape[:2]
    for left, top, tile_w, tile_h in rects:
        right = left + tile_w
        bottom = top + tile_h
        if left < 0 or top < 0 or right > width or bottom > height:
            raise ValueError("Tile crop rectangle exceeded frame bounds")
        tiles.append(frame[top:bottom, left:right].copy())
    return tiles


def _prepare_tile_binary(tile_image: np.ndarray) -> np.ndarray:
    if tile_image.ndim == 2:
        gray = tile_image
    else:
        gray = cv2.cvtColor(tile_image, cv2.COLOR_RGB2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Keep dark glyph strokes and suppress small noise blobs.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    return binary


def _trim_binary(binary: np.ndarray) -> np.ndarray:
    ys, xs = np.where(binary > 0)
    if ys.size == 0 or xs.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    return binary[y0:y1, x0:x1]


def _prepare_token_canvas(tile_image: np.ndarray) -> Optional[np.ndarray]:
    binary = _prepare_tile_binary(tile_image)
    trimmed = _trim_binary(binary)
    if trimmed.size == 0:
        return None
    if float(trimmed.sum()) <= 0:
        return None
    return _fit_binary_to_canvas(trimmed)


def _split_wide_region(region: np.ndarray) -> List[np.ndarray]:
    h, w = region.shape
    if w < 12 or w < int(h * 1.2):
        return [region]

    col_sum = region.sum(axis=0).astype(np.float32)
    left = int(w * 0.2)
    right = int(w * 0.8)
    if right - left < 4:
        return [region]

    window = col_sum[left:right]
    split_offset = int(np.argmin(window))
    split_col = left + split_offset

    # Ensure a meaningful split and a real valley between two glyphs.
    valley = float(col_sum[split_col])
    peak = float(np.max(col_sum)) if col_sum.size else 0.0
    if split_col < 4 or split_col > w - 4 or peak <= 0:
        return [region]
    if valley > peak * 0.55:
        return [region]

    left_img = _trim_binary(region[:, :split_col])
    right_img = _trim_binary(region[:, split_col:])
    if left_img.size == 0 or right_img.size == 0:
        return [region]
    if left_img.shape[1] < 2 or right_img.shape[1] < 2:
        return [region]
    return [left_img, right_img]


def _extract_glyph_regions(binary: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    img_area = binary.shape[0] * binary.shape[1]
    min_area = max(10, int(img_area * 0.004))
    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area:
            continue
        if h < 5 or w < 2:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return []

    boxes.sort(key=lambda b: b[0])
    merged: List[List[int]] = []
    for x, y, w, h in boxes:
        if not merged:
            merged.append([x, y, x + w, y + h])
            continue
        prev = merged[-1]
        prev_x0, prev_y0, prev_x1, prev_y1 = prev
        gap_limit = max(2, int(min(prev_x1 - prev_x0, w) * 0.35))
        overlaps = x <= prev_x1
        close_gap = x <= prev_x1 + gap_limit
        if overlaps or close_gap:
            prev[0] = min(prev_x0, x)
            prev[1] = min(prev_y0, y)
            prev[2] = max(prev_x1, x + w)
            prev[3] = max(prev_y1, y + h)
        else:
            merged.append([x, y, x + w, y + h])

    if len(merged) > 2:
        merged.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        merged = sorted(merged[:2], key=lambda b: b[0])

    regions: List[np.ndarray] = []
    for x0, y0, x1, y1 in merged:
        glyph = _trim_binary(binary[y0:y1, x0:x1])
        if glyph.size:
            regions.extend(_split_wide_region(glyph))

    if len(regions) > 2:
        regions = sorted(regions, key=lambda r: r.shape[0] * r.shape[1], reverse=True)[:2]
    return regions


def _fit_binary_to_canvas(binary: np.ndarray, size: int = _GLYPH_CANVAS_SIZE) -> np.ndarray:
    h, w = binary.shape
    if h <= 0 or w <= 0:
        return np.zeros((size, size), dtype=np.uint8)

    pad = max(2, int(size * 0.08))
    target = size - (2 * pad)
    scale = min(target / float(max(w, 1)), target / float(max(h, 1)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = (resized > 0).astype(np.uint8)
    return canvas


@lru_cache(maxsize=1)
def _template_bank() -> Tuple[np.ndarray, np.ndarray]:
    """Build a bank of binary glyph templates.

    Uses **PIL-rendered bold system fonts** (Arial Bold, Impact, etc.) as the
    primary templates because the game uses a very bold sans-serif font.
    OpenCV Hershey fonts are kept as a fallback.
    """
    templates: List[np.ndarray] = []
    labels: List[str] = []
    render_size = _GLYPH_CANVAS_SIZE * 2  # 80
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # --- PIL-rendered bold fonts (primary, much better match) ---
    import os
    from PIL import Image as PilImage, ImageDraw, ImageFont

    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    font_candidates = [
        "arialbd.ttf",   # Arial Bold
        "impact.ttf",    # Impact (very bold condensed)
        "calibrib.ttf",  # Calibri Bold
        "segoeuib.ttf",  # Segoe UI Bold
        "verdanab.ttf",  # Verdana Bold
    ]
    pil_fonts: List[Tuple[str, int]] = []
    for fname in font_candidates:
        fp = os.path.join(fonts_dir, fname)
        if os.path.isfile(fp):
            for size in (44, 52, 60, 70):
                pil_fonts.append((fp, size))

    # Dilation kernels to thicken rendered glyphs even further (applied AFTER fitting).
    dilate_k_sizes = [0, 3, 5, 7]

    for fp, size in pil_fonts:
        try:
            font = ImageFont.truetype(fp, size)
        except Exception:
            continue
        for ch in chars:
            img = PilImage.new("L", (render_size, render_size), 0)
            draw = ImageDraw.Draw(img)
            try:
                bbox = draw.textbbox((0, 0), ch, font=font)
            except Exception:
                continue
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if tw <= 0 or th <= 0:
                continue
            x = (render_size - tw) // 2 - bbox[0]
            y = (render_size - th) // 2 - bbox[1]
            draw.text((x, y), ch, fill=255, font=font)
            
            binary_raw = (np.array(img) > 0).astype(np.uint8)
            fitted = _fit_binary_to_canvas(binary_raw)
            
            for k in dilate_k_sizes:
                if k > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    final_binary = cv2.dilate(fitted, kernel, iterations=1)
                else:
                    final_binary = fitted.copy()
                templates.append(final_binary.reshape(-1))
                labels.append(ch)

    # --- OpenCV Hershey fonts (fallback, with thick strokes) ---
    hershey_fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]
    for ch in chars:
        for hf in hershey_fonts:
            for scale in (1.0, 1.3):
                for thickness in (4, 6, 8):
                    canvas = np.zeros((render_size, render_size), dtype=np.uint8)
                    (tw, th), baseline = cv2.getTextSize(ch, hf, scale, thickness)
                    if tw <= 0 or th <= 0:
                        continue
                    x = max(0, (render_size - tw) // 2)
                    y = max(th + baseline, (render_size + th) // 2)
                    cv2.putText(canvas, ch, (x, y), hf, scale, color=255,
                                thickness=thickness, lineType=cv2.LINE_AA)
                    binary = (canvas > 0).astype(np.uint8)
                    fitted = _fit_binary_to_canvas(binary)
                    templates.append(fitted.reshape(-1))
                    labels.append(ch)

    if not templates:
        return (
            np.zeros((1, _GLYPH_CANVAS_SIZE * _GLYPH_CANVAS_SIZE), dtype=np.uint8),
            np.array([""], dtype="<U1"),
        )
    return np.stack(templates).astype(np.uint8), np.array(labels)


def _classify_glyph(binary_glyph: np.ndarray) -> Tuple[str, float]:
    glyph = _fit_binary_to_canvas(binary_glyph).reshape(-1).astype(np.uint8)
    glyph_ink = float(glyph.sum())
    if glyph_ink <= 0:
        return "", 0.0

    templates, labels = _template_bank()
    intersection = np.minimum(templates, glyph).sum(axis=1).astype(np.float32)
    union = np.maximum(templates, glyph).sum(axis=1).astype(np.float32) + 1e-6
    scores = intersection / union
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if scores.size > 1:
        second_idx = int(np.argpartition(scores, -2)[-2])
        second_score = float(scores[second_idx])
    else:
        second_score = 0.0

    margin = max(0.0, best_score - second_score)
    confidence = min(1.0, (best_score * 0.85) + (margin * 0.65))
    return str(labels[best_idx]), confidence


def _is_reasonable_digraph(token: str) -> bool:
    if len(token) != 2:
        return False
    upper = token.upper()
    return upper in _DIGRAPHS


def _read_token_from_binary(binary: np.ndarray) -> Tuple[str, float]:
    regions = _extract_glyph_regions(binary)
    if not regions:
        return "", 0.0

    if len(regions) > 2:
        regions = regions[:2]

    glyphs: List[str] = []
    confs: List[float] = []
    for region in regions:
        ch, conf = _classify_glyph(region)
        if not ch:
            return "", 0.0
        glyphs.append(ch)
        confs.append(conf)

    token = "".join(glyphs)
    confidence = float(min(confs)) if confs else 0.0
    if len(token) == 2 and not _is_reasonable_digraph(token):
        confidence *= 0.6
    return token, confidence


def _default_tile_reader(tile_image: np.ndarray, idx: int, row: int, col: int) -> Tuple[str, float]:
    del idx, row, col
    
    gray = cv2.cvtColor(tile_image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove tiny noise with morphological opening (fixes artifacts inside Plato tiles)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    
    # Pad so text doesn't touch edges (Tesseract hates edge-touching text)
    bw = cv2.copyMakeBorder(bw, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    bw = cv2.resize(bw, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # Use strict uppercase whitelist to prevent lowercase hallucination. 
    # V1 used psm 7 (treat as single text line)
    config_word = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    raw_text = pytesseract.image_to_string(bw, config=config_word).strip()
    
    if not raw_text or len(raw_text) > 2:
        # Fallback to single character mode if empty or noisy
        config_char = "--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        raw_char = pytesseract.image_to_string(bw, config=config_char).strip()
        if raw_char and len(raw_char) <= 2:
            raw_text = raw_char
            
    text_upper = raw_text.upper()
    
    # Global artifact overrides
    if text_upper == "LL": text_upper = "I"
    if text_upper == "TT": text_upper = "I"
    if text_upper in ("L", "|", "1"): text_upper = "I"
    if text_upper == "0": text_upper = "O"
    
    text = text_upper
    
    # Allow mapping pure caps to proper digraph format if it matches known valid digraphs
    if len(text) == 2:
        if not _is_reasonable_digraph(text):
            text = text[0]
        else:
            text = text[0] + text[1].lower() # Enforce An, Qu format
            
    if text:
        return text, 0.95
    return "M", 0.0  # complete fallback


@lru_cache(maxsize=4)
def load_template_library(
    path: Optional[str] = None,
    min_score: Optional[float] = None,
) -> TemplateLibrary:
    target = Path(path) if path else OCR_TEMPLATE_LIBRARY_PATH
    payload = load_json_file(target, default={"templates": []})
    library = TemplateLibrary.from_dict(payload)
    if min_score is None:
        return library
    return TemplateLibrary(
        labels=library.labels,
        canvases=library.canvases,
        min_score=min_score,
    )


@lru_cache(maxsize=4)
def load_tile_classifier(path: Optional[str] = None, k: int = 5) -> TileKNNClassifier:
    library = load_template_library(path)
    return TileKNNClassifier.from_template_library(library, k=k)


def save_template_library(library: TemplateLibrary, path: Optional[Path] = None) -> Path:
    target = path or OCR_TEMPLATE_LIBRARY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(library.to_dict(), indent=2), encoding="utf-8")
    load_template_library.cache_clear()
    load_tile_classifier.cache_clear()
    return target


def read_tile_with_consensus(
    tile_image: np.ndarray,
    idx: int,
    row: int,
    col: int,
    *,
    template_library: Optional[TemplateLibrary] = None,
    classifier: Optional[TileKNNClassifier] = None,
    local_reader: Optional[TileReader] = None,
    template_min_score: Optional[float] = None,
) -> Tuple[str, float, str]:
    library = template_library if template_library is not None else load_template_library()
    threshold = template_min_score if template_min_score is not None else library.min_score
    if library is not None and not library.is_empty:
        token, score, margin = library.match_tile(tile_image)
        if token and score >= threshold and margin >= 0.02:
            return token, score, "template"

    active_classifier = (
        classifier
        if classifier is not None
        else TileKNNClassifier.from_template_library(library)
    )
    token, confidence = active_classifier.predict_tile(tile_image)
    if token:
        return token, confidence, "classifier"

    reader = local_reader or _default_tile_reader
    token, confidence = reader(tile_image, idx, row, col)
    return token, confidence, "local_ocr"


def build_runtime_tile_reader() -> TileReader:
    return _default_tile_reader


def create_runtime_tile_reader() -> TileReader:
    return build_runtime_tile_reader()


def get_runtime_tile_reader() -> TileReader:
    return build_runtime_tile_reader()


runtime_tile_reader: TileReader = _default_tile_reader


def save_ocr_debug_overlay(
    frame: np.ndarray,
    calibration: CalibrationConfig,
    tile_results: List[OCRTileResult],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)

    cell_w = calibration.roi_width / calibration.grid_size
    cell_h = calibration.roi_height / calibration.grid_size
    for tile in tile_results:
        left = int(round(tile.col * cell_w))
        top = int(round(tile.row * cell_h))
        right = int(round((tile.col + 1) * cell_w))
        bottom = int(round((tile.row + 1) * cell_h))
        color = "red" if tile.low_confidence else "green"
        draw.rectangle([(left, top), (right, bottom)], outline=color, width=2)
        label = f"{tile.index}:{tile.normalized_token or '?'}"
        draw.text((left + 4, top + 4), label, fill=color)

    out_path = out_dir / f"ocr_{calibration.calibration_id}_{calibration.grid_size}x{calibration.grid_size}.png"
    image.save(out_path)
    return out_path


def ocr_board(
    frame: np.ndarray,
    calibration: CalibrationConfig,
    tile_reader: Optional[TileReader] = None,
    min_confidence: float = DEFAULT_LOCAL_OCR_CONFIDENCE_THRESHOLD,
    template_library: Optional[TemplateLibrary] = None,
    debug_dir: Optional[Path] = None,
) -> OCRBoardResult:
    tile_images = extract_tile_images(frame, calibration)
    tiles: List[OCRTileResult] = []
    normalized_grid = [["" for _ in range(calibration.grid_size)] for _ in range(calibration.grid_size)]
    low_confidence_found = False
    template_match_count = 0
    local_ocr_count = 0

    for center in calibration.tile_centers:
        tile_img = tile_images[center.index]
        if tile_reader is not None:
            raw, confidence = tile_reader(tile_img, center.index, center.row, center.col)
            source_method = "local_ocr"
        else:
            raw, confidence, source_method = read_tile_with_consensus(
                tile_img,
                center.index,
                center.row,
                center.col,
                template_library=template_library,
                local_reader=None,
            )
        normalized, degraded = _normalize_with_quality(raw)
        low_confidence = (
            confidence < min_confidence
            or normalized == ""
            or degraded
            or not re.match(r"^[A-Z]{1,2}$", normalized)
        )
        tile_result = OCRTileResult(
            index=center.index,
            row=center.row,
            col=center.col,
            raw_token=str(raw),
            normalized_token=normalized,
            confidence=float(confidence),
            low_confidence=low_confidence,
            source_method=source_method,
        )
        tiles.append(tile_result)
        normalized_grid[center.row][center.col] = normalized
        low_confidence_found = low_confidence_found or low_confidence
        if source_method == "template":
            template_match_count += 1
        elif source_method in {"classifier", "local_ocr"}:
            local_ocr_count += 1

    debug_overlay_path = None
    if debug_dir is not None:
        debug_overlay_path = str(
            save_ocr_debug_overlay(
                frame=frame,
                calibration=calibration,
                tile_results=tiles,
                out_dir=Path(debug_dir),
            )
        )

    return OCRBoardResult(
        calibration_id=calibration.calibration_id,
        grid_size=calibration.grid_size,
        tiles=tiles,
        normalized_grid=normalized_grid,
        has_low_confidence=low_confidence_found,
        debug_overlay_path=debug_overlay_path,
        template_match_count=template_match_count,
        local_ocr_count=local_ocr_count,
    )


# ---------------------------------------------------------------------------
# Auto-detect pipeline: circle-based tile extraction
# ---------------------------------------------------------------------------

def extract_tile_images_from_circles(
    image_bgr: np.ndarray,
    board: DetectedBoard,
    crop_factor: float = 0.92,
) -> List[np.ndarray]:
    """Extract tile images by cropping a square inside each detected circle.

    *crop_factor* controls how deeply inside the circle we crop.
    The returned images are in **RGB** order for compatibility with the
    existing OCR tile reader.
    """
    h, w = image_bgr.shape[:2]
    tiles: List[np.ndarray] = []
    for tile in board.tiles:
        r = max(18, int(tile.radius * crop_factor))
        y0 = max(0, tile.cy - r)
        y1 = min(h, tile.cy + r)
        x0 = max(0, tile.cx - r)
        x1 = min(w, tile.cx + r)
        crop = image_bgr[y0:y1, x0:x1].copy()
        # Convert BGR → RGB for the existing tile reader
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tiles.append(crop_rgb)
    return tiles


def ocr_board_auto(
    image_bgr: np.ndarray,
    board: DetectedBoard,
    tile_reader: Optional[TileReader] = None,
    min_confidence: float = DEFAULT_LOCAL_OCR_CONFIDENCE_THRESHOLD,
    template_library: Optional[TemplateLibrary] = None,
    debug_dir: Optional[Path] = None,
) -> OCRBoardResult:
    """OCR the board using auto-detected tile positions.

    Unlike :func:`ocr_board` this does **not** require a
    :class:`CalibrationConfig`.  Tile images are extracted from the detected
    circle centres instead of a rectangular grid.
    """
    tile_images = extract_tile_images_from_circles(image_bgr, board)
    tiles_out: List[OCRTileResult] = []
    grid = [
        ["" for _ in range(board.grid_size)]
        for _ in range(board.grid_size)
    ]
    low_confidence_found = False
    template_match_count = 0
    local_ocr_count = 0

    for det_tile, tile_img in zip(board.tiles, tile_images):
        raw, confidence, source_method = read_tile_with_consensus(
            tile_img,
            det_tile.index,
            det_tile.row,
            det_tile.col,
            template_library=template_library,
            local_reader=tile_reader,
        )
        normalized, degraded = _normalize_with_quality(raw)
        low_conf = (
            confidence < min_confidence
            or normalized == ""
            or degraded
            or not re.match(r"^[A-Z]{1,2}$", normalized)
        )
        tiles_out.append(OCRTileResult(
            index=det_tile.index,
            row=det_tile.row,
            col=det_tile.col,
            raw_token=str(raw),
            normalized_token=normalized,
            confidence=float(confidence),
            low_confidence=low_conf,
            source_method=source_method,
        ))
        grid[det_tile.row][det_tile.col] = normalized
        low_confidence_found = low_confidence_found or low_conf
        if source_method == "template":
            template_match_count += 1
        elif source_method in {"classifier", "local_ocr"}:
            local_ocr_count += 1

    debug_overlay_path = None
    if debug_dir is not None:
        debug_overlay_path = str(
            _save_auto_debug_overlay(
                image_bgr=image_bgr,
                board=board,
                tile_results=tiles_out,
                out_dir=Path(debug_dir),
            )
        )

    return OCRBoardResult(
        calibration_id="auto",
        grid_size=board.grid_size,
        tiles=tiles_out,
        normalized_grid=grid,
        has_low_confidence=low_confidence_found,
        debug_overlay_path=debug_overlay_path,
        template_match_count=template_match_count,
        local_ocr_count=local_ocr_count,
    )


def ocr_board_nvidia(
    image_bgr: np.ndarray,
    board: DetectedBoard,
    api_key: str,
    api_base_url: str,
    model: str,
    debug_dir: Optional[Path] = None,
) -> OCRBoardResult:
    """OCR the board via NVIDIA vision LLM.

    Sends the full screenshot to the NVIDIA API and parses the returned grid.
    Falls back to :func:`ocr_board_auto` (Tesseract) if the API call fails.
    """
    import base64
    import urllib.request

    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode()
    n = board.grid_size

    prompt = (
        f"This image shows a {n}x{n} word-game board of circular tiles on an orange/pink background. "
        "Each tile has 1 or 2 uppercase letters. Digraphs (2 letters on one tile) include: "
        "TH, HE, QU, IN, AN, ER and similar common pairs. "
        f"Read every tile left-to-right top-to-bottom and return ONLY a JSON array of {n} arrays of {n} "
        "uppercase strings. No explanation, no markdown fences, just the JSON array."
    )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "max_tokens": 512,
        "temperature": 0,
    }).encode()

    grid: Optional[List[List[str]]] = None
    try:
        req = urllib.request.Request(
            f"{api_base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"].strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > 1:
            parsed = json.loads(text[start:end])
            if isinstance(parsed, list) and len(parsed) == n:
                grid = [[str(tok).upper()[:2].strip() for tok in row] for row in parsed]
    except Exception as exc:
        print(f"[ocr_nvidia] API error: {exc} — falling back to Tesseract")

    if grid is None:
        return ocr_board_auto(image_bgr, board, debug_dir=debug_dir)

    # Build OCRBoardResult from the parsed grid, aligned to detected tile positions
    tiles_out: List[OCRTileResult] = []
    for det_tile in board.tiles:
        token = grid[det_tile.row][det_tile.col] if det_tile.row < n and det_tile.col < n else ""
        valid = bool(re.match(r"^[A-Z]{1,2}$", token))
        tiles_out.append(OCRTileResult(
            index=det_tile.index,
            row=det_tile.row,
            col=det_tile.col,
            raw_token=token,
            normalized_token=token,
            confidence=1.0 if valid else 0.0,
            low_confidence=not valid,
            source_method="nvidia_ocr",
        ))

    debug_overlay_path = None
    if debug_dir is not None:
        debug_overlay_path = str(
            _save_auto_debug_overlay(
                image_bgr=image_bgr,
                board=board,
                tile_results=tiles_out,
                out_dir=Path(debug_dir),
            )
        )

    return OCRBoardResult(
        calibration_id="auto-nvidia",
        grid_size=n,
        tiles=tiles_out,
        normalized_grid=grid,
        has_low_confidence=any(t.low_confidence for t in tiles_out),
        debug_overlay_path=debug_overlay_path,
        template_match_count=0,
        local_ocr_count=0,
    )


def _save_auto_debug_overlay(
    image_bgr: np.ndarray,
    board: DetectedBoard,
    tile_results: List[OCRTileResult],
    out_dir: Path,
) -> Path:
    """Draw detected circles + OCR results on the image for debugging."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image)

    for tile, ocr in zip(board.tiles, tile_results):
        r = tile.radius
        left = tile.cx - r
        top = tile.cy - r
        right = tile.cx + r
        bottom = tile.cy + r
        colour = "red" if ocr.low_confidence else "lime"
        draw.ellipse([(left, top), (right, bottom)], outline=colour, width=3)
        label = f"{ocr.index}:{ocr.normalized_token or '?'}"
        draw.text(
            (tile.cx - r, tile.cy - r - 16),
            label,
            fill=colour,
        )

    out_path = out_dir / f"ocr_auto_{board.grid_size}x{board.grid_size}.png"
    image.save(out_path)
    return out_path
