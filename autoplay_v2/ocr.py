from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from autoplay_v2.calibration import generate_tile_crop_rects
from autoplay_v2.models import CalibrationConfig, OCRBoardResult, OCRTileResult

TileReader = Callable[[np.ndarray, int, int, int], Tuple[str, float]]


def normalize_token(raw_token: str) -> str:
    letters = "".join(ch for ch in (raw_token or "") if ch.isalpha())
    if not letters:
        return ""
    if len(letters) == 1:
        return letters.upper()
    return letters[:2].upper()


def _normalize_with_quality(raw_token: str) -> Tuple[str, bool]:
    letters = "".join(ch for ch in (raw_token or "") if ch.isalpha())
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


def _default_tile_reader(tile_image: np.ndarray, idx: int, row: int, col: int) -> Tuple[str, float]:
    del tile_image, idx, row, col
    # Runtime integration should provide a concrete OCR reader.
    return "", 0.0


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
    min_confidence: float = 0.35,
    debug_dir: Optional[Path] = None,
) -> OCRBoardResult:
    reader = tile_reader or _default_tile_reader
    tile_images = extract_tile_images(frame, calibration)
    tiles: List[OCRTileResult] = []
    normalized_grid = [["" for _ in range(calibration.grid_size)] for _ in range(calibration.grid_size)]
    low_confidence_found = False

    for center in calibration.tile_centers:
        tile_img = tile_images[center.index]
        raw, confidence = reader(tile_img, center.index, center.row, center.col)
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
        )
        tiles.append(tile_result)
        normalized_grid[center.row][center.col] = normalized
        low_confidence_found = low_confidence_found or low_confidence

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
    )
