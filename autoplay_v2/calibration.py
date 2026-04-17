from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

from autoplay_v2.config import (
    CALIBRATION_PATH,
    DEFAULT_GRID_SIZE,
    DEFAULT_HOTKEY,
    DEFAULT_TILE_PADDING,
    load_json_file,
    save_json_file,
)
from autoplay_v2.models import CalibrationConfig, TileCenter


def validate_roi(roi_left: int, roi_top: int, roi_width: int, roi_height: int) -> None:
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI width and height must be positive")
    if roi_left < 0 or roi_top < 0:
        raise ValueError("ROI origin must be non-negative")


def generate_tile_centers(
    roi_left: int,
    roi_top: int,
    roi_width: int,
    roi_height: int,
    grid_size: int,
) -> List[TileCenter]:
    if grid_size not in (4, 5):
        raise ValueError("Grid size must be 4 or 5")
    validate_roi(roi_left, roi_top, roi_width, roi_height)

    cell_w = roi_width / grid_size
    cell_h = roi_height / grid_size
    centers: List[TileCenter] = []
    idx = 0
    for row in range(grid_size):
        for col in range(grid_size):
            cx = int(round(roi_left + (col + 0.5) * cell_w))
            cy = int(round(roi_top + (row + 0.5) * cell_h))
            centers.append(TileCenter(index=idx, row=row, col=col, x=cx, y=cy))
            idx += 1
    return centers


def generate_tile_crop_rects(
    calibration: CalibrationConfig,
    relative_to_roi: bool = True,
) -> List[Tuple[int, int, int, int]]:
    cell_w = calibration.roi_width / calibration.grid_size
    cell_h = calibration.roi_height / calibration.grid_size
    pad = max(0, calibration.tile_padding)
    rects: List[Tuple[int, int, int, int]] = []
    for center in calibration.tile_centers:
        left = int(round(center.col * cell_w + pad))
        top = int(round(center.row * cell_h + pad))
        right = int(round((center.col + 1) * cell_w - pad))
        bottom = int(round((center.row + 1) * cell_h - pad))
        if not relative_to_roi:
            left += calibration.roi_left
            top += calibration.roi_top
            right += calibration.roi_left
            bottom += calibration.roi_top
        width = max(1, right - left)
        height = max(1, bottom - top)
        rects.append((left, top, width, height))
    return rects


def create_calibration(
    roi_left: int,
    roi_top: int,
    roi_width: int,
    roi_height: int,
    grid_size: int = DEFAULT_GRID_SIZE,
    tile_padding: int = DEFAULT_TILE_PADDING,
    emulator_label: str = "android-studio",
    trigger_hotkey: str = DEFAULT_HOTKEY,
    calibration_id: str = "default",
) -> CalibrationConfig:
    validate_roi(roi_left, roi_top, roi_width, roi_height)
    centers = generate_tile_centers(
        roi_left=roi_left,
        roi_top=roi_top,
        roi_width=roi_width,
        roi_height=roi_height,
        grid_size=grid_size,
    )
    return CalibrationConfig(
        calibration_id=calibration_id,
        emulator_label=emulator_label,
        grid_size=grid_size,
        roi_left=roi_left,
        roi_top=roi_top,
        roi_width=roi_width,
        roi_height=roi_height,
        tile_padding=tile_padding,
        trigger_hotkey=trigger_hotkey,
        tile_centers=centers,
    )


def create_and_save_calibration(
    roi_left: int,
    roi_top: int,
    roi_width: int,
    roi_height: int,
    grid_size: int = DEFAULT_GRID_SIZE,
    tile_padding: int = DEFAULT_TILE_PADDING,
    emulator_label: str = "android-studio",
    trigger_hotkey: str = DEFAULT_HOTKEY,
    calibration_id: str = "default",
) -> CalibrationConfig:
    calibration = create_calibration(
        roi_left=roi_left,
        roi_top=roi_top,
        roi_width=roi_width,
        roi_height=roi_height,
        grid_size=grid_size,
        tile_padding=tile_padding,
        emulator_label=emulator_label,
        trigger_hotkey=trigger_hotkey,
        calibration_id=calibration_id,
    )
    save_calibration(calibration)
    return calibration


def save_calibration(calibration: CalibrationConfig, path=CALIBRATION_PATH) -> None:
    save_json_file(path, calibration.to_dict())


def load_calibration(path=CALIBRATION_PATH) -> CalibrationConfig:
    payload = load_json_file(path)
    calibration = CalibrationConfig.from_dict(payload)
    # Normalize ordering to row-major in case external edits changed order.
    ordered = sorted(calibration.tile_centers, key=lambda c: (c.row, c.col, c.index))
    if ordered != calibration.tile_centers:
        calibration = replace(calibration, tile_centers=ordered)
    return calibration
