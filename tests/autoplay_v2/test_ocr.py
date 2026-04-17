from pathlib import Path

import numpy as np

from autoplay_v2.calibration import create_calibration
from autoplay_v2.ocr import normalize_token, ocr_board


def test_normalize_token_supports_letters_and_digraphs():
    assert normalize_token("a") == "A"
    assert normalize_token("Th") == "TH"
    assert normalize_token(" qu ") == "QU"
    assert normalize_token("H3") == "H"
    assert normalize_token("") == ""


def test_ocr_board_supports_5x5_and_keeps_digraphs(tmp_path: Path):
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=500,
        roi_height=500,
        grid_size=5,
        calibration_id="cal-5",
    )
    frame = np.full((500, 500, 3), 200, dtype=np.uint8)

    tokens = ["Th", "He", "Qu"] + ["A"] * 22

    def reader(tile_image, idx, row, col):
        return tokens[idx], 0.95

    result = ocr_board(frame, calibration, tile_reader=reader, debug_dir=tmp_path)
    assert result.grid_size == 5
    assert len(result.tiles) == 25
    assert result.normalized_grid[0][:3] == ["TH", "HE", "QU"]
    assert result.has_low_confidence is False
    assert result.debug_overlay_path is not None
    assert Path(result.debug_overlay_path).exists()


def test_ocr_board_supports_4x4():
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=400,
        roi_height=400,
        grid_size=4,
        calibration_id="cal-4",
    )
    frame = np.full((400, 400, 3), 180, dtype=np.uint8)

    def reader(tile_image, idx, row, col):
        return "B", 0.8

    result = ocr_board(frame, calibration, tile_reader=reader)
    assert result.grid_size == 4
    assert len(result.normalized_grid) == 4
    assert all(len(row) == 4 for row in result.normalized_grid)


def test_ocr_board_marks_low_confidence_for_unusable_tokens():
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=500,
        roi_height=500,
        grid_size=5,
        calibration_id="cal-low",
    )
    frame = np.full((500, 500, 3), 150, dtype=np.uint8)

    def reader(tile_image, idx, row, col):
        if idx == 0:
            return "???", 0.99
        return "A", 0.9

    result = ocr_board(frame, calibration, tile_reader=reader)
    assert result.has_low_confidence is True
    assert any(tile.low_confidence for tile in result.tiles)
