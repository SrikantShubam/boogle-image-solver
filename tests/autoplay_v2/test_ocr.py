from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from autoplay_v2.calibration import create_calibration
from autoplay_v2.models import DetectedBoard, DetectedTile
from autoplay_v2.ocr import (
    TileKNNClassifier,
    TemplateLibrary,
    normalize_token,
    ocr_board,
    ocr_board_auto,
    read_tile_with_consensus,
)


def _build_encoded_board_frame(
    calibration,
    token_grid: List[List[str]],
) -> Tuple[np.ndarray, Dict[int, str]]:
    frame = np.zeros((calibration.roi_height, calibration.roi_width, 3), dtype=np.uint8)
    token_to_value: Dict[str, int] = {}
    next_value = 20
    cell_w = calibration.roi_width / calibration.grid_size
    cell_h = calibration.roi_height / calibration.grid_size

    for row in range(calibration.grid_size):
        for col in range(calibration.grid_size):
            token = token_grid[row][col]
            if token not in token_to_value:
                token_to_value[token] = next_value
                next_value += 7
            left = int(round(col * cell_w))
            top = int(round(row * cell_h))
            right = int(round((col + 1) * cell_w))
            bottom = int(round((row + 1) * cell_h))
            frame[top:bottom, left:right, :] = token_to_value[token]

    value_to_token = {value: token for token, value in token_to_value.items()}
    return frame, value_to_token


def _reader_from_encoded_tiles(value_to_token: Dict[int, str]):
    def reader(tile_image, idx, row, col):
        del idx, row, col
        mean_value = int(round(float(tile_image.mean())))
        closest_value = min(value_to_token.keys(), key=lambda value: abs(value - mean_value))
        return value_to_token[closest_value], 0.97

    return reader


def test_normalize_token_supports_letters_and_digraphs():
    assert normalize_token("a") == "A"
    assert normalize_token("Th") == "TH"
    assert normalize_token(" qu ") == "QU"
    assert normalize_token("hE9") == "HE"
    assert normalize_token("H3") == "HE"
    assert normalize_token("0") == "O"
    assert normalize_token("5") == "S"
    assert normalize_token("1N") == "IN"
    assert normalize_token("") == ""


def test_ocr_board_decodes_image_driven_letters_and_digraphs(tmp_path: Path):
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=500,
        roi_height=500,
        grid_size=5,
        tile_padding=0,
        calibration_id="cal-5",
    )
    token_grid = [
        ["Th", "He", "Qu", "A", "B"],
        ["C", "D", "E", "F", "G"],
        ["H", "I", "J", "K", "L"],
        ["M", "N", "O", "P", "R"],
        ["S", "T", "U", "V", "W"],
    ]
    frame, value_to_token = _build_encoded_board_frame(calibration, token_grid)
    reader = _reader_from_encoded_tiles(value_to_token)

    result = ocr_board(frame, calibration, tile_reader=reader, debug_dir=tmp_path)
    assert result.grid_size == 5
    assert len(result.tiles) == 25
    assert result.normalized_grid[0][:3] == ["TH", "HE", "QU"]
    assert result.normalized_grid[4][4] == "W"
    assert result.has_low_confidence is False
    assert result.debug_overlay_path is not None
    assert Path(result.debug_overlay_path).exists()


def test_ocr_board_supports_4x4_with_image_driven_reader():
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=400,
        roi_height=400,
        grid_size=4,
        tile_padding=0,
        calibration_id="cal-4",
    )
    token_grid = [
        ["Q", "U", "I", "Z"],
        ["B", "O", "G", "G"],
        ["L", "E", "H", "E"],
        ["A", "R", "T", "S"],
    ]
    frame, value_to_token = _build_encoded_board_frame(calibration, token_grid)
    reader = _reader_from_encoded_tiles(value_to_token)

    result = ocr_board(frame, calibration, tile_reader=reader)
    assert result.grid_size == 4
    assert len(result.normalized_grid) == 4
    assert all(len(row) == 4 for row in result.normalized_grid)
    assert result.normalized_grid[0] == ["Q", "U", "I", "Z"]
    assert result.has_low_confidence is False


def test_ocr_board_marks_low_confidence_for_degraded_digraph_tokens():
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=500,
        roi_height=500,
        grid_size=5,
        tile_padding=0,
        calibration_id="cal-low",
    )
    token_grid = [["A"] * 5 for _ in range(5)]
    token_grid[0][0] = "THE"
    frame, value_to_token = _build_encoded_board_frame(calibration, token_grid)
    reader = _reader_from_encoded_tiles(value_to_token)

    result = ocr_board(frame, calibration, tile_reader=reader)
    assert result.tiles[0].normalized_token == "TH"
    assert result.tiles[0].low_confidence is True
    assert result.has_low_confidence is True


def test_ocr_board_marks_low_confidence_when_token_confidence_too_low():
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=500,
        roi_height=500,
        grid_size=5,
        tile_padding=0,
        calibration_id="cal-conf",
    )
    token_grid = [["A"] * 5 for _ in range(5)]
    frame, value_to_token = _build_encoded_board_frame(calibration, token_grid)
    normal_reader = _reader_from_encoded_tiles(value_to_token)

    def low_conf_reader(tile_image, idx, row, col):
        token, _ = normal_reader(tile_image, idx, row, col)
        if idx == 1:
            return token, 0.2
        return token, 0.95

    result = ocr_board(frame, calibration, tile_reader=low_conf_reader)
    assert result.tiles[1].low_confidence is True
    assert result.has_low_confidence is True


def test_template_library_matches_real_tile_shape_before_local_fallback():
    tile = np.zeros((48, 48, 3), dtype=np.uint8)
    tile[10:38, 18:30, :] = 255
    library = TemplateLibrary.from_tile_images(
        [("I", tile)],
        min_score=0.90,
    )

    token, confidence, source = read_tile_with_consensus(
        tile,
        idx=0,
        row=0,
        col=0,
        template_library=library,
        local_reader=lambda *_args: ("M", 0.99),
    )

    assert token == "I"
    assert confidence >= 0.90
    assert source == "template"


def test_tile_knn_classifier_predicts_known_tile():
    tile = np.zeros((48, 48, 3), dtype=np.uint8)
    cv2.line(tile, (10, 10), (37, 37), (255, 255, 255), 5)
    cv2.line(tile, (37, 10), (10, 37), (255, 255, 255), 5)
    alt_tile = np.zeros((48, 48, 3), dtype=np.uint8)
    cv2.line(alt_tile, (11, 10), (38, 37), (255, 255, 255), 5)
    cv2.line(alt_tile, (38, 10), (11, 37), (255, 255, 255), 5)
    library = TemplateLibrary.from_tile_images(
        [("X", tile)],
        min_score=1.01,
    )
    classifier = TileKNNClassifier.from_template_library(library)

    token, confidence = classifier.predict_tile(alt_tile)

    assert token == "X"
    assert confidence > 0.5


def test_read_tile_with_consensus_uses_classifier_fallback():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.line(image, (16, 16), (47, 47), (255, 255, 255), 6)
    cv2.line(image, (47, 16), (16, 47), (255, 255, 255), 6)
    training_tile = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.line(training_tile, (16, 16), (47, 47), (255, 255, 255), 6)
    cv2.line(training_tile, (47, 16), (16, 47), (255, 255, 255), 6)
    classifier_library = TemplateLibrary.from_tile_images([("X", training_tile)], min_score=1.01)

    token, confidence, source = read_tile_with_consensus(
        image,
        idx=0,
        row=0,
        col=0,
        template_library=classifier_library,
    )

    assert token == "X"
    assert confidence > 0.5
    assert source == "classifier"
