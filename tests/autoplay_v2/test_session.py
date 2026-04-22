from __future__ import annotations

import numpy as np

from autoplay_v2.models import DetectedBoard, DetectedTile, OCRBoardResult, OCRTileResult, SolvedWord
from autoplay_v2.session import auto_detect_and_solve, auto_play_loop


def test_auto_detect_and_solve_uses_local_ocr_pipeline(monkeypatch):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    board = DetectedBoard(
        grid_size=1,
        tiles=[DetectedTile(index=0, row=0, col=0, cx=32, cy=32, radius=20)],
        roi_left=0,
        roi_top=0,
        roi_width=64,
        roi_height=64,
    )
    ocr_result = OCRBoardResult(
        calibration_id="auto",
        grid_size=1,
        tiles=[
            OCRTileResult(
                index=0,
                row=0,
                col=0,
                raw_token="A",
                normalized_token="A",
                confidence=0.99,
                low_confidence=False,
                source_method="template",
            )
        ],
        normalized_grid=[["A"]],
        has_low_confidence=False,
        template_match_count=1,
        local_ocr_count=0,
    )

    monkeypatch.setattr("autoplay_v2.session.detect_board", lambda *_args, **_kwargs: board)
    monkeypatch.setattr("autoplay_v2.session.ocr_board_auto", lambda *_args, **_kwargs: ocr_result)
    monkeypatch.setattr("autoplay_v2.session.build_solver_resources", lambda words: words)
    monkeypatch.setattr("autoplay_v2.session.solve_board_with_paths", lambda grid, _resources: [])
    monkeypatch.setattr("autoplay_v2.session.rank_solved_words", lambda solved: solved)
    monkeypatch.setattr(
        "autoplay_v2.session.nvidia_ocr_board",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("NVIDIA OCR should not be called")),
    )

    result = auto_detect_and_solve(
        image_bgr=image,
        words={"A"},
        status=lambda _msg: None,
    )

    assert result is not None
    assert result["ocr"].tiles[0].source_method == "template"


def test_auto_play_loop_aborts_when_consensus_ocr_is_low_confidence(monkeypatch, tmp_path):
    ocr_result = OCRBoardResult(
        calibration_id="auto",
        grid_size=1,
        tiles=[
            OCRTileResult(
                index=0,
                row=0,
                col=0,
                raw_token="M",
                normalized_token="M",
                confidence=0.10,
                low_confidence=True,
                source_method="local_ocr",
            )
        ],
        normalized_grid=[["M"]],
        has_low_confidence=True,
        template_match_count=0,
        local_ocr_count=1,
    )
    result = {
        "board": DetectedBoard(
            grid_size=1,
            tiles=[DetectedTile(index=0, row=0, col=0, cx=32, cy=32, radius=20)],
            roi_left=0,
            roi_top=0,
            roi_width=64,
            roi_height=64,
        ),
        "ocr": ocr_result,
        "ranked_words": [
            SolvedWord(word="BAD", path=[0], score=1.0, length=3, token_count=3)
        ],
        "grid_str": "M",
    }
    messages: list[str] = []

    monkeypatch.setattr("autoplay_v2.session.ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr("autoplay_v2.session.AUTOPLAY_RUNS_DIR", tmp_path)
    monkeypatch.setattr("autoplay_v2.session.auto_detect_and_solve", lambda **_kwargs: result)
    monkeypatch.setattr(
        "autoplay_v2.session.playback_word_auto",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("playback should not run")),
    )

    import threading

    auto_play_loop(
        stop_event=threading.Event(),
        status=messages.append,
        dry_run=False,
    )

    assert any("Aborted" in message for message in messages)
