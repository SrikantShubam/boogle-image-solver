from pathlib import Path

import numpy as np
import pytest

from autoplay_v2.calibration import create_calibration
from autoplay_v2.hotkey import DualHotkeyRuntime, HotkeyTrigger
from autoplay_v2.models import CapturedFrame, OCRBoardResult, OCRTileResult, SolvedWord, SwipeAttempt
from autoplay_v2.session import run_once


def _fake_ocr_result(calibration_id: str, grid_size: int = 5, low_confidence: bool = False):
    tiles = []
    for idx in range(grid_size * grid_size):
        row, col = divmod(idx, grid_size)
        tiles.append(
            OCRTileResult(
                index=idx,
                row=row,
                col=col,
                raw_token="A",
                normalized_token="A",
                confidence=0.99,
                low_confidence=low_confidence and idx == 0,
            )
        )
    return OCRBoardResult(
        calibration_id=calibration_id,
        grid_size=grid_size,
        tiles=tiles,
        normalized_grid=[["A"] * grid_size for _ in range(grid_size)],
        has_low_confidence=low_confidence,
        debug_overlay_path=None,
    )


def test_run_once_orchestration_order(tmp_path: Path):
    order = []
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=500,
        roi_height=500,
        grid_size=5,
        calibration_id="session-cal",
    )
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    solved = [SolvedWord(word="TEST", path=[0, 1, 2, 3], score=2.0, length=4, token_count=4)]

    def capture_fn(*args, **kwargs):
        order.append("capture")
        return CapturedFrame(
            calibration_id="session-cal",
            captured_at="2026-04-17T10:00:00+00:00",
            frame=frame,
            source="fixture",
        )

    def ocr_fn(*args, **kwargs):
        order.append("ocr")
        return _fake_ocr_result("session-cal")

    def solve_fn(*args, **kwargs):
        order.append("solve")
        return solved

    def rank_fn(items):
        order.append("rank")
        return items

    def playback_fn(*args, **kwargs):
        order.append("playback")
        return SwipeAttempt(
            word="TEST",
            path=[0, 1, 2, 3],
            coordinates=[[10, 10], [20, 10]],
            duration_ms=80,
            status="failed",
            message="failed",
        )

    def feedback_fn(**kwargs):
        order.append("feedback")
        return kwargs

    artifact = run_once(
        calibration=calibration,
        words={"TEST"},
        fixture_path=None,
        dry_run=False,
        runs_dir=tmp_path,
        deps={
            "capture_fn": capture_fn,
            "ocr_fn": ocr_fn,
            "solve_fn": solve_fn,
            "rank_fn": rank_fn,
            "playback_fn": playback_fn,
            "feedback_fn": feedback_fn,
        },
    )
    assert artifact.calibration_id == "session-cal"
    assert order == ["capture", "ocr", "solve", "rank", "playback", "feedback"]


def test_run_once_aborts_on_low_confidence_ocr(tmp_path: Path):
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=500,
        roi_height=500,
        grid_size=5,
        calibration_id="session-low",
    )
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    called = {"solve": 0}

    def capture_fn(*args, **kwargs):
        return CapturedFrame(
            calibration_id="session-low",
            captured_at="2026-04-17T10:00:00+00:00",
            frame=frame,
            source="fixture",
        )

    def ocr_fn(*args, **kwargs):
        return _fake_ocr_result("session-low", low_confidence=True)

    def solve_fn(*args, **kwargs):
        called["solve"] += 1
        return []

    artifact = run_once(
        calibration=calibration,
        words={"TEST"},
        runs_dir=tmp_path,
        deps={
            "capture_fn": capture_fn,
            "ocr_fn": ocr_fn,
            "solve_fn": solve_fn,
        },
    )
    assert called["solve"] == 0
    assert artifact.notes == "ocr_low_confidence_abort"


def test_run_once_requires_runtime_reader_for_default_ocr(tmp_path: Path, monkeypatch):
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=500,
        roi_height=500,
        grid_size=5,
        calibration_id="session-reader",
    )
    frame = np.zeros((500, 500, 3), dtype=np.uint8)

    def capture_fn(*args, **kwargs):
        return CapturedFrame(
            calibration_id="session-reader",
            captured_at="2026-04-17T10:00:00+00:00",
            frame=frame,
            source="fixture",
        )

    def no_reader(*args, **kwargs):
        raise RuntimeError("no runtime reader configured")

    monkeypatch.setattr("autoplay_v2.session._resolve_runtime_tile_reader", no_reader)
    with pytest.raises(RuntimeError, match="no runtime reader configured"):
        run_once(
            calibration=calibration,
            words={"TEST"},
            runs_dir=tmp_path,
            deps={
                "capture_fn": capture_fn,
            },
        )


def test_failed_word_logging_on_playback_rejection(tmp_path: Path):
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=500,
        roi_height=500,
        grid_size=5,
        calibration_id="session-reject",
    )
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    feedback_rows = []

    def capture_fn(*args, **kwargs):
        return CapturedFrame(
            calibration_id="session-reject",
            captured_at="2026-04-17T10:00:00+00:00",
            frame=frame,
            source="fixture",
        )

    def playback_fn(*args, **kwargs):
        return SwipeAttempt(
            word="TEST",
            path=[0, 1, 2, 3],
            coordinates=[[10, 10], [20, 20]],
            duration_ms=90,
            status="failed",
            message="device rejected",
        )

    def feedback_fn(**kwargs):
        feedback_rows.append(kwargs)
        return kwargs

    run_once(
        calibration=calibration,
        words={"TEST"},
        dry_run=False,
        runs_dir=tmp_path,
        deps={
            "capture_fn": capture_fn,
            "ocr_fn": lambda *a, **k: _fake_ocr_result("session-reject"),
            "solve_fn": lambda *a, **k: [
                SolvedWord(word="TEST", path=[0, 1, 2, 3], score=2.0, length=4, token_count=4)
            ],
            "rank_fn": lambda items: items,
            "playback_fn": playback_fn,
            "feedback_fn": feedback_fn,
        },
    )
    assert feedback_rows
    assert feedback_rows[0]["status"] == "rejected"


def test_hotkey_trigger_calls_session_once_when_reentered():
    calls = {"count": 0}
    hotkey = None

    def run_once_fn():
        calls["count"] += 1
        hotkey.trigger()

    hotkey = HotkeyTrigger(run_once_fn)
    hotkey.trigger()
    assert calls["count"] == 1


def test_dual_hotkey_runtime_dispatches_both_actions():
    calls = []
    runtime = DualHotkeyRuntime(
        play_once_fn=lambda: calls.append("play"),
        calibrate_fn=lambda: calls.append("calibrate"),
    )
    runtime.trigger_calibrate()
    runtime.trigger_play()
    assert calls == ["calibrate", "play"]


def test_hotkey_runtime_rejects_non_windows(monkeypatch):
    from autoplay_v2 import hotkey as hotkey_module

    monkeypatch.setattr(hotkey_module.platform, "system", lambda: "Linux")
    try:
        hotkey_module.run_hotkey_loop(lambda: None, lambda: None)
    except RuntimeError as exc:
        assert "Windows only" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for non-Windows hotkey runtime")
