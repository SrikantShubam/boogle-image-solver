from pathlib import Path

from autoplay_v2.config import (
    AUTOPLAY_CONFIG_DIR,
    AUTOPLAY_DATA_DIR,
    AUTOPLAY_RUNS_DIR,
    DEFAULT_CALIBRATE_HOTKEY,
    DEFAULT_HOTKEY,
    DEFAULT_PLAY_HOTKEY,
    repo_root,
)
from autoplay_v2.models import CalibrationConfig, SolvedWord, TileCenter


def test_repo_paths_are_inside_repo_root():
    root = repo_root()
    assert AUTOPLAY_CONFIG_DIR == root / "config"
    assert AUTOPLAY_DATA_DIR == root / "data"
    assert AUTOPLAY_RUNS_DIR == root / "runs"
    for path in (AUTOPLAY_CONFIG_DIR, AUTOPLAY_DATA_DIR, AUTOPLAY_RUNS_DIR):
        assert root in path.parents or path == root


def test_default_hotkeys():
    assert DEFAULT_HOTKEY == "shift+a+s"
    assert DEFAULT_PLAY_HOTKEY == "shift+a+s"
    assert DEFAULT_CALIBRATE_HOTKEY == "ctrl+shift+a+s"


def test_calibration_config_round_trip():
    calibration = CalibrationConfig(
        calibration_id="cal-001",
        emulator_label="android-studio",
        grid_size=5,
        roi_left=100,
        roi_top=120,
        roi_width=500,
        roi_height=500,
        tile_padding=6,
        trigger_hotkey="shift+a+s",
        tile_centers=[
            TileCenter(index=0, row=0, col=0, x=150, y=170),
            TileCenter(index=24, row=4, col=4, x=550, y=570),
        ],
    )
    payload = calibration.to_dict()
    restored = CalibrationConfig.from_dict(payload)
    assert restored == calibration


def test_solved_word_round_trip():
    solved = SolvedWord(
        word="THEN",
        path=[0, 1, 6, 7],
        score=2.0,
        length=4,
        token_count=4,
    )
    payload = solved.to_dict()
    restored = SolvedWord.from_dict(payload)
    assert restored == solved
