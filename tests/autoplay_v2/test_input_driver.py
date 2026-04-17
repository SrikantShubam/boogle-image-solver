from autoplay_v2.calibration import create_calibration
from autoplay_v2.input_driver import (
    generate_adb_swipe_commands,
    path_to_screen_coordinates,
    playback_word,
)
from autoplay_v2.models import SolvedWord


def test_path_to_screen_coordinates_is_deterministic():
    calibration = create_calibration(
        roi_left=100,
        roi_top=200,
        roi_width=400,
        roi_height=400,
        grid_size=4,
        calibration_id="coord-cal",
    )
    coords = path_to_screen_coordinates([0, 1, 5], calibration)
    assert coords == [[150, 250], [250, 250], [250, 350]]


def test_generate_adb_swipe_commands_from_tile_path():
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=400,
        roi_height=400,
        grid_size=4,
        calibration_id="cmd-cal",
    )
    commands = generate_adb_swipe_commands([0, 1, 5], calibration, segment_duration_ms=75)
    assert len(commands) == 2
    assert commands[0] == ["adb", "shell", "input", "swipe", "50", "50", "150", "50", "75"]


def test_playback_word_dry_run_does_not_dispatch_commands():
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=400,
        roi_height=400,
        grid_size=4,
        calibration_id="dry-cal",
    )
    solved = SolvedWord(word="TEST", path=[0, 1, 5], score=2.0, length=4, token_count=4)

    called = {"count": 0}

    def runner(cmd):
        called["count"] += 1
        return 0

    attempt = playback_word(
        solved_word=solved,
        calibration=calibration,
        dry_run=True,
        command_runner=runner,
    )
    assert attempt.status == "dry_run"
    assert called["count"] == 0
    assert len(attempt.coordinates) == 3
