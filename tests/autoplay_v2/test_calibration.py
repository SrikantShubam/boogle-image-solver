from autoplay_v2.calibration import (
    calibrate_interactive,
    create_calibration,
    generate_tile_centers,
    load_calibration,
    prompt_grid_size,
    select_roi_with_overlay,
    save_calibration,
)
from autoplay_v2.config import repo_root


def test_generate_tile_centers_row_major_for_5x5():
    centers = generate_tile_centers(
        roi_left=100,
        roi_top=200,
        roi_width=500,
        roi_height=500,
        grid_size=5,
    )
    assert len(centers) == 25
    assert centers[0].row == 0 and centers[0].col == 0
    assert centers[-1].row == 4 and centers[-1].col == 4
    assert centers[0].x < centers[1].x
    assert centers[0].y == centers[1].y
    assert centers[0].y < centers[5].y


def test_generate_tile_centers_row_major_for_4x4():
    centers = generate_tile_centers(
        roi_left=50,
        roi_top=80,
        roi_width=400,
        roi_height=400,
        grid_size=4,
    )
    assert len(centers) == 16
    assert centers[0].index == 0
    assert centers[-1].index == 15
    assert centers[4].row == 1 and centers[4].col == 0


def test_create_and_reload_calibration():
    out_path = repo_root() / "runs" / "_pytest_calibration.json"
    calibration = create_calibration(
        roi_left=100,
        roi_top=120,
        roi_width=500,
        roi_height=500,
        grid_size=5,
        tile_padding=10,
        emulator_label="pixel-emu",
    )
    save_calibration(calibration, out_path)
    loaded = load_calibration(out_path)
    assert loaded == calibration
    out_path.unlink(missing_ok=True)


def test_create_calibration_validates_roi():
    try:
        create_calibration(
            roi_left=0,
            roi_top=0,
            roi_width=0,
            roi_height=100,
            grid_size=5,
            emulator_label="bad",
        )
    except ValueError as exc:
        assert "positive" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for invalid ROI")


def test_prompt_grid_size_defaults_to_5():
    grid = prompt_grid_size(default_grid_size=5, input_fn=lambda _: "")
    assert grid == 5


def test_prompt_grid_size_switches_to_4():
    grid = prompt_grid_size(default_grid_size=5, input_fn=lambda _: "4")
    assert grid == 4


def test_calibrate_interactive_uses_roi_selector_and_saves(tmp_path):
    output_path = repo_root() / "runs" / "_pytest_interactive_calibration.json"

    selected = (10, 20, 300, 320)
    calibration = calibrate_interactive(
        emulator_label="android-studio",
        trigger_hotkey="shift+a+s",
        calibration_id="interactive-cal",
        tile_padding=6,
        save_path=output_path,
        roi_selector=lambda: selected,
        grid_input_fn=lambda _: "",
    )
    loaded = load_calibration(output_path)
    assert loaded.calibration_id == "interactive-cal"
    assert loaded.grid_size == 5
    assert (loaded.roi_left, loaded.roi_top, loaded.roi_width, loaded.roi_height) == selected
    output_path.unlink(missing_ok=True)


def test_select_roi_overlay_rejects_non_windows(monkeypatch):
    import autoplay_v2.calibration as calibration_module

    monkeypatch.setattr(calibration_module.platform, "system", lambda: "Linux")
    try:
        select_roi_with_overlay()
    except RuntimeError as exc:
        assert "Windows only" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for non-Windows overlay")
