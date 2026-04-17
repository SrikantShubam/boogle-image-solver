from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from autoplay_v2.calibration import create_calibration
from autoplay_v2.capture import capture_roi, save_debug_capture


def test_capture_roi_from_fixture_image(tmp_path: Path):
    fixture_path = tmp_path / "frame.png"
    arr = np.zeros((400, 600, 3), dtype=np.uint8)
    arr[100:300, 200:450] = [10, 220, 90]
    Image.fromarray(arr).save(fixture_path)

    calibration = create_calibration(
        roi_left=200,
        roi_top=100,
        roi_width=250,
        roi_height=200,
        grid_size=5,
        emulator_label="fixture",
        calibration_id="fixture-cal",
    )

    capture = capture_roi(calibration, fixture_path=fixture_path)
    assert capture.frame.shape[:2] == (200, 250)
    assert capture.calibration_id == "fixture-cal"
    datetime.fromisoformat(capture.captured_at)


def test_save_debug_capture_writes_output(tmp_path: Path):
    frame = np.full((100, 120, 3), 127, dtype=np.uint8)
    path = save_debug_capture(
        frame=frame,
        out_dir=tmp_path,
        calibration_id="cal-a",
        captured_at="2026-04-17T10:00:00+00:00",
    )
    assert path.exists()
    assert path.name.startswith("capture_cal-a_")
