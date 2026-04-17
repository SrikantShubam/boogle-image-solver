from pathlib import Path

import numpy as np
from PIL import Image

from autoplay_v2.calibration import create_calibration, save_calibration
from autoplay_v2.config import repo_root
from autoplay_v2.session import run_once


def test_e2e_fixture_flow(tmp_path: Path):
    calibration_path = repo_root() / "runs" / "_pytest_e2e_calibration.json"
    calibration = create_calibration(
        roi_left=0,
        roi_top=0,
        roi_width=500,
        roi_height=500,
        grid_size=5,
        calibration_id="e2e-cal",
    )
    save_calibration(calibration, calibration_path)

    fixture_path = tmp_path / "fixture.png"
    frame = np.full((500, 500, 3), 240, dtype=np.uint8)
    Image.fromarray(frame).save(fixture_path)

    tokens = [
        "Th",
        "E",
        "N",
        "A",
        "R",
    ] + ["A"] * 20

    def tile_reader(tile_img, idx, row, col):
        del tile_img, row, col
        return tokens[idx], 0.99

    feedback_rows = []
    artifact = run_once(
        calibration=None,
        calibration_path=calibration_path,
        words={"THEN", "THENA", "THENAR"},
        fixture_path=fixture_path,
        dry_run=True,
        runs_dir=tmp_path / "runs",
        max_words=3,
        deps={
            "tile_reader_factory": lambda: tile_reader,
            "feedback_fn": lambda **kwargs: feedback_rows.append(kwargs),
        },
    )

    assert artifact.run_id
    assert artifact.calibration_id == "e2e-cal"
    assert artifact.board_tokens[:5] == ["TH", "E", "N", "A", "R"]
    assert artifact.solved_words
    solved_words = {item.word for item in artifact.solved_words}
    assert {"THEN", "THENA", "THENAR"}.issubset(solved_words)
    assert artifact.swipe_attempts
    assert all(attempt.status == "dry_run" for attempt in artifact.swipe_attempts)

    run_file = (tmp_path / "runs" / artifact.run_id / "run.json")
    assert run_file.exists()
    assert feedback_rows
    calibration_path.unlink(missing_ok=True)
