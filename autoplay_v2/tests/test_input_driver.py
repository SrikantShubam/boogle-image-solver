import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].parent))

from autoplay_v2 import input_driver
from autoplay_v2.models import DetectedBoard, DetectedTile, SolvedWord


class InputDriverTests(unittest.TestCase):
    def tearDown(self) -> None:
        input_driver.set_swipe_method("motionevent")

    def _blocking_diagonal_board(self) -> DetectedBoard:
        return DetectedBoard(
            grid_size=2,
            tiles=[
                DetectedTile(index=0, row=0, col=0, cx=100, cy=100, radius=60),
                DetectedTile(index=1, row=0, col=1, cx=180, cy=120, radius=60),
                DetectedTile(index=2, row=1, col=0, cx=120, cy=180, radius=60),
                DetectedTile(index=3, row=1, col=1, cx=200, cy=200, radius=60),
            ],
            roi_left=0,
            roi_top=0,
            roi_width=260,
            roi_height=260,
        )

    def test_motionevent_script_uses_input_command_and_dense_moves(self) -> None:
        coords = [(100, 200), (140, 220), (180, 240)]

        script = input_driver._build_motionevent_script(coords, step_delay_ms=20)

        self.assertIn("input motionevent DOWN 100 200", script)
        self.assertIn("input motionevent MOVE", script)
        self.assertIn("input motionevent UP 180 240", script)
        self.assertNotIn("cmd input motionevent", script)
        self.assertGreater(script.count("input motionevent MOVE"), len(coords) - 1)

    def test_planned_swipe_route_avoids_diagonal_blockers(self) -> None:
        board = self._blocking_diagonal_board()
        direct = input_driver.path_to_device_coordinates([0, 3], board)
        direct_touched = input_driver._touched_tile_sequence(
            input_driver._sample_polyline_points(direct),
            board.tiles,
        )
        planned = input_driver.plan_swipe_coordinates([0, 3], board)
        planned_touched = input_driver._touched_tile_sequence(
            input_driver._sample_polyline_points(planned),
            board.tiles,
        )

        self.assertNotEqual(direct_touched, [0, 3])
        self.assertEqual(planned_touched, [0, 3])
        self.assertGreater(len(planned), len(direct))

    def test_build_best_swipe_route_prefers_exact_non_direct_strategy(self) -> None:
        board = self._blocking_diagonal_board()

        route, touched, strategy, is_exact = input_driver.build_best_swipe_route([0, 3], board)

        self.assertTrue(is_exact)
        self.assertEqual(touched, [0, 3])
        self.assertNotEqual(strategy, "direct")
        self.assertEqual(
            input_driver._touched_tile_sequence(input_driver._sample_polyline_points(route), board.tiles),
            [0, 3],
        )

    def test_continuous_swipe_does_not_fall_back_when_motionevent_fails(self) -> None:
        coords = [(100, 200), (140, 220)]

        with patch.object(input_driver, "swipe_motionevent", return_value=False) as motion_mock, patch.object(
            input_driver, "swipe_chain", side_effect=AssertionError("fallback should not run")
        ):
            input_driver.set_swipe_method("motionevent")
            ok = input_driver.continuous_swipe(coords, step_delay_ms=18)

        self.assertFalse(ok)
        motion_mock.assert_called_once_with(coords, step_delay_ms=18)

    def test_playback_word_auto_skips_unsafe_route(self) -> None:
        board = self._blocking_diagonal_board()
        solved_word = SolvedWord(word="TEST", path=[0, 3], score=5, length=4, token_count=4)

        with patch.object(
            input_driver,
            "build_best_swipe_route",
            return_value=([(100, 100), (150, 150)], [0, 1, 3], "direct", False),
        ), patch.object(
            input_driver,
            "continuous_swipe",
            side_effect=AssertionError("unsafe route should not be played"),
        ):
            attempt = input_driver.playback_word_auto(solved_word, board, dry_run=False, step_delay_ms=18)

        self.assertEqual(attempt.status, "skipped")
        self.assertIn("Predicted swipe mismatch", attempt.message)

    def test_swipechain_method_still_uses_swipe_chain(self) -> None:
        coords = [(100, 200), (140, 220)]

        with patch.object(input_driver, "swipe_chain", return_value=True) as chain_mock:
            input_driver.set_swipe_method("swipechain")
            ok = input_driver.continuous_swipe(coords, step_delay_ms=18)

        self.assertTrue(ok)
        chain_mock.assert_called_once_with(coords, segment_duration_ms=50)


if __name__ == "__main__":
    unittest.main()
