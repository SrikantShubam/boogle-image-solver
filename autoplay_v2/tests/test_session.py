import pathlib
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].parent))

from autoplay_v2.models import DetectedBoard, DetectedTile, OCRBoardResult, SolvedWord, SwipeAttempt
from autoplay_v2.session import auto_play_loop


class SessionTimingTests(unittest.TestCase):
    def test_auto_play_loop_uses_speed_profile_timing(self) -> None:
        board = DetectedBoard(
            grid_size=4,
            tiles=[
                DetectedTile(index=0, row=0, col=0, cx=100, cy=100, radius=20),
                DetectedTile(index=1, row=0, col=1, cx=160, cy=100, radius=20),
            ],
            roi_left=0,
            roi_top=0,
            roi_width=200,
            roi_height=200,
        )
        ranked_words = [
            SolvedWord(word="CAT", path=[0, 1], score=10, length=3, token_count=3),
            SolvedWord(word="CAR", path=[1, 0], score=9, length=3, token_count=3),
        ]
        ocr = OCRBoardResult(
            calibration_id="auto",
            grid_size=4,
            tiles=[],
            normalized_grid=[["C", "A"], ["T", "R"]],
            has_low_confidence=False,
        )
        played_attempt = SwipeAttempt(
            word="CAT",
            path=[0, 1],
            coordinates=[[100, 100], [160, 100]],
            duration_ms=18,
            status="played",
        )

        with patch("autoplay_v2.session.ensure_runtime_dirs"), patch(
            "autoplay_v2.session.auto_detect_and_solve",
            return_value={"board": board, "ranked_words": ranked_words, "ocr": ocr},
        ), patch("autoplay_v2.session.playback_word_auto", return_value=played_attempt) as playback_mock, patch(
            "autoplay_v2.session.time.sleep"
        ) as sleep_mock, patch("pathlib.Path.mkdir"), patch(
            "pathlib.Path.write_text"
        ):
            auto_play_loop(stop_event=threading.Event(), dry_run=False)

        self.assertEqual(playback_mock.call_count, 2)
        self.assertEqual(playback_mock.call_args.kwargs["step_delay_ms"], 3)
        sleep_mock.assert_called_once_with(0.005)

    def test_auto_play_loop_skips_blacklisted_motif_without_dispatch(self) -> None:
        board = DetectedBoard(
            grid_size=2,
            tiles=[
                DetectedTile(index=0, row=0, col=0, cx=100, cy=100, radius=20),
                DetectedTile(index=1, row=0, col=1, cx=160, cy=100, radius=20),
                DetectedTile(index=2, row=1, col=0, cx=100, cy=160, radius=20),
                DetectedTile(index=3, row=1, col=1, cx=160, cy=160, radius=20),
            ],
            roi_left=0,
            roi_top=0,
            roi_width=220,
            roi_height=220,
        )
        ranked_words = [
            SolvedWord(word="CAT", path=[0, 1, 3], score=10, length=3, token_count=3),
        ]
        ocr = OCRBoardResult(
            calibration_id="auto",
            grid_size=2,
            tiles=[],
            normalized_grid=[["C", "A"], ["T", "R"]],
            has_low_confidence=False,
        )

        with patch("autoplay_v2.session.ensure_runtime_dirs"), patch(
            "autoplay_v2.session.auto_detect_and_solve",
            return_value={"board": board, "ranked_words": ranked_words, "ocr": ocr},
        ), patch(
            "autoplay_v2.session.load_path_blacklist",
            return_value={
                "signatures": {},
                "motifs": {
                    "+0+1>+1+0": {"motif": "+0+1>+1+0", "fail_rate": 0.38}
                },
            },
        ), patch(
            "autoplay_v2.session.playback_word_auto",
            side_effect=AssertionError("blacklisted path should not be dispatched"),
        ), patch(
            "autoplay_v2.session._write_failed_words_file",
            return_value=None,
        ), patch("pathlib.Path.mkdir"), patch(
            "pathlib.Path.write_text"
        ):
            auto_play_loop(stop_event=threading.Event(), dry_run=False)


if __name__ == "__main__":
    unittest.main()
