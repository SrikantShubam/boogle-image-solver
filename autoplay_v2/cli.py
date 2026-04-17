from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from autoplay_v2.calibration import calibrate_interactive, create_and_save_calibration, load_calibration
from autoplay_v2.config import (
    AUTOPLAY_RUNS_DIR,
    CALIBRATION_PATH,
    DEFAULT_CALIBRATE_HOTKEY,
    DEFAULT_PLAY_HOTKEY,
)
from autoplay_v2.feedback import read_recent_feedback
from autoplay_v2.hotkey import run_hotkey_loop
from autoplay_v2.session import run_once


def _cmd_calibrate(args: argparse.Namespace) -> int:
    manual_mode = all(v is not None for v in (args.left, args.top, args.width, args.height))
    if manual_mode:
        calibration = create_and_save_calibration(
            roi_left=args.left,
            roi_top=args.top,
            roi_width=args.width,
            roi_height=args.height,
            grid_size=args.grid_size,
            tile_padding=args.tile_padding,
            emulator_label=args.emulator_label,
            trigger_hotkey=args.play_hotkey,
            calibration_id=args.calibration_id,
        )
    else:
        calibration = calibrate_interactive(
            emulator_label=args.emulator_label,
            trigger_hotkey=args.play_hotkey,
            calibration_id=args.calibration_id,
            tile_padding=args.tile_padding,
        )
    print(f"Saved calibration: {CALIBRATION_PATH}")
    print(json.dumps(calibration.to_dict(), indent=2))
    return 0


def _cmd_play_once(args: argparse.Namespace) -> int:
    calibration = load_calibration()
    artifact = run_once(
        calibration=calibration,
        fixture_path=Path(args.fixture) if args.fixture else None,
        dry_run=not args.live,
        max_words=args.max_words,
    )
    print(f"Run ID: {artifact.run_id}")
    print(f"Solved words: {len(artifact.solved_words)}")
    print(f"Swipe attempts: {len(artifact.swipe_attempts)}")
    if artifact.notes:
        print(f"Notes: {artifact.notes}")
    return 0


def _cmd_hotkey(args: argparse.Namespace) -> int:
    def _play_runner() -> None:
        calibration = load_calibration()
        artifact = run_once(calibration=calibration, dry_run=not args.live, max_words=args.max_words)
        print(
            f"[autoplay-v2] run={artifact.run_id} solved={len(artifact.solved_words)} "
            f"attempts={len(artifact.swipe_attempts)}"
        )

    def _calibrate_runner() -> None:
        calibration = calibrate_interactive(
            emulator_label=args.emulator_label,
            trigger_hotkey=args.play_hotkey,
            calibration_id=args.calibration_id,
            tile_padding=args.tile_padding,
        )
        print(
            f"[autoplay-v2] calibration updated id={calibration.calibration_id} "
            f"grid={calibration.grid_size} roi=({calibration.roi_left},{calibration.roi_top},"
            f"{calibration.roi_width},{calibration.roi_height})"
        )

    run_hotkey_loop(
        play_once_fn=_play_runner,
        calibrate_fn=_calibrate_runner,
        play_hotkey=args.play_hotkey,
        calibrate_hotkey=args.calibrate_hotkey,
    )
    return 0


def _cmd_review_last(args: argparse.Namespace) -> int:
    run_dirs = sorted(
        [path for path in AUTOPLAY_RUNS_DIR.glob("*") if path.is_dir()],
        key=lambda p: p.name,
    )
    if not run_dirs:
        print("No runs found.")
        return 0
    latest = run_dirs[-1]
    run_file = latest / "run.json"
    if run_file.exists():
        print(run_file.read_text(encoding="utf-8"))
    else:
        print(f"No run.json in {latest}")
    if args.show_feedback:
        entries = read_recent_feedback(limit=args.feedback_limit)
        print(json.dumps(entries, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoplay-v2")
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", help="Create and persist board calibration")
    calibrate.add_argument("--left", type=int, default=None, help="Fallback/debug manual ROI left")
    calibrate.add_argument("--top", type=int, default=None, help="Fallback/debug manual ROI top")
    calibrate.add_argument("--width", type=int, default=None, help="Fallback/debug manual ROI width")
    calibrate.add_argument("--height", type=int, default=None, help="Fallback/debug manual ROI height")
    calibrate.add_argument("--grid-size", type=int, choices=[4, 5], default=5)
    calibrate.add_argument("--tile-padding", type=int, default=8)
    calibrate.add_argument("--emulator-label", type=str, default="android-studio")
    calibrate.add_argument("--play-hotkey", type=str, default=DEFAULT_PLAY_HOTKEY)
    calibrate.add_argument("--calibration-id", type=str, default="default")
    calibrate.set_defaults(func=_cmd_calibrate)

    play_once = sub.add_parser("play-once", help="Run one capture->solve->play session")
    play_once.add_argument("--fixture", type=str, default=None)
    play_once.add_argument("--live", action="store_true", help="Use live ADB playback")
    play_once.add_argument("--max-words", type=int, default=None)
    play_once.set_defaults(func=_cmd_play_once)

    hotkey = sub.add_parser("hotkey", help="Start hotkey runtime")
    hotkey.add_argument("--play-hotkey", type=str, default=DEFAULT_PLAY_HOTKEY)
    hotkey.add_argument("--calibrate-hotkey", type=str, default=DEFAULT_CALIBRATE_HOTKEY)
    hotkey.add_argument("--emulator-label", type=str, default="android-studio")
    hotkey.add_argument("--calibration-id", type=str, default="default")
    hotkey.add_argument("--tile-padding", type=int, default=8)
    hotkey.add_argument("--live", action="store_true", help="Use live ADB playback")
    hotkey.add_argument("--max-words", type=int, default=None)
    hotkey.set_defaults(func=_cmd_hotkey)

    review = sub.add_parser("review-last", help="Inspect latest run artifact")
    review.add_argument("--show-feedback", action="store_true")
    review.add_argument("--feedback-limit", type=int, default=20)
    review.set_defaults(func=_cmd_review_last)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
