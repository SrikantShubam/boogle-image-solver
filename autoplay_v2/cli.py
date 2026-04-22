from __future__ import annotations

import argparse
import json
import threading
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
from autoplay_v2.session import auto_detect_and_solve, auto_play_loop, run_once


# ---------------------------------------------------------------------------
# NEW: auto-detect commands
# ---------------------------------------------------------------------------

def _cmd_auto(args: argparse.Namespace) -> int:
    """One-shot: auto-detect board, OCR, solve, print results."""
    debug_dir: Optional[Path] = None
    if args.debug is not None:
        debug_dir = Path(args.debug)
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug images → {debug_dir.resolve()}")

    result = auto_detect_and_solve(
        status=lambda msg: print(msg),
        debug_dir=debug_dir,
    )
    if result is None:
        return 2
    for w in result["ranked_words"][:30]:
        print(f"  {w.word:20s}  score={w.score:.0f}  len={w.length}  path={w.path}")
    return 0


def _cmd_auto_play(args: argparse.Namespace) -> int:
    """Continuous play: auto-detect → solve → swipe all words."""
    stop = threading.Event()
    try:
        auto_play_loop(
            stop_event=stop,
            status=lambda msg: print(msg),
            dry_run=not args.live,
            max_words=args.max_words,
            word_delay=args.word_delay,
            strategy=args.strategy,
            game_duration_s=args.game_duration,
            short_word_injection=not args.no_short_word_injection,
            last_seconds_window_s=args.last_seconds_window,
            speed_solve_budget_s=args.speed_solve_budget,
            speed_max_word_len=args.speed_max_word_len,
            speed_max_candidates=args.speed_max_candidates,
            use_path_blacklist=not args.no_path_blacklist,
            path_blacklist_path=Path(args.path_blacklist) if args.path_blacklist else None,
        )
    except KeyboardInterrupt:
        stop.set()
        print("\nStopped.")
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    """Launch the floating Start/Stop GUI."""
    from autoplay_v2.gui import AutoplayGUI
    gui = AutoplayGUI(dry_run=not args.live)
    gui.run()
    return 0


# ---------------------------------------------------------------------------
# Legacy commands (kept for backward compat)
# ---------------------------------------------------------------------------

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
    try:
        artifact = run_once(
            calibration=calibration,
            fixture_path=Path(args.fixture) if args.fixture else None,
            dry_run=not args.live,
            max_words=args.max_words,
        )
    except RuntimeError as exc:
        print(f"OCR integration error: {exc}")
        return 2
    print(f"Run ID: {artifact.run_id}")
    print(f"Solved words: {len(artifact.solved_words)}")
    print(f"Swipe attempts: {len(artifact.swipe_attempts)}")
    if artifact.notes:
        print(f"Notes: {artifact.notes}")
    return 0


def _cmd_hotkey(args: argparse.Namespace) -> int:
    def _play_runner() -> None:
        calibration = load_calibration()
        try:
            artifact = run_once(
                calibration=calibration,
                dry_run=not args.live,
                max_words=args.max_words,
            )
        except RuntimeError as exc:
            print(f"[autoplay-v2] OCR integration error: {exc}")
            return
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


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoplay-v2")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- NEW commands ---
    auto = sub.add_parser("auto", help="One-shot: auto-detect → OCR → solve (print only)")
    auto.add_argument(
        "--debug",
        metavar="DIR",
        default=None,
        help="Save debug images (HSV mask, candidates, Hough) to DIR",
    )
    auto.set_defaults(func=_cmd_auto)

    auto_play = sub.add_parser("auto-play", help="Auto-detect → solve → swipe words")
    auto_play.add_argument("--live", action="store_true", help="Execute swipes (default: dry-run)")
    auto_play.add_argument("--max-words", type=int, default=None)
    auto_play.add_argument("--word-delay", type=float, default=0.04, help="Seconds between words")
    auto_play.add_argument("--strategy", choices=["speed", "balanced"], default="speed")
    auto_play.add_argument("--game-duration", type=float, default=60.0, help="Round duration in seconds")
    auto_play.add_argument(
        "--no-short-word-injection",
        action="store_true",
        help="Disable 3/4-letter injection in speed strategy",
    )
    auto_play.add_argument(
        "--last-seconds-window",
        type=float,
        default=8.0,
        help="Seconds left to switch to shortest-word sweep in speed strategy",
    )
    auto_play.add_argument(
        "--speed-solve-budget",
        type=float,
        default=None,
        help="Cap solver time in speed mode (seconds); default 2.0",
    )
    auto_play.add_argument(
        "--speed-max-word-len",
        type=int,
        default=7,
        help="Maximum word length to keep in speed mode candidate set",
    )
    auto_play.add_argument(
        "--speed-max-candidates",
        type=int,
        default=None,
        help="Maximum number of candidate words to attempt in speed mode (default 120)",
    )
    auto_play.add_argument(
        "--no-path-blacklist",
        action="store_true",
        help="Disable do-not-attempt path-signature blacklist",
    )
    auto_play.add_argument(
        "--path-blacklist",
        type=str,
        default=None,
        help="Optional explicit path blacklist JSON (default data/path_blacklist.json)",
    )
    auto_play.set_defaults(func=_cmd_auto_play)

    gui = sub.add_parser("gui", help="Launch floating Start/Stop GUI")
    gui.add_argument("--live", action="store_true", help="Execute swipes (default: dry-run)")
    gui.set_defaults(func=_cmd_gui)

    # --- Legacy commands ---
    calibrate = sub.add_parser("calibrate", help="(Legacy) Create and persist board calibration")
    calibrate.add_argument("--left", type=int, default=None)
    calibrate.add_argument("--top", type=int, default=None)
    calibrate.add_argument("--width", type=int, default=None)
    calibrate.add_argument("--height", type=int, default=None)
    calibrate.add_argument("--grid-size", type=int, choices=[4, 5], default=5)
    calibrate.add_argument("--tile-padding", type=int, default=8)
    calibrate.add_argument("--emulator-label", type=str, default="android-studio")
    calibrate.add_argument("--play-hotkey", type=str, default=DEFAULT_PLAY_HOTKEY)
    calibrate.add_argument("--calibration-id", type=str, default="default")
    calibrate.set_defaults(func=_cmd_calibrate)

    play_once = sub.add_parser("play-once", help="(Legacy) Run one calibration-based session")
    play_once.add_argument("--fixture", type=str, default=None)
    play_once.add_argument("--live", action="store_true")
    play_once.add_argument("--max-words", type=int, default=None)
    play_once.set_defaults(func=_cmd_play_once)

    hotkey = sub.add_parser("hotkey", help="(Legacy) Start hotkey runtime")
    hotkey.add_argument("--play-hotkey", type=str, default=DEFAULT_PLAY_HOTKEY)
    hotkey.add_argument("--calibrate-hotkey", type=str, default=DEFAULT_CALIBRATE_HOTKEY)
    hotkey.add_argument("--emulator-label", type=str, default="android-studio")
    hotkey.add_argument("--calibration-id", type=str, default="default")
    hotkey.add_argument("--tile-padding", type=int, default=8)
    hotkey.add_argument("--live", action="store_true")
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
