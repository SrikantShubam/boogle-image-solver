"""Session orchestration for the auto-detect pipeline.

Provides:
- ``auto_detect_and_solve`` — one-shot: capture → detect → OCR → solve
- ``auto_play_loop``        — continuous: capture → solve → play all words,
                              interruptible via a ``threading.Event``.
"""
from __future__ import annotations

import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import os
from PIL import Image

from autoplay_v2.board_detector import detect_board, is_bonus_tile
from autoplay_v2.capture import capture_adb_screenshot
from autoplay_v2.config import AUTOPLAY_RUNS_DIR, ensure_runtime_dirs, repo_root
from autoplay_v2.input_driver import (
    continuous_swipe,
    path_to_device_coordinates,
    playback_word_auto,
)
from autoplay_v2.path_filters import (
    load_path_blacklist,
    path_move_signature,
    path_transition_motifs,
)
from autoplay_v2.models import (
    DetectedBoard,
    OCRBoardResult,
    RunArtifact,
    SolvedWord,
    SwipeAttempt,
)
from autoplay_v2.template_ocr import template_ocr_board
from autoplay_v2.ranking import rank_solved_words
from autoplay_v2.solver import build_solver_resources, solve_board_with_paths

# Set NVIDIA API credentials from hardcoded defaults (overridden by env vars)
if not os.environ.get('NVIDIA_API_KEY'):
    os.environ['NVIDIA_API_KEY'] = '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'
if not os.environ.get('NVIDIA_API_BASE_URL'):
    os.environ['NVIDIA_API_BASE_URL'] = 'https://integrate.api.nvidia.com/v1'

DEFAULT_WORDS_PATH = repo_root() / "words.txt"

StatusCallback = Callable[[str], None]
_WORDS_CACHE: Dict[tuple[str, int, int], Set[str]] = {}
_SOLVER_CACHE: Dict[tuple[str, int, int], Dict[str, object]] = {}


def _noop_status(_msg: str) -> None:
    pass


def _play_priority(
    word: SolvedWord,
    is_exact: bool,
) -> tuple[int, int, int, float, str, tuple[int, ...]]:
    if word.length == 3:
        length_group = 0
    elif word.length == 4:
        length_group = 1
    else:
        length_group = 2
    return (
        0 if is_exact else 1,
        length_group,
        word.length,
        -word.score,
        word.word,
        tuple(word.path),
    )


def _speed_priority(word: SolvedWord) -> tuple[int, float, int, str, tuple[int, ...]]:
    """Prefer longer/higher-yield words early for short round timers."""
    # Length-first dominates for time-constrained rounds.
    return (
        -word.length,
        -word.score,
        len(word.path),
        word.word,
        tuple(word.path),
    )


def _speed_candidate_filter(
    ranked: List[SolvedWord],
    max_len: int = 7,
) -> List[SolvedWord]:
    return [word for word in ranked if 3 <= word.length <= max_len]


def _estimate_play_time_s(word: SolvedWord, step_delay_ms: int, word_delay: float) -> float:
    # Rough but stable estimate: swipe segment time + dispatch + post-word delay.
    segments = max(1, len(word.path) - 1)
    swipe = (segments * max(1, step_delay_ms)) / 1000.0
    return max(0.08, swipe + max(0.0, word_delay) + 0.025)


def _build_speed_play_order(
    ranked: List[SolvedWord],
    short_word_injection: bool,
    inject_every: int = 3,
) -> List[SolvedWord]:
    if not short_word_injection:
        return list(ranked)
    long_words = [word for word in ranked if word.length >= 5]
    short_words = [word for word in ranked if word.length <= 4]
    if not short_words:
        return list(ranked)

    play_order: List[SolvedWord] = []
    i_long = 0
    i_short = 0
    while i_long < len(long_words) or i_short < len(short_words):
        for _ in range(inject_every):
            if i_long >= len(long_words):
                break
            play_order.append(long_words[i_long])
            i_long += 1
        if i_short < len(short_words):
            play_order.append(short_words[i_short])
            i_short += 1
        if i_long >= len(long_words):
            while i_short < len(short_words):
                play_order.append(short_words[i_short])
                i_short += 1
            break
    return play_order


def _pick_next_speed_word(
    pending: List[SolvedWord],
    remaining: float,
    last_seconds_window_s: float,
) -> SolvedWord:
    if remaining > last_seconds_window_s:
        return pending.pop(0)
    # Last-seconds sweep: prioritize fastest likely words.
    best_idx = min(
        range(len(pending)),
        key=lambda idx: (
            pending[idx].length,
            -pending[idx].score,
            pending[idx].word,
            tuple(pending[idx].path),
        ),
    )
    return pending.pop(best_idx)


def _write_failed_words_file(
    run_dir: Path,
    run_id: str,
    board_signature: str,
    attempts: List[SwipeAttempt],
) -> Optional[Path]:
    failed = [attempt for attempt in attempts if attempt.status in ("skipped", "failed")]
    if not failed:
        return None
    target = run_dir / "failed_words.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for attempt in failed:
            row = {
                "run_id": run_id,
                "board_signature": board_signature,
                "word": attempt.word,
                "status": attempt.status,
                "reason": attempt.reject_reason or "",
                "message": attempt.message,
                "path": list(attempt.path),
                "predicted_touched": list(attempt.predicted_touched),
                "route_confidence": attempt.route_confidence,
                "duration_ms": attempt.duration_ms,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return target


def _utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _load_words(words_path: Path = DEFAULT_WORDS_PATH, min_len: int = 3) -> Set[str]:
    words: Set[str] = set()
    with Path(words_path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            word = raw.strip().upper()
            if len(word) >= min_len and word.isalpha():
                words.add(word)
    return words


def _cache_key_for_words_path(words_path: Path) -> tuple[str, int, int]:
    p = Path(words_path)
    st = p.stat()
    return (str(p.resolve()), int(st.st_mtime_ns), int(st.st_size))


def _get_solver_resources(
    words: Optional[Set[str]],
    words_path: Path,
) -> tuple[Set[str], Dict[str, object]]:
    if words is not None:
        active_words = words
        resources = build_solver_resources(active_words)
        return active_words, resources

    key = _cache_key_for_words_path(words_path)
    active_words = _WORDS_CACHE.get(key)
    if active_words is None:
        active_words = _load_words(words_path=words_path)
        _WORDS_CACHE[key] = active_words
    resources = _SOLVER_CACHE.get(key)
    if resources is None:
        resources = build_solver_resources(active_words)
        _SOLVER_CACHE[key] = resources
    return active_words, resources


def _board_signature(grid: list[list[str]]) -> str:
    return "|".join(",".join(token for token in row) for row in grid)


def _grid_str(grid: list[list[str]]) -> str:
    """Pretty-print a grid for status messages."""
    return "\n".join("  ".join(tok.ljust(2) for tok in row) for row in grid)


def _ocr_quality_score(ocr_result: OCRBoardResult) -> float:
    """Higher is better; used to choose between first-pass and recapture OCR."""
    diag = ocr_result.diagnostics or {}
    final_score = float(diag.get("final_score", diag.get("candidate_score", 0.0)))
    low_conf_count = int(diag.get("low_conf_count", 0))
    invalid_count = int(diag.get("invalid_token_count", 0))
    ambiguous_count = int(diag.get("ambiguous_tile_count", 0))
    return final_score - 0.02 * low_conf_count - 0.03 * invalid_count - 0.01 * ambiguous_count



def _save_low_confidence_review_bundle(
    run_dir: Path,
    image_bgr,
    ocr_result: OCRBoardResult,
) -> Path:
    review_dir = Path(run_dir) / "low_confidence_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    screenshot_path = review_dir / "capture.png"
    Image.fromarray(image_bgr[:, :, ::-1]).save(screenshot_path)

    manual_tiles = []
    for tile in ocr_result.tiles:
        manual_tiles.append(
            {
                "index": tile.index,
                "row": tile.row,
                "col": tile.col,
                "raw_token": tile.raw_token,
                "predicted_token": tile.normalized_token,
                "confidence": tile.confidence,
                "low_confidence": tile.low_confidence,
                "manual_token": "",
                "notes": "",
            }
        )

    payload = {
        "review_required": True,
        "screenshot_path": screenshot_path.name,
        "normalized_grid": [list(row) for row in ocr_result.normalized_grid],
        "template_match_count": ocr_result.template_match_count,
        "local_ocr_count": ocr_result.local_ocr_count,
        "manual_tiles": manual_tiles,
    }

    bundle_path = review_dir / "manual_review.json"
    bundle_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return bundle_path

# ---------------------------------------------------------------------------
# One-shot: capture → detect → OCR → solve
# ---------------------------------------------------------------------------

def auto_detect_and_solve(
    image_bgr=None,
    words: Optional[Set[str]] = None,
    words_path: Path = DEFAULT_WORDS_PATH,
    status: StatusCallback = _noop_status,
    debug_dir: Optional[Path] = None,
    force_grid_size: Optional[int] = None,
    allow_recapture: bool = True,
    solve_time_budget_s: Optional[float] = None,
) -> Optional[dict]:
    """Capture (or reuse) a screenshot, detect board, OCR, and solve.

    Returns a dict with keys:
      - ``board``: :class:`DetectedBoard`
      - ``ocr``: :class:`OCRBoardResult`
      - ``ranked_words``: list of :class:`SolvedWord`
      - ``grid_str``: pretty-printed grid string
    or ``None`` if the board could not be detected.
    """
    captured_live = image_bgr is None
    capture_attempts = 1
    recapture_used = False

    # 1. Capture
    if image_bgr is None:
        status("Capturing device screen...")
        image_bgr = capture_adb_screenshot()
    status(f"  Screenshot: {image_bgr.shape[1]}x{image_bgr.shape[0]}")

    # 2. Detect board
    status("Detecting board...")
    board = detect_board(image_bgr, debug_dir=debug_dir, force_grid_size=force_grid_size)
    if board is None:
        status("No board detected - is the game visible?")
        return None
    status(f"  Found {board.grid_size}x{board.grid_size} board, {len(board.tiles)} tiles")

    # 3. OCR - local template matching
    status("Reading tiles via template OCR...")
    ocr_result = template_ocr_board(
        image_bgr,
        board,
        debug_dir=debug_dir,
    )
    diag = ocr_result.diagnostics or {}
    recapture_hint = bool(diag.get("recapture_recommended", False))
    if captured_live and allow_recapture and recapture_hint:
        status("  OCR ambiguity detected; recapturing once for verification...")
        image_bgr_2 = capture_adb_screenshot()
        board_2 = detect_board(image_bgr_2, debug_dir=debug_dir, force_grid_size=force_grid_size)
        if board_2 is not None:
            ocr_result_2 = template_ocr_board(
                image_bgr_2,
                board_2,
                debug_dir=debug_dir,
            )
            capture_attempts = 2
            score_1 = _ocr_quality_score(ocr_result)
            score_2 = _ocr_quality_score(ocr_result_2)
            if score_2 > score_1:
                status("  Recapture selected (higher OCR quality score).")
                image_bgr = image_bgr_2
                board = board_2
                ocr_result = ocr_result_2
                recapture_used = True
            else:
                status("  Keeping original OCR result after recapture comparison.")
        else:
            status("  Recapture board detection failed; keeping original OCR result.")

    grid = ocr_result.normalized_grid
    gs = _grid_str(grid)
    status(f"  Board:\n{gs}")
    status(
        f"  OCR sources: template={ocr_result.template_match_count}, "
        f"classifier_fallback={ocr_result.local_ocr_count}"
    )

    low = sum(1 for t in ocr_result.tiles if t.low_confidence)
    if low > 0:
        status(f"  Warning: {low} tile(s) with low confidence")

    # 4. Solve
    status("Solving...")
    active_words, resources = _get_solver_resources(words=words, words_path=words_path)
    solve_deadline: Optional[float] = None
    if solve_time_budget_s is not None and solve_time_budget_s > 0:
        solve_deadline = time.perf_counter() + solve_time_budget_s
    solved = solve_board_with_paths(grid, resources, deadline=solve_deadline)
    ranked = list(rank_solved_words(solved))
    status(f"  Found {len(ranked)} valid words")

    if ranked:
        top5 = ", ".join(w.word for w in ranked[:5])
        status(f"  Top words: {top5}")

    return {
        "board": board,
        "image_bgr": image_bgr,
        "ocr": ocr_result,
        "ranked_words": ranked,
        "grid_str": gs,
        "capture_attempts": capture_attempts,
        "recapture_used": recapture_used,
    }


# ---------------------------------------------------------------------------
# Continuous play loop (GUI-triggered)
# ---------------------------------------------------------------------------

def auto_play_loop(
    stop_event: threading.Event,
    status: StatusCallback = _noop_status,
    dry_run: bool = False,
    word_delay: float = 0.04,
    step_delay_ms: int = 5,
    max_words: Optional[int] = None,
    words: Optional[Set[str]] = None,
    words_path: Path = DEFAULT_WORDS_PATH,
    force_grid_size: Optional[int] = None,
    strategy: str = "speed",
    game_duration_s: float = 60.0,
    short_word_injection: bool = True,
    last_seconds_window_s: float = 8.0,
    speed_solve_budget_s: Optional[float] = None,
    speed_max_word_len: int = 7,
    speed_max_candidates: Optional[int] = None,
    use_path_blacklist: bool = True,
    path_blacklist_path: Optional[Path] = None,
) -> None:
    """Capture → detect → OCR → solve → play all words, stoppable.

    This is the main entry point called from the GUI's "Start" button.
    It runs on a background thread.  Set ``stop_event`` to halt early.
    """
    ensure_runtime_dirs()
    run_start = time.perf_counter()
    run_id = _utc_now_stamp()
    run_dir = Path(AUTOPLAY_RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    solve_budget = speed_solve_budget_s
    if strategy == "speed" and solve_budget is None:
        solve_budget = 2.0

    result = auto_detect_and_solve(
        words=words,
        words_path=words_path,
        status=status,
        debug_dir=run_dir,
        force_grid_size=force_grid_size,
        allow_recapture=(strategy != "speed"),
        solve_time_budget_s=solve_budget,
    )
    if result is None:
        status("Aborted — no board found.")
        return

    board: DetectedBoard = result["board"]
    ranked: List[SolvedWord] = result["ranked_words"]
    ocr_result: OCRBoardResult = result["ocr"]

    if strategy == "speed":
        ranked = _speed_candidate_filter(ranked, max_len=max(4, int(speed_max_word_len)))
        ranked = sorted(ranked, key=_speed_priority)
        ranked = _build_speed_play_order(
            ranked,
            short_word_injection=short_word_injection,
            inject_every=3,
        )
        status("  Speed mode: filtered candidates + short-word injection.")
    else:
        ranked = sorted(
            ranked,
            key=lambda word: _play_priority(word, is_exact=True),
        )
        status("  Balanced mode: prioritizing 3-letter then 4-letter words.")

    review_bundle_path: Optional[Path] = None
    if ocr_result.has_low_confidence:
        low = sum(1 for t in ocr_result.tiles if t.low_confidence)
        review_bundle_path = _save_low_confidence_review_bundle(
            run_dir=run_dir,
            image_bgr=result["image_bgr"],
            ocr_result=ocr_result,
        )
        status(
            f"  Warning: {low} low-confidence tile(s). "
            f"Saved review bundle to {review_bundle_path.parent.name}/{review_bundle_path.name} and continuing."
        )

    effective_max_words = max_words
    if strategy == "speed" and effective_max_words is None:
        effective_max_words = 120 if speed_max_candidates is None else speed_max_candidates
    if effective_max_words is not None and effective_max_words > 0:
        ranked = ranked[:effective_max_words]

    # Speed profile: trim delays for live rounds.
    effective_step_delay_ms = step_delay_ms
    effective_word_delay = word_delay
    if strategy == "speed" and not dry_run:
        effective_step_delay_ms = max(1, min(step_delay_ms, 3))
        effective_word_delay = min(word_delay, 0.005)

    total = len(ranked)
    if total == 0:
        status("No words to play.")
        return

    played = 0
    skipped = 0
    failed = 0
    attempts: List[SwipeAttempt] = []
    completed_words: List[SolvedWord] = []
    pending: List[SolvedWord] = list(ranked)
    ewma_dispatch_s = 0.18
    path_blacklist = load_path_blacklist(path=path_blacklist_path) if use_path_blacklist else {"signatures": {}, "motifs": {}}
    sig_blacklist = path_blacklist.get("signatures", {})
    motif_blacklist = path_blacklist.get("motifs", {})
    if sig_blacklist or motif_blacklist:
        status(
            f"  Path blacklist loaded: {len(sig_blacklist)} signatures, "
            f"{len(motif_blacklist)} motifs"
        )

    status(f"\n>> Playing {total} words ({'DRY RUN' if dry_run else 'LIVE'})...\n")

    i = 0
    while pending:
        if stop_event.is_set():
            status(f"\nStopped by user after {played}/{total} words.")
            break

        elapsed = time.perf_counter() - run_start
        remaining = game_duration_s - elapsed
        if strategy == "speed":
            word = _pick_next_speed_word(
                pending,
                remaining=remaining,
                last_seconds_window_s=last_seconds_window_s,
            )
        else:
            word = pending.pop(0)

        if strategy == "speed" and not dry_run:
            est = max(
                _estimate_play_time_s(word, effective_step_delay_ms, effective_word_delay),
                ewma_dispatch_s,
            )
            if remaining < 0.20:
                status(f"\nStopping for timer budget: {remaining:.2f}s left.")
                break
            # Keep short words alive longer near timer end.
            if word.length >= 5 and remaining < est * 0.95:
                i += 1
                continue
            if word.length == 4 and remaining < est * 0.75:
                i += 1
                continue
            if word.length <= 3 and remaining < est * 0.60:
                i += 1
                continue

        status(f"  [{i+1}/{total}] {word.word}  ({word.score:.0f} pts, {word.length} letters)")

        sig = path_move_signature(board, word.path)
        motifs = path_transition_motifs(board, word.path)
        blocked_info = None
        blocked_reason = ""
        if sig_blacklist and sig in sig_blacklist:
            blocked_info = sig_blacklist[sig]
            blocked_reason = "path_blacklisted_signature"
        elif motif_blacklist:
            for motif in motifs:
                if motif in motif_blacklist:
                    blocked_info = motif_blacklist[motif]
                    blocked_reason = "path_blacklisted_motif"
                    break
        if blocked_info is not None:
            attempt = SwipeAttempt(
                word=word.word,
                path=list(word.path),
                coordinates=[list(pt) for pt in path_to_device_coordinates(word.path, board)],
                duration_ms=0,
                status="skipped",
                message=(
                    f"Skipped by path blacklist ({blocked_reason}): "
                    f"(fail_rate={float(blocked_info.get('fail_rate', 0.0)):.2f})"
                ),
                route_confidence=0.0,
                predicted_touched=[],
                reject_reason=blocked_reason,
            )
            attempts.append(attempt)
            skipped += 1
            status(f"    {attempt.message}")
            i += 1
            continue

        t0 = time.perf_counter()
        attempt = playback_word_auto(
            solved_word=word,
            board=board,
            dry_run=dry_run,
            step_delay_ms=effective_step_delay_ms,
        )
        dt = max(0.01, time.perf_counter() - t0)
        ewma_dispatch_s = 0.70 * ewma_dispatch_s + 0.30 * dt
        attempts.append(attempt)

        if attempt.status in ("played", "dry_run"):
            played += 1
            completed_words.append(word)
        elif attempt.status == "skipped":
            skipped += 1
            if attempt.message:
                status(f"    {attempt.message}")
        elif attempt.status == "failed":
            failed += 1
            if attempt.message:
                status(f"    {attempt.message}")
        elif attempt.message:
            status(f"    {attempt.message}")

        if attempt.status == "played" and effective_word_delay > 0 and pending:
            time.sleep(effective_word_delay)
        i += 1

    status(f"\nDone - played {played}/{total} words (skipped={skipped}, failed={failed}).")

    # Persist run artifact
    grid_tokens = [tok for row in ocr_result.normalized_grid for tok in row]
    board_signature = _board_signature(ocr_result.normalized_grid)
    artifact_notes: List[str] = []
    if review_bundle_path is not None:
        artifact_notes.append(f"low_confidence_review_saved:{review_bundle_path.name}")
    artifact = RunArtifact(
        run_id=run_id,
        calibration_id="auto",
        created_at=datetime.now(timezone.utc).isoformat(),
        board_tokens=grid_tokens,
        solved_words=completed_words,
        swipe_attempts=attempts,
        notes=None,
    )
    failed_words_path = _write_failed_words_file(
        run_dir=run_dir,
        run_id=run_id,
        board_signature=board_signature,
        attempts=attempts,
    )
    if failed_words_path is not None:
        artifact_notes.append(f"failed_words_saved:{failed_words_path.name}")
    if artifact_notes:
        artifact = RunArtifact(
            run_id=artifact.run_id,
            calibration_id=artifact.calibration_id,
            created_at=artifact.created_at,
            board_tokens=artifact.board_tokens,
            solved_words=artifact.solved_words,
            swipe_attempts=artifact.swipe_attempts,
            notes=";".join(artifact_notes),
        )
    artifact_path = run_dir / "run.json"
    artifact_path.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    status(f"  Run saved to {run_dir.name}/")
    if failed_words_path is not None:
        status(f"  Failed attempts saved to {run_dir.name}/{failed_words_path.name}")


# ---------------------------------------------------------------------------
# Legacy one-shot session (backward compat — used by existing CLI/tests)
# ---------------------------------------------------------------------------

def run_once(
    calibration=None,
    calibration_path=None,
    words=None,
    words_path=DEFAULT_WORDS_PATH,
    fixture_path=None,
    dry_run=True,
    runs_dir=AUTOPLAY_RUNS_DIR,
    max_words=None,
    tile_reader=None,
    command_runner=None,
    deps=None,
):
    """Legacy one-shot session using calibration-based pipeline."""
    from autoplay_v2.calibration import load_calibration
    from autoplay_v2.capture import capture_roi, save_debug_capture
    from autoplay_v2.config import CALIBRATION_PATH, ensure_runtime_dirs
    from autoplay_v2.feedback import append_feedback_entry
    from autoplay_v2.input_driver import playback_word
    from autoplay_v2 import ocr as ocr_module
    from autoplay_v2.ocr import ocr_board

    ensure_runtime_dirs()
    deps = deps or {}

    cal_path = calibration_path or CALIBRATION_PATH
    current_calibration = calibration or load_calibration(cal_path)
    run_id = _utc_now_stamp()
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    capture_fn = deps.get("capture_fn", capture_roi)
    capture = capture_fn(current_calibration, fixture_path=fixture_path)
    save_debug_capture(
        frame=capture.frame,
        out_dir=run_dir,
        calibration_id=current_calibration.calibration_id,
        captured_at=capture.captured_at,
    )

    ocr_fn = deps.get("ocr_fn", ocr_board)
    effective_tile_reader = tile_reader
    if ocr_fn is ocr_board and tile_reader is None:
        for factory_name in (
            "build_runtime_tile_reader",
            "create_runtime_tile_reader",
            "get_runtime_tile_reader",
        ):
            factory = getattr(ocr_module, factory_name, None)
            if callable(factory):
                effective_tile_reader = factory()
                break
        if effective_tile_reader is None:
            direct = getattr(ocr_module, "runtime_tile_reader", None)
            if callable(direct):
                effective_tile_reader = direct

    ocr_result = ocr_fn(
        capture.frame,
        current_calibration,
        tile_reader=effective_tile_reader,
        debug_dir=run_dir,
    )

    def _flatten(grid):
        return [tok for row in grid for tok in row]

    board_tokens = _flatten(ocr_result.normalized_grid)
    if ocr_result.has_low_confidence:
        artifact = RunArtifact(
            run_id=run_id,
            calibration_id=current_calibration.calibration_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            board_tokens=board_tokens,
            solved_words=[],
            swipe_attempts=[],
            notes="ocr_low_confidence_abort",
        )
        (run_dir / "run.json").write_text(
            json.dumps(artifact.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return artifact

    active_words = words or _load_words(words_path=words_path)
    resources = build_solver_resources(active_words)
    solve_fn = deps.get("solve_fn", solve_board_with_paths)
    solved_words = solve_fn(ocr_result.normalized_grid, resources)

    rank_fn = deps.get("rank_fn", rank_solved_words)
    ranked_words = list(rank_fn(solved_words))
    if max_words is not None and max_words > 0:
        ranked_words = ranked_words[:max_words]

    playback_fn = deps.get("playback_fn", playback_word)
    feedback_fn = deps.get("feedback_fn", append_feedback_entry)
    board_sig = _board_signature(ocr_result.normalized_grid)
    attempts: list[SwipeAttempt] = []
    for solved in ranked_words:
        attempt = playback_fn(
            solved_word=solved,
            calibration=current_calibration,
            dry_run=dry_run,
            command_runner=command_runner,
        )
        attempts.append(attempt)
        try:
            fb_status = (
                "accepted" if attempt.status == "played"
                else "rejected" if attempt.status == "failed"
                else "unknown"
            )
            feedback_fn(
                word=attempt.word,
                status=fb_status,
                board_signature=board_sig,
                run_id=run_id,
            )
        except Exception:
            pass

    artifact = RunArtifact(
        run_id=run_id,
        calibration_id=current_calibration.calibration_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        board_tokens=board_tokens,
        solved_words=ranked_words,
        swipe_attempts=attempts,
    )
    (run_dir / "run.json").write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact
