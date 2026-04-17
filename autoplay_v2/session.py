from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Set

from autoplay_v2.calibration import load_calibration
from autoplay_v2.capture import capture_roi, save_debug_capture
from autoplay_v2.config import AUTOPLAY_RUNS_DIR, CALIBRATION_PATH, ensure_runtime_dirs, repo_root
from autoplay_v2.feedback import append_feedback_entry
from autoplay_v2.input_driver import playback_word
from autoplay_v2.models import CalibrationConfig, RunArtifact, SolvedWord, SwipeAttempt
from autoplay_v2 import ocr as ocr_module
from autoplay_v2.ocr import ocr_board
from autoplay_v2.ranking import rank_solved_words
from autoplay_v2.solver import build_solver_resources, solve_board_with_paths

DEFAULT_WORDS_PATH = repo_root() / "words.txt"


def _utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _board_signature(grid: Sequence[Sequence[str]]) -> str:
    return "|".join(",".join(token for token in row) for row in grid)


def _flatten_grid(grid: Sequence[Sequence[str]]) -> list[str]:
    return [token for row in grid for token in row]


def _load_words(words_path: Path = DEFAULT_WORDS_PATH, min_len: int = 3) -> Set[str]:
    words: Set[str] = set()
    with Path(words_path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            word = raw.strip().upper()
            if len(word) >= min_len and word.isalpha():
                words.add(word)
    return words


def _status_for_feedback(attempt: SwipeAttempt) -> str:
    if attempt.status == "played":
        return "accepted"
    if attempt.status == "failed":
        return "rejected"
    return "unknown"


def _resolve_runtime_tile_reader(
    tile_reader: Optional[Callable[..., tuple[str, float]]],
    deps: Dict[str, Any],
) -> Callable[..., tuple[str, float]]:
    if tile_reader is not None:
        return tile_reader

    injected_reader = deps.get("tile_reader")
    if callable(injected_reader):
        return injected_reader

    injected_factory = deps.get("tile_reader_factory")
    if callable(injected_factory):
        resolved_reader = injected_factory()
        if callable(resolved_reader):
            return resolved_reader
        raise RuntimeError("OCR tile_reader_factory did not return a callable reader.")

    for factory_name in (
        "build_runtime_tile_reader",
        "create_runtime_tile_reader",
        "get_runtime_tile_reader",
    ):
        factory = getattr(ocr_module, factory_name, None)
        if not callable(factory):
            continue
        resolved_reader = factory()
        if callable(resolved_reader):
            return resolved_reader

    direct_reader = getattr(ocr_module, "runtime_tile_reader", None)
    if callable(direct_reader):
        return direct_reader

    raise RuntimeError(
        "No runtime OCR reader is configured. Provide tile_reader, tile_reader_factory, "
        "or expose build_runtime_tile_reader/create_runtime_tile_reader/get_runtime_tile_reader "
        "from autoplay_v2.ocr."
    )


def run_once(
    calibration: Optional[CalibrationConfig] = None,
    calibration_path: Path = CALIBRATION_PATH,
    words: Optional[Set[str]] = None,
    words_path: Path = DEFAULT_WORDS_PATH,
    fixture_path: Optional[Path] = None,
    dry_run: bool = True,
    runs_dir: Path = AUTOPLAY_RUNS_DIR,
    max_words: Optional[int] = None,
    tile_reader: Optional[Callable[..., tuple[str, float]]] = None,
    command_runner: Optional[Callable[[list[str]], int]] = None,
    deps: Optional[Dict[str, Any]] = None,
) -> RunArtifact:
    ensure_runtime_dirs()
    deps = deps or {}

    current_calibration = calibration or load_calibration(calibration_path)
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
    if ocr_fn is ocr_board:
        effective_tile_reader = _resolve_runtime_tile_reader(
            tile_reader=tile_reader,
            deps=deps,
        )
    ocr_result = ocr_fn(
        capture.frame,
        current_calibration,
        tile_reader=effective_tile_reader,
        debug_dir=run_dir,
    )

    board_tokens = _flatten_grid(ocr_result.normalized_grid)
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
    solved_words: list[SolvedWord] = solve_fn(ocr_result.normalized_grid, resources)

    rank_fn = deps.get("rank_fn", rank_solved_words)
    ranked_words: list[SolvedWord] = list(rank_fn(solved_words))
    if max_words is not None and max_words > 0:
        ranked_words = ranked_words[:max_words]

    playback_fn = deps.get("playback_fn", playback_word)
    feedback_fn = deps.get("feedback_fn", append_feedback_entry)
    board_signature = _board_signature(ocr_result.normalized_grid)
    attempts: list[SwipeAttempt] = []
    for solved in ranked_words:
        attempt: SwipeAttempt = playback_fn(
            solved_word=solved,
            calibration=current_calibration,
            dry_run=dry_run,
            command_runner=command_runner,
        )
        attempts.append(attempt)
        feedback_fn(
            word=attempt.word,
            status=_status_for_feedback(attempt),
            board_signature=board_signature,
            run_id=run_id,
        )

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
