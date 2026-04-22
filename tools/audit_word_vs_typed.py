"""Audit intended solver words vs predicted typed words on image datasets.

Outputs:
- per_word.csv: row-level details for each analyzed word
- top_mismatches.csv: highest-severity divergences
- summary.json: aggregated metrics

Default dataset:
  new images/
  new images/ground_truth.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autoplay_v2.board_detector import detect_board
from autoplay_v2.input_driver import build_best_swipe_route, playback_word_auto
from autoplay_v2.models import DetectedBoard, SolvedWord
from autoplay_v2.path_filters import path_move_signature
from autoplay_v2.ranking import rank_solved_words
from autoplay_v2.session import _get_solver_resources
from autoplay_v2.solver import solve_board_with_paths
from autoplay_v2.template_ocr import template_ocr_board


@dataclass(frozen=True)
class WordRow:
    mode: str
    image: str
    intended_word: str
    typed_word_predicted: str
    matched: bool
    severity: int
    bucket: str
    strategy: str
    is_exact_path: bool
    route_confidence: Optional[float]
    reject_reason: str
    status: str
    path_len: int
    touched_len: int
    extra_tiles: int
    missing_tiles: int
    order_mismatch: int
    start_end_mismatch: bool
    score: float
    length: int
    intended_path: str
    path_signature: str


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _load_ground_truth(path: Path) -> Dict[str, List[List[str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[List[str]]] = {}
    for fname, grid in raw.items():
        out[str(fname)] = [[str(tok).upper() for tok in row] for row in grid]
    return out


def _iter_mode_grids(
    mode: str,
    image_bgr,
    board: DetectedBoard,
    truth_grid: List[List[str]],
) -> Iterable[Tuple[str, List[List[str]]]]:
    if mode in ("ocr", "both"):
        ocr_res = template_ocr_board(image_bgr, board)
        yield ("ocr", [[str(tok).upper() for tok in row] for row in ocr_res.normalized_grid])
    if mode in ("ground_truth", "both"):
        yield ("ground_truth", truth_grid)


def _apply_ordering(words: List[SolvedWord], strategy_profile: str) -> List[SolvedWord]:
    if strategy_profile == "speed":
        return sorted(
            words,
            key=lambda w: (-w.length, -w.score, w.word, tuple(w.path)),
        )
    # balanced: prefer short words first.
    def _balanced_key(w: SolvedWord) -> Tuple[int, int, float, str, Tuple[int, ...]]:
        if w.length <= 3:
            group = 0
        elif w.length == 4:
            group = 1
        else:
            group = 2
        return (group, w.length, -w.score, w.word, tuple(w.path))

    return sorted(words, key=_balanced_key)


def _tokens_from_indices(board: DetectedBoard, grid: List[List[str]], path: Sequence[int]) -> List[str]:
    tokens: List[str] = []
    for idx in path:
        tile = board.tile_by_index.get(int(idx))
        if tile is None:
            continue
        tokens.append(grid[tile.row][tile.col])
    return tokens


def _classify_mismatch(path: Sequence[int], touched: Sequence[int]) -> Tuple[str, int, int, int, bool, int]:
    target = [int(v) for v in path]
    actual = [int(v) for v in touched]

    extras = [idx for idx in actual if idx not in target]
    missing = [idx for idx in target if idx not in actual]
    order_mismatch = sum(1 for a, b in zip(actual, target) if a != b)
    start_end_mismatch = bool(actual) and (actual[0] != target[0] or actual[-1] != target[-1])

    if actual == target:
        return ("exact_match", 0, len(extras), len(missing), start_end_mismatch, order_mismatch)

    issues = 0
    if start_end_mismatch:
        issues += 1
    if extras:
        issues += 1
    if missing:
        issues += 1
    if order_mismatch > 0:
        issues += 1

    if start_end_mismatch:
        bucket = "start_end_mismatch"
        severity = 100 + len(extras) * 10 + len(missing) * 10 + order_mismatch
    elif extras and missing:
        bucket = "multi_error"
        severity = 80 + len(extras) * 8 + len(missing) * 8 + order_mismatch
    elif extras:
        bucket = "extra_tile_insertion"
        severity = 70 + len(extras) * 8 + order_mismatch
    elif missing:
        bucket = "missing_tile"
        severity = 60 + len(missing) * 8 + order_mismatch
    elif order_mismatch:
        bucket = "order_mismatch"
        severity = 40 + order_mismatch
    else:
        bucket = "multi_error" if issues else "exact_match"
        severity = 30
    return (bucket, severity, len(extras), len(missing), start_end_mismatch, order_mismatch)


def _write_csv(path: Path, rows: List[WordRow]) -> None:
    fields = [
        "mode",
        "image",
        "intended_word",
        "typed_word_predicted",
        "matched",
        "severity",
        "bucket",
        "strategy",
        "is_exact_path",
        "route_confidence",
        "reject_reason",
        "status",
        "path_len",
        "touched_len",
        "extra_tiles",
        "missing_tiles",
        "order_mismatch",
        "start_end_mismatch",
        "score",
        "length",
        "intended_path",
        "path_signature",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mode": row.mode,
                    "image": row.image,
                    "intended_word": row.intended_word,
                    "typed_word_predicted": row.typed_word_predicted,
                    "matched": row.matched,
                    "severity": row.severity,
                    "bucket": row.bucket,
                    "strategy": row.strategy,
                    "is_exact_path": row.is_exact_path,
                    "route_confidence": row.route_confidence,
                    "reject_reason": row.reject_reason,
                    "status": row.status,
                    "path_len": row.path_len,
                    "touched_len": row.touched_len,
                    "extra_tiles": row.extra_tiles,
                    "missing_tiles": row.missing_tiles,
                    "order_mismatch": row.order_mismatch,
                    "start_end_mismatch": row.start_end_mismatch,
                    "score": row.score,
                    "length": row.length,
                    "intended_path": row.intended_path,
                    "path_signature": row.path_signature,
                }
            )


def audit_dataset(
    dataset_dir: Path,
    ground_truth_path: Path,
    mode: str,
    strategy_profile: str,
    max_words_per_image: Optional[int],
    words_path: Path,
) -> Tuple[List[WordRow], Dict[str, object]]:
    truth_map = _load_ground_truth(ground_truth_path)
    _, resources = _get_solver_resources(words=None, words_path=words_path)

    rows: List[WordRow] = []
    summary: Dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "ground_truth_path": str(ground_truth_path),
        "mode": mode,
        "strategy_profile": strategy_profile,
        "max_words_per_image": max_words_per_image,
        "images_total": len(truth_map),
    }

    mode_counts: Dict[str, Counter[str]] = {
        "ocr": Counter(),
        "ground_truth": Counter(),
    }
    bucket_counts: Dict[str, Counter[str]] = {
        "ocr": Counter(),
        "ground_truth": Counter(),
    }
    reject_reason_counts: Dict[str, Counter[str]] = {
        "ocr": Counter(),
        "ground_truth": Counter(),
    }
    image_fail_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    confusion_counts: Dict[str, Counter[str]] = {
        "ocr": Counter(),
        "ground_truth": Counter(),
    }

    processed_images = 0
    skipped_images = 0

    for fname in sorted(truth_map.keys()):
        image_path = dataset_dir / fname
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            skipped_images += 1
            continue
        truth_grid = truth_map[fname]
        n = len(truth_grid)
        board = detect_board(image_bgr, force_grid_size=n)
        if board is None:
            skipped_images += 1
            continue
        processed_images += 1

        for mode_name, grid in _iter_mode_grids(mode, image_bgr, board, truth_grid):
            solved = solve_board_with_paths(grid, resources)
            ranked = _apply_ordering(list(rank_solved_words(solved)), strategy_profile=strategy_profile)
            if max_words_per_image is not None and max_words_per_image > 0:
                ranked = ranked[:max_words_per_image]

            for solved_word in ranked:
                _, touched, strategy, is_exact = build_best_swipe_route(solved_word.path, board)
                attempt = playback_word_auto(
                    solved_word=solved_word,
                    board=board,
                    dry_run=True,
                    step_delay_ms=3,
                )
                pred_touched = attempt.predicted_touched or [int(v) for v in touched]
                typed_tokens = _tokens_from_indices(board, grid, pred_touched)
                typed_word = "".join(typed_tokens)
                bucket, severity, extra_tiles, missing_tiles, start_end_mismatch, order_mismatch = _classify_mismatch(
                    solved_word.path,
                    pred_touched,
                )
                matched = bucket == "exact_match"

                row = WordRow(
                    mode=mode_name,
                    image=fname,
                    intended_word=solved_word.word,
                    typed_word_predicted=typed_word,
                    matched=matched,
                    severity=severity,
                    bucket=bucket,
                    strategy=strategy,
                    is_exact_path=is_exact,
                    route_confidence=attempt.route_confidence,
                    reject_reason=attempt.reject_reason,
                    status=attempt.status,
                    path_len=len(solved_word.path),
                    touched_len=len(pred_touched),
                    extra_tiles=extra_tiles,
                    missing_tiles=missing_tiles,
                    order_mismatch=order_mismatch,
                    start_end_mismatch=start_end_mismatch,
                    score=solved_word.score,
                    length=solved_word.length,
                    intended_path="|".join(str(v) for v in solved_word.path),
                    path_signature=path_move_signature(board, solved_word.path),
                )
                rows.append(row)

                mode_counts[mode_name]["total"] += 1
                if matched:
                    mode_counts[mode_name]["matched"] += 1
                else:
                    mode_counts[mode_name]["mismatched"] += 1
                    image_fail_counts[mode_name][fname] += 1
                    confusion_counts[mode_name][f"{solved_word.word}->{typed_word}"] += 1
                bucket_counts[mode_name][bucket] += 1
                if attempt.reject_reason:
                    reject_reason_counts[mode_name][attempt.reject_reason] += 1

    summary["processed_images"] = processed_images
    summary["skipped_images"] = skipped_images

    mode_summary: Dict[str, object] = {}
    for mode_name in ("ocr", "ground_truth"):
        total = mode_counts[mode_name]["total"]
        matched = mode_counts[mode_name]["matched"]
        mismatched = mode_counts[mode_name]["mismatched"]
        mode_summary[mode_name] = {
            "total_words": total,
            "matched_words": matched,
            "mismatched_words": mismatched,
            "exact_match_rate_pct": (100.0 * matched / total) if total else 0.0,
            "bucket_counts": dict(bucket_counts[mode_name]),
            "top_reject_reasons": reject_reason_counts[mode_name].most_common(10),
            "top_failing_images": image_fail_counts[mode_name].most_common(10),
            "top_confusions": confusion_counts[mode_name].most_common(15),
        }
    summary["modes"] = mode_summary
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit intended solver words vs predicted typed words.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO / "new images",
        help="Directory containing screenshot images",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=REPO / "new images" / "ground_truth.json",
        help="Path to ground-truth grid JSON",
    )
    parser.add_argument(
        "--words-path",
        type=Path,
        default=REPO / "words.txt",
        help="Dictionary words file",
    )
    parser.add_argument(
        "--mode",
        choices=["ocr", "ground_truth", "both"],
        default="both",
        help="Grid source(s) to analyze",
    )
    parser.add_argument(
        "--strategy-profile",
        choices=["speed", "balanced"],
        default="speed",
        help="Word ordering profile used for analysis ordering",
    )
    parser.add_argument(
        "--max-words-per-image",
        type=int,
        default=None,
        help="Limit analyzed words per image (after ranking/order)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory; default runs/word_path_audit_<timestamp>",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (REPO / "runs" / f"word_path_audit_{_utc_stamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = audit_dataset(
        dataset_dir=args.dataset_dir,
        ground_truth_path=args.ground_truth,
        mode=args.mode,
        strategy_profile=args.strategy_profile,
        max_words_per_image=args.max_words_per_image,
        words_path=args.words_path,
    )

    per_word_csv = output_dir / "per_word.csv"
    _write_csv(per_word_csv, rows)

    top_rows = sorted(
        [row for row in rows if not row.matched],
        key=lambda row: (row.severity, row.length, row.score),
        reverse=True,
    )[:100]
    top_csv = output_dir / "top_mismatches.csv"
    _write_csv(top_csv, top_rows)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Output directory: {output_dir}")
    for mode_name, data in summary["modes"].items():
        total = int(data["total_words"])
        matched = int(data["matched_words"])
        rate = float(data["exact_match_rate_pct"])
        print(f"{mode_name}: matched={matched}/{total} ({rate:.2f}%)")
    print(f"Per-word CSV: {per_word_csv.name}")
    print(f"Top mismatches CSV: {top_csv.name}")
    print(f"Summary JSON: {summary_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
