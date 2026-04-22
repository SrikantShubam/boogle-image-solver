from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Callable, Optional

import cv2

from autoplay_v2.board_detector import detect_board
from autoplay_v2.config import OCR_TEMPLATE_LIBRARY_PATH, repo_root
from autoplay_v2.nvidia_ocr import nvidia_ocr_board
from autoplay_v2.ocr import (
    TemplateLibrary,
    TileKNNClassifier,
    load_template_library,
    ocr_board_auto,
    read_tile_with_consensus,
)


def _load_ground_truth(path: Path) -> dict[str, list[list[str]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score(grid: list[list[str]], truth: list[list[str]]) -> tuple[int, int]:
    total = len(truth) * len(truth[0])
    correct = 0
    for row_idx, row in enumerate(truth):
        for col_idx, expected in enumerate(row):
            got = grid[row_idx][col_idx] if row_idx < len(grid) and col_idx < len(grid[row_idx]) else ""
            correct += int(got == expected)
    return correct, total


def _template_only_grid(image_bgr, board, library: TemplateLibrary) -> list[list[str]]:
    from autoplay_v2.ocr import extract_tile_images_from_circles

    grid = [["" for _ in range(board.grid_size)] for _ in range(board.grid_size)]
    for tile, tile_image in zip(board.tiles, extract_tile_images_from_circles(image_bgr, board)):
        token, _confidence, _source = read_tile_with_consensus(
            tile_image,
            tile.index,
            tile.row,
            tile.col,
            template_library=library,
            local_reader=lambda *_args: ("", 0.0),
        )
        grid[tile.row][tile.col] = token
    return grid


def _classifier_only_grid(image_bgr, board, library: TemplateLibrary) -> list[list[str]]:
    from autoplay_v2.ocr import extract_tile_images_from_circles

    classifier = TileKNNClassifier.from_template_library(library)
    grid = [["" for _ in range(board.grid_size)] for _ in range(board.grid_size)]
    for tile, tile_image in zip(board.tiles, extract_tile_images_from_circles(image_bgr, board)):
        token, _confidence = classifier.predict_tile(tile_image)
        grid[tile.row][tile.col] = token
    return grid


def _run_method(
    fn: Callable[[], list[list[str]]],
    truth: list[list[str]],
) -> dict:
    started = time.perf_counter()
    grid = fn()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    correct, total = _score(grid, truth)
    return {
        "grid": grid,
        "correct": correct,
        "total": total,
        "accuracy_pct": round((correct / total) * 100.0, 2) if total else 0.0,
        "latency_ms": round(elapsed_ms, 2),
    }


def benchmark_dataset(
    dataset_dir: Path,
    ground_truth_path: Path,
    template_library_path: Path,
    *,
    include_nvidia: bool,
    limit: Optional[int],
) -> dict:
    truth_map = _load_ground_truth(ground_truth_path)
    library = load_template_library(str(template_library_path))
    results: list[dict] = []

    for filename, truth in sorted(truth_map.items())[: limit or None]:
        image_path = dataset_dir / filename
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            results.append({"file": filename, "error": "unreadable"})
            continue

        board = detect_board(image_bgr, force_grid_size=len(truth))
        if board is None:
            results.append({"file": filename, "error": "board_detection_failed"})
            continue

        board_result = {
            "file": filename,
            "grid_size": board.grid_size,
            "template": _run_method(lambda: _template_only_grid(image_bgr, board, library), truth),
            "classifier": _run_method(lambda: _classifier_only_grid(image_bgr, board, library), truth),
            "consensus": _run_method(
                lambda: ocr_board_auto(
                    image_bgr,
                    board,
                    template_library=library,
                ).normalized_grid,
                truth,
            ),
        }
        if include_nvidia:
            board_result["nvidia_11b"] = _run_method(
                lambda: nvidia_ocr_board(image_bgr, board).normalized_grid,
                truth,
            )
        results.append(board_result)

    summary: dict[str, dict[str, float]] = {}
    for method in ("template", "classifier", "consensus", "nvidia_11b"):
        method_rows = [row[method] for row in results if method in row]
        if not method_rows:
            continue
        summary[method] = {
            "tiles_correct": int(sum(row["correct"] for row in method_rows)),
            "tiles_total": int(sum(row["total"] for row in method_rows)),
            "accuracy_pct": round(
                (sum(row["correct"] for row in method_rows) / sum(row["total"] for row in method_rows)) * 100.0,
                2,
            ),
            "avg_latency_ms": round(mean(row["latency_ms"] for row in method_rows), 2),
            "median_latency_ms": round(median(row["latency_ms"] for row in method_rows), 2),
        }

    return {
        "dataset_dir": str(dataset_dir),
        "ground_truth_path": str(ground_truth_path),
        "template_library_path": str(template_library_path),
        "boards": results,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local OCR methods against the labeled board dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=repo_root() / "images screenshots",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=repo_root() / "images screenshots" / "ground_truth.json",
    )
    parser.add_argument(
        "--template-library",
        type=Path,
        default=OCR_TEMPLATE_LIBRARY_PATH,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-nvidia", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "runs" / "ocr_benchmark.json",
    )
    args = parser.parse_args()

    results = benchmark_dataset(
        dataset_dir=args.dataset_dir,
        ground_truth_path=args.ground_truth,
        template_library_path=args.template_library,
        include_nvidia=args.include_nvidia,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print("Method         Accuracy    Avg ms    Med ms")
    print("--------------------------------------------")
    for method, summary in results["summary"].items():
        print(
            f"{method:<13}"
            f"{summary['accuracy_pct']:>8.2f}%"
            f"{summary['avg_latency_ms']:>10.2f}"
            f"{summary['median_latency_ms']:>10.2f}"
        )
    print(f"\nSaved benchmark results to {args.output}")


if __name__ == "__main__":
    main()
