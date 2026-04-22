from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import cv2

from autoplay_v2.board_detector import detect_board
from autoplay_v2.config import OCR_TEMPLATE_LIBRARY_PATH, repo_root
from autoplay_v2.ocr import TemplateLibrary, extract_tile_images_from_circles, save_template_library


def _load_ground_truth(path: Path) -> dict[str, list[list[str]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def harvest_templates(
    dataset_dir: Path,
    ground_truth_path: Path,
    output_path: Path,
    *,
    min_score: float,
) -> tuple[Path, dict]:
    truth_map = _load_ground_truth(ground_truth_path)
    entries: List[Tuple[str, object]] = []
    token_counts: Counter[str] = Counter()
    failed_files: list[str] = []

    for filename, truth in sorted(truth_map.items()):
        image_path = dataset_dir / filename
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            failed_files.append(f"{filename}:unreadable")
            continue

        board = detect_board(image_bgr, force_grid_size=len(truth))
        if board is None or board.grid_size != len(truth):
            failed_files.append(f"{filename}:board_detection_failed")
            continue

        tile_images = extract_tile_images_from_circles(image_bgr, board)
        for tile, tile_image in zip(board.tiles, tile_images):
            token = truth[tile.row][tile.col]
            entries.append((token, tile_image))
            token_counts[token] += 1

    library = TemplateLibrary.from_tile_images(entries, min_score=min_score)
    saved_path = save_template_library(library, output_path)
    summary = {
        "dataset_dir": str(dataset_dir),
        "ground_truth_path": str(ground_truth_path),
        "output_path": str(saved_path),
        "template_count": int(library.labels.size),
        "unique_tokens": len(set(library.labels.tolist())),
        "token_counts": dict(sorted(token_counts.items())),
        "failed_files": failed_files,
    }
    summary_path = saved_path.with_name("template_library.summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return saved_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest OCR templates from the labeled board dataset.")
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
        "--output",
        type=Path,
        default=OCR_TEMPLATE_LIBRARY_PATH,
    )
    parser.add_argument("--min-score", type=float, default=0.91)
    args = parser.parse_args()

    output_path, summary = harvest_templates(
        dataset_dir=args.dataset_dir,
        ground_truth_path=args.ground_truth,
        output_path=args.output,
        min_score=args.min_score,
    )
    print(json.dumps({"output_path": str(output_path), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
