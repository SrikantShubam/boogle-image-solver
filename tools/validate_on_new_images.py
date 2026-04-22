"""Validate template OCR on live-capture screenshots in new images/."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autoplay_v2.board_detector import detect_board
from autoplay_v2.template_ocr import template_ocr_board

DIR = REPO / "new images"
GT = json.loads((DIR / "ground_truth.json").read_text())


def main() -> int:
    total = correct = 0
    boards_total = boards_exact = 0
    failure_buckets: Counter[str] = Counter()
    geometry_modes: Counter[str] = Counter()
    second_pass_boards = 0
    second_pass_tiles = 0
    recapture_recommended = 0
    miss_by_char: Counter[str] = Counter()
    total_by_char: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()

    for fname, truth in GT.items():
        img = cv2.imread(str(DIR / fname))
        if img is None:
            print(f"[skip] {fname}: missing")
            continue
        n = len(truth)
        board = detect_board(img, force_grid_size=n)
        if board is None:
            print(f"[skip] {fname}: no board")
            continue
        res = template_ocr_board(img, board)
        got = res.normalized_grid

        boards_total += 1
        geometry_modes[res.selected_geometry_mode] += 1
        if isinstance(res.diagnostics, dict):
            failure_buckets[str(res.diagnostics.get("failure_bucket", "unknown"))] += 1
            if bool(res.diagnostics.get("second_pass_used", False)):
                second_pass_boards += 1
            second_pass_tiles += int(res.diagnostics.get("second_pass_tile_count", 0))
            if bool(res.diagnostics.get("recapture_recommended", False)):
                recapture_recommended += 1

        ok = 0
        mism = []
        for r in range(n):
            for c in range(n):
                t = truth[r][c].upper()
                g = got[r][c].upper()
                total_by_char[t] += 1
                total += 1
                if g == t:
                    correct += 1
                    ok += 1
                else:
                    miss_by_char[t] += 1
                    confusions[(t, g)] += 1
                    mism.append((r, c, t, g))
        if ok == n * n:
            boards_exact += 1

        bucket = "unknown"
        score = None
        if isinstance(res.diagnostics, dict):
            bucket = str(res.diagnostics.get("failure_bucket", "unknown"))
            score = res.diagnostics.get("candidate_score")
        extra = f"mode={res.selected_geometry_mode} bucket={bucket}"
        if score is not None:
            extra += f" score={score}"
        print(f"\n=== {fname}  {ok}/{n*n}  {100.0*ok/(n*n):.1f}%  {extra} ===")
        for row in got:
            print("  " + " ".join(f"{t:>3s}" for t in row))
        if mism:
            print("  mismatches (r,c: truth -> got):")
            for r, c, t, g in mism:
                print(f"    [{r},{c}] {t:>3s} -> {g}")

    print(f"\n=== Overall: {correct}/{total}  {100.0*correct/max(1,total):.2f}% ===")
    print(f"=== Board Exact: {boards_exact}/{boards_total}  {100.0*boards_exact/max(1,boards_total):.2f}% ===")
    print("=== Geometry Mode Usage ===")
    for mode, cnt in geometry_modes.items():
        print(f"  {mode}: {cnt}")
    print("=== Failure Buckets ===")
    for bucket, cnt in failure_buckets.items():
        print(f"  {bucket}: {cnt}")
    print(f"=== Second Pass Used (boards): {second_pass_boards}/{boards_total} ===")
    print(f"=== Second Pass Used (tiles): {second_pass_tiles} ===")
    print(f"=== Recapture Recommended (boards): {recapture_recommended}/{boards_total} ===")
    print("=== Top Character Misses ===")
    for ch, misses in miss_by_char.most_common(15):
        tot = total_by_char[ch]
        print(f"  {ch}: {misses}/{tot} ({100.0*misses/max(1,tot):.2f}%)")
    print("=== Top Confusions ===")
    for (t, g), cnt in confusions.most_common(15):
        print(f"  {t}->{g}: {cnt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
