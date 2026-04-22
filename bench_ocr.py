"""Benchmark: Tesseract/glyph vs NVIDIA vs Claude vision for tile OCR.

Usage:
    python bench_ocr.py                      # Tesseract only
    python bench_ocr.py --nvidia             # + NVIDIA  (needs NVIDIA_API_KEY)
    python bench_ocr.py --claude             # + Claude  (needs ANTHROPIC_API_KEY)
    python bench_ocr.py --nvidia --claude --limit 5
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

repo = Path(__file__).parent
sys.path.insert(0, str(repo))

from autoplay_v2.board_detector import detect_board
from autoplay_v2.ocr import ocr_board_auto

SCREENSHOTS_DIR = repo / "images screenshots"
GROUND_TRUTH_PATH = SCREENSHOTS_DIR / "ground_truth.json"

VISION_PROMPT = (
    "This image shows a word-game board made of circular tiles on an orange/pink background. "
    "Each tile contains 1 or 2 letters. Digraphs (2 letters on one tile) include: "
    "TH, HE, QU, IN, AN, ER, and similar pairs. "
    "Read every tile left-to-right, top-to-bottom. "
    "Return ONLY a JSON 2D array of uppercase strings — one inner array per row. "
    "Example 5x5: [[\"A\",\"TH\",\"B\",\"C\",\"D\"],[\"E\",\"F\",\"G\",\"H\",\"IN\"],...]. "
    "No explanation, no markdown, just the JSON array."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_to_b64(image_bgr: np.ndarray, quality: int = 90) -> str:
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def _parse_grid_json(text: str, expected_size: int) -> list[list[str]] | None:
    # Find outermost JSON array (handles both [[...]] and [\n  [...]\n])
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end < 1:
        return None
    try:
        grid = json.loads(text[start:end])
        if not isinstance(grid, list) or len(grid) != expected_size:
            return None
        return [[str(tok).upper()[:2].strip() for tok in row] for row in grid]
    except Exception:
        return None


def _score(detected: list[list[str]], truth: list[list[str]]) -> tuple[int, int]:
    correct = total = 0
    for r, row in enumerate(truth):
        for c, exp in enumerate(row):
            total += 1
            got = detected[r][c] if r < len(detected) and c < len(detected[r]) else ""
            correct += got == exp
    return correct, total


def _print_mismatches(detected: list[list[str]], truth: list[list[str]]) -> None:
    for r, row in enumerate(truth):
        for c, exp in enumerate(row):
            got = detected[r][c] if r < len(detected) and c < len(detected[r]) else "?"
            if got != exp:
                print(f"    ({r},{c}) expected={exp!r}  got={got!r}")


# ---------------------------------------------------------------------------
# Vision API calls (send full image, no prior board detection needed)
# ---------------------------------------------------------------------------

def _call_openai_compat(
    image_bgr: np.ndarray,
    grid_size: int,
    api_url: str,
    api_key: str,
    model: str,
) -> list[list[str]] | None:
    b64 = _img_to_b64(image_bgr)
    payload = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_tokens": 512,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"].strip()
        return _parse_grid_json(text, grid_size)
    except Exception as e:
        print(f"    API error ({model}): {e}")
        return None


def _ocr_nvidia(image_bgr: np.ndarray, grid_size: int, api_key: str) -> list[list[str]] | None:
    return _call_openai_compat(
        image_bgr, grid_size,
        api_url="https://integrate.api.nvidia.com/v1/chat/completions",
        api_key=api_key,
        model="meta/llama-3.2-11b-vision-instruct",
    )


def _ocr_claude(image_bgr: np.ndarray, grid_size: int, api_key: str) -> list[list[str]] | None:
    b64 = _img_to_b64(image_bgr)
    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 512,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
        text = data["content"][0]["text"].strip()
        return _parse_grid_json(text, grid_size)
    except Exception as e:
        print(f"    Claude API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-image benchmark
# ---------------------------------------------------------------------------

def run_one(
    path: Path,
    truth: list[list[str]],
    nvidia_key: str | None,
    claude_key: str | None,
) -> dict:
    grid_size = len(truth)
    result = {"file": path.name, "grid_size": grid_size, "detected": False}

    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        print(f"  ERROR: could not load {path.name}")
        return result

    # ---- Board detection (used only for Tesseract crop) ----
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        board = detect_board(image_bgr)
    detect_ms = (time.perf_counter() - t0) * 1000

    # ---- Method 1: Tesseract/glyph ----
    if board is not None and board.grid_size == grid_size:
        result["detected"] = True
        t0 = time.perf_counter()
        ocr = ocr_board_auto(image_bgr, board)
        tess_ms = (time.perf_counter() - t0) * 1000
        tess_grid = ocr.normalized_grid
        tess_correct, total = _score(tess_grid, truth)
        result["tesseract"] = {
            "correct": tess_correct, "total": total,
            "pct": 100 * tess_correct / total,
            "ms": detect_ms + tess_ms, "grid": tess_grid,
        }
    else:
        # Board not detected or wrong size — Tesseract gets 0
        total = grid_size * grid_size
        result["tesseract"] = {
            "correct": 0, "total": total, "pct": 0.0,
            "ms": detect_ms,
            "error": "not_detected" if board is None else f"wrong_size_{board.grid_size}x{board.grid_size}",
        }

    total = grid_size * grid_size

    # ---- Method 2: NVIDIA vision (full image) ----
    if nvidia_key:
        t0 = time.perf_counter()
        nv_grid = _ocr_nvidia(image_bgr, grid_size, nvidia_key)
        nv_ms = (time.perf_counter() - t0) * 1000
        if nv_grid:
            nv_correct, _ = _score(nv_grid, truth)
            result["nvidia"] = {
                "correct": nv_correct, "total": total,
                "pct": 100 * nv_correct / total,
                "ms": nv_ms, "grid": nv_grid,
            }
        else:
            result["nvidia"] = {"correct": 0, "total": total, "pct": 0.0, "ms": nv_ms, "error": "no_response"}

    # ---- Method 3: Claude vision (full image) ----
    if claude_key:
        t0 = time.perf_counter()
        cl_grid = _ocr_claude(image_bgr, grid_size, claude_key)
        cl_ms = (time.perf_counter() - t0) * 1000
        if cl_grid:
            cl_correct, _ = _score(cl_grid, truth)
            result["claude"] = {
                "correct": cl_correct, "total": total,
                "pct": 100 * cl_correct / total,
                "ms": cl_ms, "grid": cl_grid,
            }
        else:
            result["claude"] = {"correct": 0, "total": total, "pct": 0.0, "ms": cl_ms, "error": "no_response"}

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--claude", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    nvidia_key = claude_key = None
    if args.nvidia:
        nvidia_key = os.environ.get("NVIDIA_API_KEY")
        if not nvidia_key:
            print("ERROR: --nvidia requires NVIDIA_API_KEY env var"); sys.exit(1)
    if args.claude:
        claude_key = os.environ.get("ANTHROPIC_API_KEY")
        if not claude_key:
            print("ERROR: --claude requires ANTHROPIC_API_KEY env var"); sys.exit(1)

    with open(GROUND_TRUTH_PATH) as f:
        ground_truth: dict = json.load(f)

    images = sorted(SCREENSHOTS_DIR.glob("*.jpeg")) + sorted(SCREENSHOTS_DIR.glob("*.png"))
    images = [p for p in images if p.name in ground_truth]
    if args.limit:
        images = images[: args.limit]

    methods = ["tesseract"] + (["nvidia"] if nvidia_key else []) + (["claude"] if claude_key else [])
    print(f"Benchmarking {len(images)} images  |  methods: {', '.join(methods)}\n")

    results = []
    totals: dict[str, list[int]] = {m: [0, 0] for m in methods}
    times: dict[str, list[float]] = {m: [] for m in methods}

    for path in images:
        truth = ground_truth[path.name]
        print(f"[{path.name}]  ({len(truth)}x{len(truth[0])})")
        r = run_one(path, truth, nvidia_key, claude_key)
        results.append(r)

        for m in methods:
            d = r.get(m)
            if not d:
                continue
            totals[m][0] += d["correct"]
            totals[m][1] += d["total"]
            times[m].append(d["ms"])
            err = d.get("error", "")
            flag = "OK " if d["pct"] == 100 else ("~  " if d["pct"] >= 80 else "BAD")
            suffix = f"  [{err}]" if err else ""
            print(f"  {m:10s}: [{flag}] {d['correct']}/{d['total']} = {d['pct']:.0f}%  ({d['ms']:.0f} ms){suffix}")
            if d["pct"] < 100 and not err:
                _print_mismatches(d["grid"], truth)

    # ---- Summary ----
    print(f"\n{'='*55}")
    print(f"OVERALL  ({len(images)} boards, {sum(t[1] for t in totals.values() if t[1])//len(methods)} tiles each)\n")
    print(f"  {'Method':<12}  {'Tiles':>10}  {'Accuracy':>9}  {'Avg ms':>8}  {'Med ms':>8}")
    print(f"  {'-'*54}")
    for m in methods:
        c, t = totals[m]
        if t == 0:
            continue
        pct = 100 * c / t
        ms_list = times[m]
        avg_ms = sum(ms_list) / len(ms_list) if ms_list else 0
        med_ms = sorted(ms_list)[len(ms_list)//2] if ms_list else 0
        print(f"  {m:<12}  {c:>4}/{t:<4}  {pct:>8.1f}%  {avg_ms:>7.0f}ms  {med_ms:>7.0f}ms")

    out = repo / "bench_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {out.name}")


if __name__ == "__main__":
    main()
