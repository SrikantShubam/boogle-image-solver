"""Build a path-signature blacklist from audit per_word.csv output."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _latest_audit_dir() -> Path:
    candidates = sorted(
        [p for p in (REPO / "runs").glob("word_path_audit_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError("No word_path_audit_* directories found in runs/")
    return candidates[-1]


def build_blacklist(
    csv_path: Path,
    mode: str,
    min_attempts: int,
    min_fail_rate: float,
    min_extra_share: float,
    max_patterns: int,
) -> List[Dict[str, object]]:
    agg: Dict[str, Dict[str, object]] = defaultdict(lambda: {
        "attempts": 0,
        "mismatches": 0,
        "buckets": Counter(),
        "examples": [],
    })

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_mode = str(row.get("mode", ""))
            if mode != "both" and row_mode != mode:
                continue
            sig = str(row.get("path_signature", "")).strip()
            if not sig:
                continue
            bucket = str(row.get("bucket", "unknown"))
            matched = str(row.get("matched", "")).strip().lower() == "true"
            intended = str(row.get("intended_word", ""))
            typed = str(row.get("typed_word_predicted", ""))
            image = str(row.get("image", ""))

            item = agg[sig]
            item["attempts"] = int(item["attempts"]) + 1
            item["buckets"][bucket] += 1  # type: ignore[index]
            if not matched:
                item["mismatches"] = int(item["mismatches"]) + 1
                if len(item["examples"]) < 6:  # type: ignore[index]
                    item["examples"].append({  # type: ignore[index]
                        "image": image,
                        "intended_word": intended,
                        "typed_word_predicted": typed,
                        "bucket": bucket,
                    })

    out: List[Dict[str, object]] = []
    for sig, item in agg.items():
        attempts = int(item["attempts"])
        mismatches = int(item["mismatches"])
        if attempts < max(1, min_attempts):
            continue
        fail_rate = mismatches / attempts
        if fail_rate < min_fail_rate:
            continue

        buckets: Counter = item["buckets"]  # type: ignore[assignment]
        extra_count = int(buckets.get("extra_tile_insertion", 0))
        extra_share = extra_count / max(1, mismatches)
        if extra_share < min_extra_share:
            continue

        out.append(
            {
                "signature": sig,
                "attempts": attempts,
                "mismatches": mismatches,
                "fail_rate": round(fail_rate, 6),
                "extra_insertion_share": round(extra_share, 6),
                "bucket_counts": dict(buckets),
                "examples": list(item["examples"]),  # type: ignore[arg-type]
            }
        )

    out.sort(
        key=lambda p: (
            float(p["fail_rate"]),
            int(p["mismatches"]),
            int(p["attempts"]),
        ),
        reverse=True,
    )
    if max_patterns > 0:
        out = out[:max_patterns]
    return out


def _extract_motifs_from_signature(sig: str) -> List[str]:
    marker = "|M:"
    if marker not in sig:
        return []
    tail = sig.split(marker, 1)[1].strip()
    if not tail:
        return []
    moves = [m for m in tail.split("|") if m]
    if len(moves) < 2:
        return []
    return [f"{a}>{b}" for a, b in zip(moves, moves[1:])]


def build_motif_blacklist(
    csv_path: Path,
    mode: str,
    min_attempts: int,
    min_fail_rate: float,
    min_extra_share: float,
    max_patterns: int,
) -> List[Dict[str, object]]:
    agg: Dict[str, Dict[str, object]] = defaultdict(lambda: {
        "attempts": 0,
        "mismatches": 0,
        "extra_insertions": 0,
        "examples": [],
    })
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_mode = str(row.get("mode", ""))
            if mode != "both" and row_mode != mode:
                continue
            sig = str(row.get("path_signature", "")).strip()
            motifs = _extract_motifs_from_signature(sig)
            if not motifs:
                continue
            matched = str(row.get("matched", "")).strip().lower() == "true"
            bucket = str(row.get("bucket", "unknown"))
            example = {
                "image": str(row.get("image", "")),
                "intended_word": str(row.get("intended_word", "")),
                "typed_word_predicted": str(row.get("typed_word_predicted", "")),
                "bucket": bucket,
            }
            for motif in motifs:
                item = agg[motif]
                item["attempts"] = int(item["attempts"]) + 1
                if not matched:
                    item["mismatches"] = int(item["mismatches"]) + 1
                    if bucket == "extra_tile_insertion":
                        item["extra_insertions"] = int(item["extra_insertions"]) + 1
                    if len(item["examples"]) < 8:  # type: ignore[index]
                        item["examples"].append(example)  # type: ignore[index]

    out: List[Dict[str, object]] = []
    for motif, item in agg.items():
        attempts = int(item["attempts"])
        mismatches = int(item["mismatches"])
        if attempts < max(1, min_attempts):
            continue
        if mismatches == 0:
            continue
        fail_rate = mismatches / attempts
        if fail_rate < min_fail_rate:
            continue
        extra_share = int(item["extra_insertions"]) / mismatches
        if extra_share < min_extra_share:
            continue
        out.append(
            {
                "motif": motif,
                "attempts": attempts,
                "mismatches": mismatches,
                "fail_rate": round(fail_rate, 6),
                "extra_insertion_share": round(extra_share, 6),
                "examples": list(item["examples"]),  # type: ignore[arg-type]
            }
        )
    out.sort(
        key=lambda p: (
            float(p["fail_rate"]),
            int(p["mismatches"]),
            int(p["attempts"]),
        ),
        reverse=True,
    )
    if max_patterns > 0:
        out = out[:max_patterns]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build path blacklist from per_word.csv audit output.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Path to per_word.csv (defaults to latest runs/word_path_audit_*/per_word.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=["ocr", "ground_truth", "both"],
        default="ground_truth",
        help="Which audit mode rows to learn from",
    )
    parser.add_argument("--min-attempts", type=int, default=3)
    parser.add_argument("--min-fail-rate", type=float, default=0.80)
    parser.add_argument("--min-extra-share", type=float, default=0.50)
    parser.add_argument("--motif-min-attempts", type=int, default=25)
    parser.add_argument("--motif-min-fail-rate", type=float, default=0.26)
    parser.add_argument("--motif-min-extra-share", type=float, default=0.45)
    parser.add_argument("--max-patterns", type=int, default=150)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "data" / "path_blacklist.json",
    )
    args = parser.parse_args()

    input_csv = args.input_csv
    if input_csv is None:
        input_csv = _latest_audit_dir() / "per_word.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    patterns = build_blacklist(
        csv_path=input_csv,
        mode=args.mode,
        min_attempts=args.min_attempts,
        min_fail_rate=args.min_fail_rate,
        min_extra_share=args.min_extra_share,
        max_patterns=args.max_patterns,
    )
    motifs = build_motif_blacklist(
        csv_path=input_csv,
        mode=args.mode,
        min_attempts=args.motif_min_attempts,
        min_fail_rate=args.motif_min_fail_rate,
        min_extra_share=args.motif_min_extra_share,
        max_patterns=args.max_patterns,
    )
    payload = {
        "version": 1,
        "generated_at": _utc_stamp(),
        "source_csv": str(input_csv),
        "mode": args.mode,
        "criteria": {
            "min_attempts": args.min_attempts,
            "min_fail_rate": args.min_fail_rate,
            "min_extra_share": args.min_extra_share,
            "max_patterns": args.max_patterns,
            "motif_min_attempts": args.motif_min_attempts,
            "motif_min_fail_rate": args.motif_min_fail_rate,
            "motif_min_extra_share": args.motif_min_extra_share,
        },
        "patterns": patterns,
        "motifs": motifs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(patterns)} signatures and {len(motifs)} motifs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
