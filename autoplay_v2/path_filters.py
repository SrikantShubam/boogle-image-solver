from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from autoplay_v2.config import repo_root
from autoplay_v2.models import DetectedBoard


DEFAULT_PATH_BLACKLIST = repo_root() / "data" / "path_blacklist.json"


def path_moves(board: DetectedBoard, path: Sequence[int]) -> List[str]:
    points: List[Tuple[int, int]] = []
    for idx in path:
        tile = board.tile_by_index.get(int(idx))
        if tile is None:
            continue
        points.append((tile.row, tile.col))
    if len(points) < 2:
        return []
    moves: List[str] = []
    for (r1, c1), (r2, c2) in zip(points, points[1:]):
        dr = r2 - r1
        dc = c2 - c1
        moves.append(f"{dr:+d}{dc:+d}")
    return moves


def path_move_signature(board: DetectedBoard, path: Sequence[int]) -> str:
    moves = path_moves(board, path)
    return f"L{len(path)}|M:{'|'.join(moves)}"


def path_transition_motifs(board: DetectedBoard, path: Sequence[int]) -> List[str]:
    moves = path_moves(board, path)
    if len(moves) < 2:
        return []
    return [f"{a}>{b}" for a, b in zip(moves, moves[1:])]


def load_path_blacklist(path: Optional[Path] = None) -> Dict[str, Dict[str, Dict[str, object]]]:
    target = path or DEFAULT_PATH_BLACKLIST
    if not Path(target).exists():
        return {"signatures": {}, "motifs": {}}
    payload = json.loads(Path(target).read_text(encoding="utf-8"))

    raw_patterns = payload.get("patterns", [])
    raw_motifs = payload.get("motifs", [])
    sig_out: Dict[str, Dict[str, object]] = {}
    motif_out: Dict[str, Dict[str, object]] = {}
    if isinstance(raw_patterns, list):
        for item in raw_patterns:
            if not isinstance(item, dict):
                continue
            sig = str(item.get("signature", "")).strip()
            if not sig:
                continue
            sig_out[sig] = dict(item)
    if isinstance(raw_motifs, list):
        for item in raw_motifs:
            if not isinstance(item, dict):
                continue
            motif = str(item.get("motif", "")).strip()
            if not motif:
                continue
            motif_out[motif] = dict(item)
    return {"signatures": sig_out, "motifs": motif_out}
