"""Input driver: continuous swipe playback via ADB.

Supports two swipe methods:
1. **motionevent** (default) — uses ``adb shell input motionevent`` which is
   available on Android 10+ and works in screen-pixel coordinates.
2. **swipechain** — chains ``input swipe`` commands per segment.  This lifts
   the finger between segments so some games may reject it.
"""
from __future__ import annotations

import math
import re
import subprocess
import time
import shutil
import sys
import os
from functools import lru_cache

_ADB_CMD = "adb"
if not shutil.which("adb"):
    _fallback = os.path.join(os.path.dirname(sys.executable), "adb.exe")
    if os.path.exists(_fallback):
        _ADB_CMD = _fallback
from typing import Callable, List, Optional, Sequence, Tuple

from autoplay_v2.models import DetectedBoard, DetectedTile, SolvedWord, SwipeAttempt

Command = List[str]
CommandRunner = Callable[[Command], int]


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def path_to_device_coordinates(
    path: Sequence[int],
    board: DetectedBoard,
) -> List[Tuple[int, int]]:
    """Map a tile-index path to raw tile-centre ``(x, y)`` coordinates."""
    tile_map = board.tile_by_index
    coords: List[Tuple[int, int]] = []
    for idx in path:
        tile = tile_map.get(int(idx))
        if tile is None:
            raise ValueError(f"Tile index {idx} not found in detected board")
        coords.append((tile.cx, tile.cy))
    return coords


def _sample_polyline_points(
    coords: Sequence[Tuple[int, int]],
    samples_per_px: float = 0.08,
) -> List[Tuple[float, float]]:
    if len(coords) < 2:
        return [(float(x), float(y)) for x, y in coords]

    points: List[Tuple[float, float]] = []
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        dist = math.dist((x1, y1), (x2, y2))
        samples = max(8, int(dist * samples_per_px))
        for step in range(samples + 1):
            t = step / float(samples)
            points.append((
                x1 + (x2 - x1) * t,
                y1 + (y2 - y1) * t,
            ))
    return points


def _touched_tile_sequence(
    points: Sequence[Tuple[float, float]],
    tiles: Sequence[DetectedTile],
    radius_scale: float = 0.72,
) -> List[int]:
    seq: List[int] = []
    last_idx: Optional[int] = None
    for x, y in points:
        hits: List[Tuple[float, int]] = []
        for tile in tiles:
            dist = math.dist((x, y), (tile.cx, tile.cy))
            if dist <= tile.radius * radius_scale:
                hits.append((dist, tile.index))
        if not hits:
            continue
        hits.sort()
        idx = hits[0][1]
        if idx != last_idx:
            seq.append(idx)
            last_idx = idx
    return seq


def _path_length(coords: Sequence[Tuple[int, int]]) -> float:
    return sum(math.dist(start, end) for start, end in zip(coords, coords[1:]))


def _dedupe_points(coords: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    deduped: List[Tuple[int, int]] = []
    for point in coords:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    return deduped


def _segment_candidate_paths(
    start: Tuple[int, int],
    end: Tuple[int, int],
    clearance: float,
) -> List[List[Tuple[int, int]]]:
    ax, ay = start
    bx, by = end
    dist = math.dist(start, end)
    if dist == 0:
        return [[start, end]]

    dx = (bx - ax) / dist
    dy = (by - ay) / dist
    px = -dy
    py = dx
    inset = min(clearance * 0.45, max(10.0, dist * 0.2))
    lateral = max(clearance * 0.9, 18.0)
    envelope = max(clearance * 1.05, 24.0)
    min_x, max_x = sorted((ax, bx))
    min_y, max_y = sorted((ay, by))

    candidates: List[List[Tuple[int, int]]] = [[start, end]]
    for sign in (-1.0, 1.0):
        off_x = px * lateral * sign
        off_y = py * lateral * sign
        candidates.append(_dedupe_points([
            start,
            (int(round(ax + dx * inset + off_x)), int(round(ay + dy * inset + off_y))),
            (int(round(bx - dx * inset + off_x)), int(round(by - dy * inset + off_y))),
            end,
        ]))
        candidates.append(_dedupe_points([
            start,
            (int(round((ax + bx) / 2.0 + off_x)), int(round((ay + by) / 2.0 + off_y))),
            end,
        ]))

    candidates.extend([
        _dedupe_points([start, (ax, int(round(min_y - envelope))), (bx, int(round(min_y - envelope))), end]),
        _dedupe_points([start, (ax, int(round(max_y + envelope))), (bx, int(round(max_y + envelope))), end]),
        _dedupe_points([start, (int(round(min_x - envelope)), ay), (int(round(min_x - envelope)), by), end]),
        _dedupe_points([start, (int(round(max_x + envelope)), ay), (int(round(max_x + envelope)), by), end]),
        _dedupe_points([
            start,
            (ax, int(round(min_y - envelope))),
            (int(round(max_x + envelope)), int(round(min_y - envelope))),
            (int(round(max_x + envelope)), by),
            end,
        ]),
        _dedupe_points([
            start,
            (ax, int(round(max_y + envelope))),
            (int(round(max_x + envelope)), int(round(max_y + envelope))),
            (int(round(max_x + envelope)), by),
            end,
        ]),
        _dedupe_points([
            start,
            (ax, int(round(min_y - envelope))),
            (int(round(min_x - envelope)), int(round(min_y - envelope))),
            (int(round(min_x - envelope)), by),
            end,
        ]),
        _dedupe_points([
            start,
            (ax, int(round(max_y + envelope))),
            (int(round(min_x - envelope)), int(round(max_y + envelope))),
            (int(round(min_x - envelope)), by),
            end,
        ]),
    ])

    unique: List[List[Tuple[int, int]]] = []
    seen = set()
    for candidate in candidates:
        key = tuple(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _score_segment_candidate(
    coords: Sequence[Tuple[int, int]],
    start_idx: int,
    end_idx: int,
    tiles: Sequence[DetectedTile],
    radius_scale: float = 0.72,
) -> Tuple[float, List[int]]:
    touched = _touched_tile_sequence(_sample_polyline_points(coords), tiles, radius_scale=radius_scale)
    allowed = {start_idx, end_idx}
    non_target_hits = [idx for idx in touched if idx not in allowed]

    score = float(len(non_target_hits) * 1000)
    if not touched or touched[0] != start_idx:
        score += 500.0
    if not touched or touched[-1] != end_idx:
        score += 500.0
    if start_idx not in touched:
        score += 1500.0
    if end_idx not in touched:
        score += 1500.0
    score += max(0, len(touched) - 2) * 80.0
    score += _path_length(coords) * 0.08
    return score, touched


def _plan_segment_coordinates(
    start_tile: DetectedTile,
    end_tile: DetectedTile,
    tiles: Sequence[DetectedTile],
    radius_scale: float = 0.72,
) -> List[Tuple[int, int]]:
    start = (start_tile.cx, start_tile.cy)
    end = (end_tile.cx, end_tile.cy)
    clearance = max(18.0, min(start_tile.radius, end_tile.radius) * 0.95)

    best_score: Optional[float] = None
    best_path: List[Tuple[int, int]] = [start, end]
    for candidate in _segment_candidate_paths(start, end, clearance=clearance):
        score, _ = _score_segment_candidate(
            candidate,
            start_idx=start_tile.index,
            end_idx=end_tile.index,
            tiles=tiles,
            radius_scale=radius_scale,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_path = candidate
    return best_path

def plan_swipe_coordinates(
    path: Sequence[int],
    board: DetectedBoard,
    radius_scale: float = 0.72,
) -> List[Tuple[int, int]]:
    raw_coords = path_to_device_coordinates(path, board)
    if len(raw_coords) < 2:
        return raw_coords

    tile_map = board.tile_by_index
    route: List[Tuple[int, int]] = [raw_coords[0]]
    for start_idx, end_idx in zip(path, path[1:]):
        start_tile = tile_map[int(start_idx)]
        end_tile = tile_map[int(end_idx)]
        segment = _plan_segment_coordinates(
            start_tile,
            end_tile,
            board.tiles,
            radius_scale=radius_scale,
        )
        route.extend(segment[1:])
    return _dedupe_points(route)


def _normalize_vector(dx: float, dy: float) -> Tuple[float, float]:
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return (0.0, 0.0)
    return (dx / dist, dy / dist)


def _anchor_candidates_for_position(
    path: Sequence[int],
    position: int,
    board: DetectedBoard,
) -> List[Tuple[int, int]]:
    tile = board.tile_by_index[int(path[position])]
    cx, cy = tile.cx, tile.cy
    travel = max(10.0, tile.radius * 0.42)
    directions: List[Tuple[float, float]] = [(0.0, 0.0)]

    if position > 0:
        prev_tile = board.tile_by_index[int(path[position - 1])]
        incoming = _normalize_vector(cx - prev_tile.cx, cy - prev_tile.cy)
        directions.extend([
            incoming,
            (-incoming[0], -incoming[1]),
            (-incoming[1], incoming[0]),
            (incoming[1], -incoming[0]),
        ])
    else:
        incoming = (0.0, 0.0)

    if position + 1 < len(path):
        next_tile = board.tile_by_index[int(path[position + 1])]
        outgoing = _normalize_vector(next_tile.cx - cx, next_tile.cy - cy)
        directions.extend([
            outgoing,
            (-outgoing[1], outgoing[0]),
            (outgoing[1], -outgoing[0]),
        ])
    else:
        outgoing = (0.0, 0.0)

    if position > 0 and position + 1 < len(path):
        turn = _normalize_vector(incoming[0] - outgoing[0], incoming[1] - outgoing[1])
        bisector = _normalize_vector(incoming[0] + outgoing[0], incoming[1] + outgoing[1])
        if turn != (0.0, 0.0):
            directions.extend([turn, (-turn[0], -turn[1])])
        if bisector != (0.0, 0.0):
            directions.extend([bisector, (-bisector[0], -bisector[1])])

    candidates: List[Tuple[int, int]] = []
    seen = set()
    for dx, dy in directions:
        for scale in (0.0, 0.55, 0.82):
            point = (
                int(round(cx + dx * travel * scale)),
                int(round(cy + dy * travel * scale)),
            )
            if point not in seen:
                candidates.append(point)
                seen.add(point)
    return candidates[:12]


def _score_word_route(
    path: Sequence[int],
    coords: Sequence[Tuple[int, int]],
    board: DetectedBoard,
    radius_scale: float = 0.72,
) -> Tuple[float, List[int]]:
    touched = _touched_tile_sequence(
        _sample_polyline_points(coords),
        board.tiles,
        radius_scale=radius_scale,
    )
    target = [int(idx) for idx in path]
    extras = [idx for idx in touched if idx not in target]
    missing = [idx for idx in target if idx not in touched]
    wrong_order = sum(1 for actual, expected in zip(touched, target) if actual != expected)

    score = _path_length(coords) * 0.02
    # Crossing non-target tiles is usually fatal in gameplay, so penalize heavily.
    score += len(extras) * 2600.0
    score += len(missing) * 2400.0
    score += wrong_order * 120.0
    score += abs(len(touched) - len(target)) * 90.0
    if touched and touched[0] != target[0]:
        score += 1800.0
    if touched and touched[-1] != target[-1]:
        score += 1800.0
    if touched != target:
        score += 220.0
    return score, touched


def _route_confidence(path: Sequence[int], touched: Sequence[int]) -> float:
    target = [int(idx) for idx in path]
    if not target:
        return 0.0
    if list(touched) == target:
        return 1.0
    extras = sum(1 for idx in touched if idx not in target)
    missing = sum(1 for idx in target if idx not in touched)
    wrong_order = sum(1 for actual, expected in zip(touched, target) if actual != expected)
    len_delta = abs(len(touched) - len(target))
    endpoint_penalty = 0.0
    if touched:
        if touched[0] != target[0]:
            endpoint_penalty += 0.45
        if touched[-1] != target[-1]:
            endpoint_penalty += 0.45
    else:
        endpoint_penalty = 0.75
    penalty = (
        extras * 0.42
        + missing * 0.50
        + wrong_order * 0.10
        + len_delta * 0.08
        + endpoint_penalty
    )
    return max(0.0, min(1.0, 1.0 - penalty))


def _route_playability(
    path: Sequence[int],
    touched: Sequence[int],
    confidence: float,
) -> Tuple[bool, str]:
    target = [int(idx) for idx in path]
    touched_list = [int(idx) for idx in touched]
    if not touched_list:
        return False, "no_predicted_tiles"
    if touched_list[0] != target[0]:
        return False, "route_start_mismatch"
    if touched_list[-1] != target[-1]:
        return False, "route_end_mismatch"

    missing = [idx for idx in target if idx not in touched_list]
    if missing:
        return False, "route_missing_target_tiles"

    extras = [idx for idx in touched_list if idx not in target]
    if len(extras) >= 2:
        return False, "route_crosses_extra_tiles"
    if len(extras) == 1 and confidence < 0.90:
        return False, "route_crosses_extra_tiles"
    if confidence < 0.70:
        return False, "low_route_confidence"
    return True, ""


def _best_anchor_route_for_word(
    path: Sequence[int],
    board: DetectedBoard,
) -> Optional[List[Tuple[int, int]]]:
    if len(path) < 2:
        return path_to_device_coordinates(path, board)

    candidate_sets = [_anchor_candidates_for_position(path, i, board) for i in range(len(path))]
    dp: List[dict[int, Tuple[float, Optional[int]]]] = [
        {idx: (0.0, None) for idx in range(len(candidate_sets[0]))}
    ]

    for position in range(len(path) - 1):
        next_states: dict[int, Tuple[float, Optional[int]]] = {}
        for end_idx, end_point in enumerate(candidate_sets[position + 1]):
            best_state: Optional[Tuple[float, int]] = None
            for start_idx, start_point in enumerate(candidate_sets[position]):
                prev = dp[position].get(start_idx)
                if prev is None:
                    continue
                segment_score, _ = _score_word_route(
                    [path[position], path[position + 1]],
                    [start_point, end_point],
                    board,
                )
                total = prev[0] + segment_score
                if best_state is None or total < best_state[0]:
                    best_state = (total, start_idx)
            if best_state is not None:
                next_states[end_idx] = (best_state[0], best_state[1])
        if not next_states:
            return None
        dp.append(next_states)

    last_choice = min(dp[-1], key=lambda idx: dp[-1][idx][0])
    route = [candidate_sets[-1][last_choice]]
    cursor = last_choice
    for position in range(len(path) - 1, 0, -1):
        prev_choice = dp[position][cursor][1]
        if prev_choice is None:
            return None
        route.append(candidate_sets[position - 1][prev_choice])
        cursor = prev_choice
    route.reverse()
    return route


def build_best_swipe_route(
    path: Sequence[int],
    board: DetectedBoard,
) -> Tuple[List[Tuple[int, int]], List[int], str, bool]:
    target = [int(idx) for idx in path]
    strategies: List[Tuple[str, List[Tuple[int, int]]]] = [
        ("direct", path_to_device_coordinates(path, board)),
        ("segment", plan_swipe_coordinates(path, board)),
    ]
    anchor_route = _best_anchor_route_for_word(path, board)
    if anchor_route:
        strategies.append(("anchor_dp", anchor_route))

    seen = set()
    best_route = strategies[0][1]
    best_touched = target
    best_strategy = strategies[0][0]
    best_score = float("inf")

    for strategy, route in strategies:
        key = tuple(route)
        if key in seen:
            continue
        seen.add(key)
        score, touched = _score_word_route(path, route, board)
        if score < best_score:
            best_score = score
            best_route = route
            best_touched = touched
            best_strategy = strategy

    return best_route, best_touched, best_strategy, best_touched == target


# ---------------------------------------------------------------------------
# Swipe via ``input motionevent`` (continuous touch, Android 10+)
# ---------------------------------------------------------------------------

def _build_motionevent_script(
    coords: List[Tuple[int, int]],
    step_delay_ms: int = 5,
) -> str:
    """Build a shell script that performs a continuous swipe using
    ``cmd input motionevent DOWN/MOVE/UP``. This bypasses the JVM overhead
    per command, ensuring the finger is never lifted between nodes!
    """
    if len(coords) < 2:
        return ""
        
    # Interpolate intermediate points to force a smooth drag path
    # rather than jumping massive distances instantly, which games often
    # interpret as a straight-line gesture crossing intermediate nodes.
    steps_per_seg = 4
    dense_coords = [coords[0]]
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        for step in range(1, steps_per_seg + 1):
            t = step / float(steps_per_seg)
            dense_coords.append((
                int(round(x1 + (x2 - x1) * t)),
                int(round(y1 + (y2 - y1) * t))
            ))
            
    parts: List[str] = []
    x, y = dense_coords[0]
    parts.append(f"input motionevent DOWN {x} {y}")
    
    # Delay proportionately
    mini_delay = step_delay_ms / float(steps_per_seg * 1000)
    
    for x, y in dense_coords[1:]:
        if mini_delay > 0:
            parts.append(f"sleep {mini_delay:.3f}")
        parts.append(f"input motionevent MOVE {x} {y}")
        
    x, y = dense_coords[-1]
    parts.append(f"input motionevent UP {x} {y}")
    return "\n".join(parts)


def swipe_motionevent(
    coords: List[Tuple[int, int]],
    step_delay_ms: int = 5,
    timeout: float = 30.0,
) -> bool:
    """Execute a continuous multi-point swipe via ``cmd input motionevent``.

    Returns ``True`` on success.
    """
    script = _build_motionevent_script(coords, step_delay_ms=step_delay_ms)
    if not script:
        return False
    result = subprocess.run(
        [_ADB_CMD, "shell", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    ok = result.returncode == 0 and "Error" not in result.stderr and "not found" not in result.stderr
    return ok


# ---------------------------------------------------------------------------
# Swipe via chained ``input swipe`` (fallback — lifts finger between segments)
# ---------------------------------------------------------------------------

def _build_swipe_chain_script(
    coords: List[Tuple[int, int]],
    segment_duration_ms: int = 80,
) -> str:
    """Build chained ``input swipe`` commands."""
    if len(coords) < 2:
        return ""
    parts: List[str] = []
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        parts.append(
            f"input swipe {x1} {y1} {x2} {y2} {segment_duration_ms}"
        )
    return "\n".join(parts)


def swipe_chain(
    coords: List[Tuple[int, int]],
    segment_duration_ms: int = 80,
    timeout: float = 30.0,
) -> bool:
    """Execute swipe as chained ``input swipe`` commands.

    ⚠ Lifts the finger between segments — may not work for all games.
    """
    script = _build_swipe_chain_script(
        coords, segment_duration_ms=segment_duration_ms,
    )
    if not script:
        return False
    result = subprocess.run(
        [_ADB_CMD, "shell", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Unified playback
# ---------------------------------------------------------------------------

_SWIPE_METHOD: str = "motionevent"   # "motionevent" or "swipechain"


def set_swipe_method(method: str) -> None:
    """Set the global swipe method: ``'motionevent'`` or ``'swipechain'``."""
    global _SWIPE_METHOD
    if method not in ("motionevent", "swipechain"):
        raise ValueError(f"Unknown swipe method: {method}")
    _SWIPE_METHOD = method


def continuous_swipe(
    coords: List[Tuple[int, int]],
    step_delay_ms: int = 5,
) -> bool:
    """Swipe through *coords* using the configured method.

    Returns ``True`` on success.
    """
    if len(coords) < 2:
        return False

    if _SWIPE_METHOD == "motionevent":
        ok = swipe_motionevent(coords, step_delay_ms=step_delay_ms)
        if ok:
            return True
        return False

    return swipe_chain(coords, segment_duration_ms=max(50, step_delay_ms))


def playback_word_auto(
    solved_word: SolvedWord,
    board: DetectedBoard,
    dry_run: bool = False,
    step_delay_ms: int = 5,
) -> SwipeAttempt:
    """Play a single word on the device.

    In non-dry-run mode, executes a continuous swipe through the tile
    path.  Returns a :class:`SwipeAttempt` with the result.
    """
    coords, touched_path, strategy, is_exact = build_best_swipe_route(solved_word.path, board)
    confidence = _route_confidence(solved_word.path, touched_path)
    is_playable, reject_reason = _route_playability(solved_word.path, touched_path, confidence)
    total_duration = max(0, (len(coords) - 1) * step_delay_ms)
    coord_list = [list(c) for c in coords]
    touched_repr = "->".join(str(idx) for idx in touched_path) if touched_path else "none"

    if dry_run:
        return SwipeAttempt(
            word=solved_word.word,
            path=list(solved_word.path),
            coordinates=coord_list,
            duration_ms=total_duration,
            status="dry_run" if is_playable else "skipped",
            message=(
                f"Swipe route ready via {strategy} (conf={confidence:.2f})"
                if is_playable else f"Predicted swipe mismatch via {strategy}: {touched_repr} (conf={confidence:.2f})"
            ),
            commands=[],
            route_confidence=confidence,
            predicted_touched=[int(idx) for idx in touched_path],
            reject_reason="" if is_playable else reject_reason,
        )

    if len(coords) < 2:
        return SwipeAttempt(
            word=solved_word.word,
            path=list(solved_word.path),
            coordinates=coord_list,
            duration_ms=0,
            status="skipped",
            message="Word path has fewer than two tiles",
            commands=[],
            route_confidence=confidence,
            predicted_touched=[int(idx) for idx in touched_path],
            reject_reason="path_too_short",
        )

    if not is_playable:
        return SwipeAttempt(
            word=solved_word.word,
            path=list(solved_word.path),
            coordinates=coord_list,
            duration_ms=total_duration,
            status="skipped",
            message=f"Predicted swipe mismatch via {strategy}: {touched_repr} (conf={confidence:.2f})",
            commands=[],
            route_confidence=confidence,
            predicted_touched=[int(idx) for idx in touched_path],
            reject_reason=reject_reason,
        )

    ok = continuous_swipe(coords, step_delay_ms=step_delay_ms)
    return SwipeAttempt(
        word=solved_word.word,
        path=list(solved_word.path),
        coordinates=coord_list,
        duration_ms=total_duration,
        status="played" if ok else "failed",
        message=(
            f"Continuous swipe dispatched via {strategy} (conf={confidence:.2f})"
            if ok else f"Swipe command failed via {strategy} (conf={confidence:.2f})"
        ),
        commands=[],
        route_confidence=confidence,
        predicted_touched=[int(idx) for idx in touched_path],
        reject_reason="",
    )


# ---------------------------------------------------------------------------
# Legacy helpers (backward compatibility with calibration-based flow)
# ---------------------------------------------------------------------------

def path_to_screen_coordinates(
    path: Sequence[int],
    calibration,
) -> List[List[int]]:
    """Legacy: map tile-index path to screen coordinates via calibration."""
    from autoplay_v2.calibration import tile_centers_by_index
    centers = tile_centers_by_index(calibration)
    coordinates: List[List[int]] = []
    for idx in path:
        center = centers.get(int(idx))
        if center is None:
            raise ValueError(f"Path index {idx} is not present in calibration")
        coordinates.append([center.x, center.y])
    return coordinates


def generate_adb_swipe_commands(
    path: Sequence[int],
    calibration,
    adb_executable: str = "adb",
    segment_duration_ms: int = 80,
) -> List[Command]:
    coords = path_to_screen_coordinates(path, calibration)
    commands: List[Command] = []
    for start, end in zip(coords, coords[1:]):
        commands.append([
            adb_executable, "shell", "input", "swipe",
            str(start[0]), str(start[1]),
            str(end[0]), str(end[1]),
            str(segment_duration_ms),
        ])
    return commands


def _subprocess_runner(cmd: Command) -> int:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return int(result.returncode)


def playback_word(
    solved_word: SolvedWord,
    calibration,
    dry_run: bool = True,
    adb_executable: str = _ADB_CMD,
    segment_duration_ms: int = 80,
    command_runner: CommandRunner | None = None,
) -> SwipeAttempt:
    commands = generate_adb_swipe_commands(
        path=solved_word.path,
        calibration=calibration,
        adb_executable=adb_executable,
        segment_duration_ms=segment_duration_ms,
    )
    coordinates = path_to_screen_coordinates(solved_word.path, calibration)
    total_duration = max(0, len(commands) * segment_duration_ms)
    runner = command_runner or _subprocess_runner

    if dry_run:
        return SwipeAttempt(
            word=solved_word.word,
            path=list(solved_word.path),
            coordinates=coordinates,
            duration_ms=total_duration,
            status="dry_run",
            message="ADB commands generated only",
            commands=[" ".join(cmd) for cmd in commands],
        )

    if not commands:
        return SwipeAttempt(
            word=solved_word.word,
            path=list(solved_word.path),
            coordinates=coordinates,
            duration_ms=0,
            status="skipped",
            message="Word path has fewer than two tiles",
            commands=[],
        )

    failures = 0
    for cmd in commands:
        code = runner(cmd)
        if code != 0:
            failures += 1
    if failures == 0:
        status = "played"
        message = "All swipe commands dispatched"
    else:
        status = "failed"
        message = f"{failures}/{len(commands)} swipe commands failed"

    return SwipeAttempt(
        word=solved_word.word,
        path=list(solved_word.path),
        coordinates=coordinates,
        duration_ms=total_duration,
        status=status,
        message=message,
        commands=[" ".join(cmd) for cmd in commands],
    )
