from __future__ import annotations

import subprocess
from typing import Callable, List, Sequence

from autoplay_v2.calibration import tile_centers_by_index
from autoplay_v2.models import CalibrationConfig, SolvedWord, SwipeAttempt

Command = List[str]
CommandRunner = Callable[[Command], int]


def path_to_screen_coordinates(path: Sequence[int], calibration: CalibrationConfig) -> List[List[int]]:
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
    calibration: CalibrationConfig,
    adb_executable: str = "adb",
    segment_duration_ms: int = 80,
) -> List[Command]:
    coords = path_to_screen_coordinates(path, calibration)
    commands: List[Command] = []
    for start, end in zip(coords, coords[1:]):
        commands.append(
            [
                adb_executable,
                "shell",
                "input",
                "swipe",
                str(start[0]),
                str(start[1]),
                str(end[0]),
                str(end[1]),
                str(segment_duration_ms),
            ]
        )
    return commands


def _subprocess_runner(cmd: Command) -> int:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return int(result.returncode)


def playback_word(
    solved_word: SolvedWord,
    calibration: CalibrationConfig,
    dry_run: bool = True,
    adb_executable: str = "adb",
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
