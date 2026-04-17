from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_HOTKEY = "Shift"
DEFAULT_GRID_SIZE = 5
DEFAULT_TILE_PADDING = 8


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


AUTOPLAY_CONFIG_DIR = repo_root() / "config"
AUTOPLAY_DATA_DIR = repo_root() / "data"
AUTOPLAY_RUNS_DIR = repo_root() / "runs"

CALIBRATION_PATH = AUTOPLAY_CONFIG_DIR / "calibration.json"
DICTIONARY_FEEDBACK_PATH = AUTOPLAY_DATA_DIR / "dictionary_feedback.jsonl"


def ensure_runtime_dirs() -> None:
    AUTOPLAY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    AUTOPLAY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUTOPLAY_RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_json_file(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else dict(default)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return data


def save_json_file(path: Path, payload: Dict[str, Any]) -> None:
    target = path.resolve()
    root = repo_root().resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Refusing to write outside repo root: {target}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
