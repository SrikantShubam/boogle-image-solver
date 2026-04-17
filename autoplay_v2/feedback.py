from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

from autoplay_v2.config import DICTIONARY_FEEDBACK_PATH, repo_root

FeedbackStatus = Literal["accepted", "rejected", "unknown"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_repo_path(path: Path) -> None:
    resolved = path.resolve()
    root = repo_root().resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"Feedback path must be inside repo root: {resolved}")


def append_feedback_entry(
    word: str,
    status: FeedbackStatus,
    board_signature: str,
    feedback_path: Path = DICTIONARY_FEEDBACK_PATH,
    run_id: Optional[str] = None,
) -> Dict[str, str]:
    if status not in {"accepted", "rejected", "unknown"}:
        raise ValueError(f"Invalid feedback status: {status}")
    payload: Dict[str, str] = {
        "timestamp": _utc_now_iso(),
        "word": word.strip().upper(),
        "status": status,
        "board_signature": board_signature,
    }
    if run_id:
        payload["run_id"] = run_id

    feedback_path = Path(feedback_path)
    _assert_repo_path(feedback_path)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with feedback_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def read_recent_feedback(
    feedback_path: Path = DICTIONARY_FEEDBACK_PATH,
    limit: int = 50,
) -> List[Dict[str, str]]:
    feedback_path = Path(feedback_path)
    if not feedback_path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with feedback_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    if limit <= 0:
        return []
    return rows[-limit:]
