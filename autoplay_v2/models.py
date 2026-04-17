from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TileCenter:
    index: int
    row: int
    col: int
    x: int
    y: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "index": self.index,
            "row": self.row,
            "col": self.col,
            "x": self.x,
            "y": self.y,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TileCenter":
        return cls(
            index=int(payload["index"]),
            row=int(payload["row"]),
            col=int(payload["col"]),
            x=int(payload["x"]),
            y=int(payload["y"]),
        )


@dataclass(frozen=True)
class CalibrationConfig:
    calibration_id: str
    emulator_label: str
    grid_size: int
    roi_left: int
    roi_top: int
    roi_width: int
    roi_height: int
    tile_padding: int = 0
    trigger_hotkey: str = "Shift"
    tile_centers: List[TileCenter] = field(default_factory=list)

    @property
    def roi_right(self) -> int:
        return self.roi_left + self.roi_width

    @property
    def roi_bottom(self) -> int:
        return self.roi_top + self.roi_height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "emulator_label": self.emulator_label,
            "grid_size": self.grid_size,
            "roi_left": self.roi_left,
            "roi_top": self.roi_top,
            "roi_width": self.roi_width,
            "roi_height": self.roi_height,
            "tile_padding": self.tile_padding,
            "trigger_hotkey": self.trigger_hotkey,
            "tile_centers": [center.to_dict() for center in self.tile_centers],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CalibrationConfig":
        centers = [TileCenter.from_dict(item) for item in payload.get("tile_centers", [])]
        return cls(
            calibration_id=str(payload["calibration_id"]),
            emulator_label=str(payload["emulator_label"]),
            grid_size=int(payload["grid_size"]),
            roi_left=int(payload["roi_left"]),
            roi_top=int(payload["roi_top"]),
            roi_width=int(payload["roi_width"]),
            roi_height=int(payload["roi_height"]),
            tile_padding=int(payload.get("tile_padding", 0)),
            trigger_hotkey=str(payload.get("trigger_hotkey", "Shift")),
            tile_centers=centers,
        )


@dataclass(frozen=True)
class OCRTileResult:
    index: int
    row: int
    col: int
    raw_token: str
    normalized_token: str
    confidence: float
    low_confidence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "row": self.row,
            "col": self.col,
            "raw_token": self.raw_token,
            "normalized_token": self.normalized_token,
            "confidence": self.confidence,
            "low_confidence": self.low_confidence,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OCRTileResult":
        return cls(
            index=int(payload["index"]),
            row=int(payload["row"]),
            col=int(payload["col"]),
            raw_token=str(payload.get("raw_token", "")),
            normalized_token=str(payload.get("normalized_token", "")),
            confidence=float(payload.get("confidence", 0.0)),
            low_confidence=bool(payload.get("low_confidence", False)),
        )


@dataclass(frozen=True)
class SolvedWord:
    word: str
    path: List[int]
    score: float
    length: int
    token_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "path": list(self.path),
            "score": self.score,
            "length": self.length,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SolvedWord":
        return cls(
            word=str(payload["word"]),
            path=[int(item) for item in payload.get("path", [])],
            score=float(payload["score"]),
            length=int(payload["length"]),
            token_count=int(payload["token_count"]),
        )


@dataclass(frozen=True)
class SwipeAttempt:
    word: str
    path: List[int]
    coordinates: List[List[int]]
    duration_ms: int
    status: str
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "path": list(self.path),
            "coordinates": [list(point) for point in self.coordinates],
            "duration_ms": self.duration_ms,
            "status": self.status,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SwipeAttempt":
        return cls(
            word=str(payload["word"]),
            path=[int(item) for item in payload.get("path", [])],
            coordinates=[
                [int(point[0]), int(point[1])] for point in payload.get("coordinates", [])
            ],
            duration_ms=int(payload["duration_ms"]),
            status=str(payload["status"]),
            message=str(payload.get("message", "")),
        )


@dataclass(frozen=True)
class RunArtifact:
    run_id: str
    calibration_id: str
    created_at: str
    board_tokens: List[str]
    solved_words: List[SolvedWord]
    swipe_attempts: List[SwipeAttempt]
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "calibration_id": self.calibration_id,
            "created_at": self.created_at,
            "board_tokens": list(self.board_tokens),
            "solved_words": [item.to_dict() for item in self.solved_words],
            "swipe_attempts": [item.to_dict() for item in self.swipe_attempts],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunArtifact":
        return cls(
            run_id=str(payload["run_id"]),
            calibration_id=str(payload["calibration_id"]),
            created_at=str(payload.get("created_at", utc_now_iso())),
            board_tokens=[str(item) for item in payload.get("board_tokens", [])],
            solved_words=[SolvedWord.from_dict(item) for item in payload.get("solved_words", [])],
            swipe_attempts=[
                SwipeAttempt.from_dict(item) for item in payload.get("swipe_attempts", [])
            ],
            notes=payload.get("notes"),
        )
