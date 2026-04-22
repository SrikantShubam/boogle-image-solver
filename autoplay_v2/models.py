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
    source_method: str = "local_ocr"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "row": self.row,
            "col": self.col,
            "raw_token": self.raw_token,
            "normalized_token": self.normalized_token,
            "confidence": self.confidence,
            "low_confidence": self.low_confidence,
            "source_method": self.source_method,
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
            source_method=str(payload.get("source_method", "local_ocr")),
        )


@dataclass(frozen=True)
class OCRBoardResult:
    calibration_id: str
    grid_size: int
    tiles: List[OCRTileResult]
    normalized_grid: List[List[str]]
    has_low_confidence: bool
    debug_overlay_path: Optional[str] = None
    template_match_count: int = 0
    local_ocr_count: int = 0
    selected_geometry_mode: str = "base"
    geometry_retry_used: bool = False
    diagnostics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "grid_size": self.grid_size,
            "tiles": [tile.to_dict() for tile in self.tiles],
            "normalized_grid": [list(row) for row in self.normalized_grid],
            "has_low_confidence": self.has_low_confidence,
            "debug_overlay_path": self.debug_overlay_path,
            "template_match_count": self.template_match_count,
            "local_ocr_count": self.local_ocr_count,
            "selected_geometry_mode": self.selected_geometry_mode,
            "geometry_retry_used": self.geometry_retry_used,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OCRBoardResult":
        return cls(
            calibration_id=str(payload["calibration_id"]),
            grid_size=int(payload["grid_size"]),
            tiles=[OCRTileResult.from_dict(item) for item in payload.get("tiles", [])],
            normalized_grid=[
                [str(token) for token in row] for row in payload.get("normalized_grid", [])
            ],
            has_low_confidence=bool(payload.get("has_low_confidence", False)),
            debug_overlay_path=payload.get("debug_overlay_path"),
            template_match_count=int(payload.get("template_match_count", 0)),
            local_ocr_count=int(payload.get("local_ocr_count", 0)),
            selected_geometry_mode=str(payload.get("selected_geometry_mode", "base")),
            geometry_retry_used=bool(payload.get("geometry_retry_used", False)),
            diagnostics=payload.get("diagnostics"),
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
    commands: List[str] = field(default_factory=list)
    route_confidence: Optional[float] = None
    predicted_touched: List[int] = field(default_factory=list)
    reject_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "path": list(self.path),
            "coordinates": [list(point) for point in self.coordinates],
            "duration_ms": self.duration_ms,
            "status": self.status,
            "message": self.message,
            "commands": list(self.commands),
            "route_confidence": self.route_confidence,
            "predicted_touched": list(self.predicted_touched),
            "reject_reason": self.reject_reason,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SwipeAttempt":
        route_confidence = payload.get("route_confidence")
        return cls(
            word=str(payload["word"]),
            path=[int(item) for item in payload.get("path", [])],
            coordinates=[
                [int(point[0]), int(point[1])] for point in payload.get("coordinates", [])
            ],
            duration_ms=int(payload["duration_ms"]),
            status=str(payload["status"]),
            message=str(payload.get("message", "")),
            commands=[str(item) for item in payload.get("commands", [])],
            route_confidence=(
                None if route_confidence is None else float(route_confidence)
            ),
            predicted_touched=[
                int(item) for item in payload.get("predicted_touched", [])
            ],
            reject_reason=str(payload.get("reject_reason", "")),
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


@dataclass(frozen=True)
class CapturedFrame:
    calibration_id: str
    captured_at: str
    frame: Any
    source: str = "live"


@dataclass(frozen=True)
class DetectedTile:
    """A single tile detected by auto-detection (circle in the game board)."""
    index: int
    row: int
    col: int
    cx: int   # centre x in image pixel coordinates
    cy: int   # centre y in image pixel coordinates
    radius: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "index": self.index,
            "row": self.row,
            "col": self.col,
            "cx": self.cx,
            "cy": self.cy,
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DetectedTile":
        return cls(
            index=int(payload["index"]),
            row=int(payload["row"]),
            col=int(payload["col"]),
            cx=int(payload["cx"]),
            cy=int(payload["cy"]),
            radius=int(payload["radius"]),
        )


@dataclass(frozen=True)
class DetectedBoard:
    """Board layout detected from a screenshot via circle detection."""
    grid_size: int          # 4 or 5
    tiles: List[DetectedTile]
    roi_left: int
    roi_top: int
    roi_width: int
    roi_height: int

    @property
    def tile_by_index(self) -> Dict[int, DetectedTile]:
        return {t.index: t for t in self.tiles}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_size": self.grid_size,
            "tiles": [t.to_dict() for t in self.tiles],
            "roi_left": self.roi_left,
            "roi_top": self.roi_top,
            "roi_width": self.roi_width,
            "roi_height": self.roi_height,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DetectedBoard":
        return cls(
            grid_size=int(payload["grid_size"]),
            tiles=[DetectedTile.from_dict(t) for t in payload.get("tiles", [])],
            roi_left=int(payload["roi_left"]),
            roi_top=int(payload["roi_top"]),
            roi_width=int(payload["roi_width"]),
            roi_height=int(payload["roi_height"]),
        )
