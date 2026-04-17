from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from autoplay_v2.models import CalibrationConfig, CapturedFrame


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _crop_frame(frame: np.ndarray, calibration: CalibrationConfig) -> np.ndarray:
    top = calibration.roi_top
    left = calibration.roi_left
    bottom = top + calibration.roi_height
    right = left + calibration.roi_width
    if top < 0 or left < 0 or bottom > frame.shape[0] or right > frame.shape[1]:
        raise ValueError("ROI is outside the captured frame bounds")
    return frame[top:bottom, left:right].copy()


def _capture_from_fixture(
    calibration: CalibrationConfig,
    fixture_path: Path,
) -> np.ndarray:
    image = Image.open(fixture_path).convert("RGB")
    frame = np.array(image)
    return _crop_frame(frame, calibration)


def _capture_live(calibration: CalibrationConfig) -> np.ndarray:
    try:
        import mss
    except ImportError as exc:
        raise RuntimeError("mss is required for live capture") from exc

    monitor = {
        "top": calibration.roi_top,
        "left": calibration.roi_left,
        "width": calibration.roi_width,
        "height": calibration.roi_height,
    }
    with mss.mss() as sct:
        shot = sct.grab(monitor)
    bgra = np.array(shot)
    return bgra[:, :, :3][:, :, ::-1]


def capture_roi(
    calibration: CalibrationConfig,
    fixture_path: Optional[Path] = None,
) -> CapturedFrame:
    captured_at = _utc_now_iso()
    if fixture_path is not None:
        frame = _capture_from_fixture(calibration, Path(fixture_path))
        source = "fixture"
    else:
        frame = _capture_live(calibration)
        source = "live"
    return CapturedFrame(
        calibration_id=calibration.calibration_id,
        captured_at=captured_at,
        frame=frame,
        source=source,
    )


def save_debug_capture(
    frame: np.ndarray,
    out_dir: Path,
    calibration_id: str,
    captured_at: str,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stamp = (
        captured_at.replace(":", "-")
        .replace(".", "-")
        .replace("+00-00", "Z")
        .replace("+", "_")
    )
    out_path = out_dir / f"capture_{calibration_id}_{safe_stamp}.png"
    Image.fromarray(frame).save(out_path)
    return out_path
