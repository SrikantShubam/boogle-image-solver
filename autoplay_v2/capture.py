from __future__ import annotations

import io
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from autoplay_v2.models import CalibrationConfig, CapturedFrame


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# ADB screencap — captures directly from the device (no multi-monitor issues)
# ---------------------------------------------------------------------------
import shutil
import sys
import os

_ADB_CMD = "adb"
if not shutil.which("adb"):
    _fallback = os.path.join(os.path.dirname(sys.executable), "adb.exe")
    if os.path.exists(_fallback):
        _ADB_CMD = _fallback

def capture_adb_screenshot(timeout: float = 10.0) -> np.ndarray:
    """Capture a screenshot from the connected Android device via ADB.

    Returns an image as a BGR numpy array (OpenCV convention).
    Raises RuntimeError if ADB is not available or no device is connected.
    """
    try:
        result = subprocess.run(
            [_ADB_CMD, "exec-out", "screencap", "-p"],
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "adb executable not found. Make sure ADB is installed and in PATH."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("ADB screencap timed out — is the device connected?")

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ADB screencap failed (rc={result.returncode}): {stderr}")

    if len(result.stdout) < 100:
        raise RuntimeError(
            "ADB screencap returned no data — is USB debugging enabled?"
        )

    image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
    frame_rgb = np.array(image)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def get_device_screen_size(timeout: float = 5.0) -> tuple[int, int]:
    """Return ``(width, height)`` of the connected device's display."""
    try:
        result = subprocess.run(
            [_ADB_CMD, "shell", "wm", "size"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return 1080, 1920  # safe fallback

    import re
    match = re.search(r"(\d+)x(\d+)", result.stdout)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1080, 1920


# ---------------------------------------------------------------------------
# Screen capture via mss (fallback — captures a region of the desktop)
# ---------------------------------------------------------------------------

def capture_screen_region(
    left: int, top: int, width: int, height: int,
) -> np.ndarray:
    """Capture a region of the desktop using mss.  Returns BGR."""
    try:
        import mss
    except ImportError as exc:
        raise RuntimeError("mss is required for screen capture") from exc

    monitor = {"left": left, "top": top, "width": width, "height": height}
    with mss.mss() as sct:
        shot = sct.grab(monitor)
    bgra = np.array(shot)
    return bgra[:, :, :3][:, :, ::-1]  # BGRA → BGR


def capture_full_screen(monitor_index: int = 0) -> np.ndarray:
    """Capture the entire desktop / primary monitor.  Returns BGR."""
    try:
        import mss
    except ImportError as exc:
        raise RuntimeError("mss is required for screen capture") from exc

    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        shot = sct.grab(monitor)
    bgra = np.array(shot)
    return bgra[:, :, :3][:, :, ::-1]


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility with existing sessions)
# ---------------------------------------------------------------------------

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
