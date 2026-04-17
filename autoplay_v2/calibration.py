from __future__ import annotations

from dataclasses import replace
import platform
from typing import Callable, List, Optional, Tuple

from autoplay_v2.config import (
    CALIBRATION_PATH,
    DEFAULT_GRID_SIZE,
    DEFAULT_HOTKEY,
    DEFAULT_TILE_PADDING,
    load_json_file,
    save_json_file,
)
from autoplay_v2.models import CalibrationConfig, TileCenter


def validate_roi(roi_left: int, roi_top: int, roi_width: int, roi_height: int) -> None:
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI width and height must be positive")
    if roi_left < 0 or roi_top < 0:
        raise ValueError("ROI origin must be non-negative")


def generate_tile_centers(
    roi_left: int,
    roi_top: int,
    roi_width: int,
    roi_height: int,
    grid_size: int,
) -> List[TileCenter]:
    if grid_size not in (4, 5):
        raise ValueError("Grid size must be 4 or 5")
    validate_roi(roi_left, roi_top, roi_width, roi_height)

    cell_w = roi_width / grid_size
    cell_h = roi_height / grid_size
    centers: List[TileCenter] = []
    idx = 0
    for row in range(grid_size):
        for col in range(grid_size):
            cx = int(round(roi_left + (col + 0.5) * cell_w))
            cy = int(round(roi_top + (row + 0.5) * cell_h))
            centers.append(TileCenter(index=idx, row=row, col=col, x=cx, y=cy))
            idx += 1
    return centers


def generate_tile_crop_rects(
    calibration: CalibrationConfig,
    relative_to_roi: bool = True,
) -> List[Tuple[int, int, int, int]]:
    cell_w = calibration.roi_width / calibration.grid_size
    cell_h = calibration.roi_height / calibration.grid_size
    pad = max(0, calibration.tile_padding)
    rects: List[Tuple[int, int, int, int]] = []
    for center in calibration.tile_centers:
        left = int(round(center.col * cell_w + pad))
        top = int(round(center.row * cell_h + pad))
        right = int(round((center.col + 1) * cell_w - pad))
        bottom = int(round((center.row + 1) * cell_h - pad))
        if not relative_to_roi:
            left += calibration.roi_left
            top += calibration.roi_top
            right += calibration.roi_left
            bottom += calibration.roi_top
        width = max(1, right - left)
        height = max(1, bottom - top)
        rects.append((left, top, width, height))
    return rects


def create_calibration(
    roi_left: int,
    roi_top: int,
    roi_width: int,
    roi_height: int,
    grid_size: int = DEFAULT_GRID_SIZE,
    tile_padding: int = DEFAULT_TILE_PADDING,
    emulator_label: str = "android-studio",
    trigger_hotkey: str = DEFAULT_HOTKEY,
    calibration_id: str = "default",
) -> CalibrationConfig:
    validate_roi(roi_left, roi_top, roi_width, roi_height)
    centers = generate_tile_centers(
        roi_left=roi_left,
        roi_top=roi_top,
        roi_width=roi_width,
        roi_height=roi_height,
        grid_size=grid_size,
    )
    return CalibrationConfig(
        calibration_id=calibration_id,
        emulator_label=emulator_label,
        grid_size=grid_size,
        roi_left=roi_left,
        roi_top=roi_top,
        roi_width=roi_width,
        roi_height=roi_height,
        tile_padding=tile_padding,
        trigger_hotkey=trigger_hotkey,
        tile_centers=centers,
    )


def create_and_save_calibration(
    roi_left: int,
    roi_top: int,
    roi_width: int,
    roi_height: int,
    grid_size: int = DEFAULT_GRID_SIZE,
    tile_padding: int = DEFAULT_TILE_PADDING,
    emulator_label: str = "android-studio",
    trigger_hotkey: str = DEFAULT_HOTKEY,
    calibration_id: str = "default",
) -> CalibrationConfig:
    calibration = create_calibration(
        roi_left=roi_left,
        roi_top=roi_top,
        roi_width=roi_width,
        roi_height=roi_height,
        grid_size=grid_size,
        tile_padding=tile_padding,
        emulator_label=emulator_label,
        trigger_hotkey=trigger_hotkey,
        calibration_id=calibration_id,
    )
    save_calibration(calibration)
    return calibration


def prompt_grid_size(
    default_grid_size: int = DEFAULT_GRID_SIZE,
    input_fn: Callable[[str], str] = input,
) -> int:
    prompt = (
        f"Grid size selection: press '4' then Enter for 4x4, "
        f"or just Enter to keep default {default_grid_size}x{default_grid_size}: "
    )
    raw = input_fn(prompt).strip()
    if raw == "4":
        return 4
    return default_grid_size


def select_roi_with_overlay() -> Tuple[int, int, int, int]:
    if platform.system().lower() != "windows":
        raise RuntimeError("Interactive calibration overlay is supported on Windows only")

    try:
        import mss
    except ImportError as exc:
        raise RuntimeError("mss is required for interactive calibration capture") from exc

    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError as exc:
        raise RuntimeError("tkinter and pillow are required for interactive calibration overlay") from exc

    with mss.mss() as sct:
        monitor = sct.monitors[0]
        shot = sct.grab(monitor)

    screenshot = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    root = tk.Tk()
    root.title("Autoplay V2 Calibration")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    canvas = tk.Canvas(root, highlightthickness=0, cursor="crosshair")
    canvas.pack(fill="both", expand=True)

    tk_img = ImageTk.PhotoImage(screenshot)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

    state = {
        "start_x": None,
        "start_y": None,
        "rect_id": None,
        "roi": None,
        "done": False,
        "cancelled": False,
    }

    info_text = (
        "Drag one bounding box around the board. "
        "Press Enter to confirm, Esc to cancel."
    )
    canvas.create_text(20, 20, anchor="nw", text=info_text, fill="yellow", font=("Segoe UI", 16, "bold"))

    def on_press(event):
        state["start_x"] = event.x
        state["start_y"] = event.y
        if state["rect_id"] is not None:
            canvas.delete(state["rect_id"])
        state["rect_id"] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="lime", width=3)

    def on_drag(event):
        if state["rect_id"] is None or state["start_x"] is None or state["start_y"] is None:
            return
        canvas.coords(state["rect_id"], state["start_x"], state["start_y"], event.x, event.y)

    def on_release(event):
        if state["start_x"] is None or state["start_y"] is None:
            return
        x1, y1 = state["start_x"], state["start_y"]
        x2, y2 = event.x, event.y
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        width = max(1, right - left)
        height = max(1, bottom - top)
        state["roi"] = (
            int(monitor["left"] + left),
            int(monitor["top"] + top),
            int(width),
            int(height),
        )

    def on_confirm(_event=None):
        if state["roi"] is None:
            return
        state["done"] = True
        root.quit()

    def on_cancel(_event=None):
        state["cancelled"] = True
        root.quit()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Return>", on_confirm)
    root.bind("<Escape>", on_cancel)

    try:
        root.mainloop()
    finally:
        root.destroy()

    if state["cancelled"] or state["roi"] is None:
        raise RuntimeError("Calibration cancelled")

    return state["roi"]


def calibrate_interactive(
    emulator_label: str = "android-studio",
    trigger_hotkey: str = DEFAULT_HOTKEY,
    calibration_id: str = "default",
    tile_padding: int = DEFAULT_TILE_PADDING,
    save_path=CALIBRATION_PATH,
    roi_selector: Callable[[], Tuple[int, int, int, int]] = select_roi_with_overlay,
    grid_input_fn: Callable[[str], str] = input,
) -> CalibrationConfig:
    roi_left, roi_top, roi_width, roi_height = roi_selector()
    grid_size = prompt_grid_size(input_fn=grid_input_fn)
    calibration = create_calibration(
        roi_left=roi_left,
        roi_top=roi_top,
        roi_width=roi_width,
        roi_height=roi_height,
        grid_size=grid_size,
        tile_padding=tile_padding,
        emulator_label=emulator_label,
        trigger_hotkey=trigger_hotkey,
        calibration_id=calibration_id,
    )
    save_calibration(calibration, path=save_path)
    return calibration


def tile_centers_by_index(calibration: CalibrationConfig) -> dict[int, TileCenter]:
    return {center.index: center for center in calibration.tile_centers}


def save_calibration(calibration: CalibrationConfig, path=CALIBRATION_PATH) -> None:
    save_json_file(path, calibration.to_dict())


def load_calibration(path=CALIBRATION_PATH) -> CalibrationConfig:
    payload = load_json_file(path)
    calibration = CalibrationConfig.from_dict(payload)
    # Normalize ordering to row-major in case external edits changed order.
    ordered = sorted(calibration.tile_centers, key=lambda c: (c.row, c.col, c.index))
    if ordered != calibration.tile_centers:
        calibration = replace(calibration, tile_centers=ordered)
    return calibration
