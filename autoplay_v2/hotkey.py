from __future__ import annotations

import platform
import threading
from typing import Callable

from autoplay_v2.config import DEFAULT_CALIBRATE_HOTKEY, DEFAULT_PLAY_HOTKEY


class HotkeyTrigger:
    def __init__(self, run_once_fn: Callable[[], object]):
        self._run_once_fn = run_once_fn
        self._lock = threading.Lock()
        self._running = False

    def trigger(self) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = True
        try:
            self._run_once_fn()
            return True
        finally:
            with self._lock:
                self._running = False


class DualHotkeyRuntime:
    def __init__(
        self,
        play_once_fn: Callable[[], object],
        calibrate_fn: Callable[[], object],
    ):
        self._play_once_fn = play_once_fn
        self._calibrate_fn = calibrate_fn
        self._trigger = HotkeyTrigger(self._dispatch)
        self._pending_action: str | None = None
        self._action_lock = threading.Lock()

    def _dispatch(self) -> None:
        action = None
        with self._action_lock:
            action = self._pending_action
            self._pending_action = None
        if action == "calibrate":
            self._calibrate_fn()
        elif action == "play":
            self._play_once_fn()

    def trigger_play(self) -> bool:
        with self._action_lock:
            self._pending_action = "play"
        return self._trigger.trigger()

    def trigger_calibrate(self) -> bool:
        with self._action_lock:
            self._pending_action = "calibrate"
        return self._trigger.trigger()


def run_hotkey_loop(
    play_once_fn: Callable[[], object],
    calibrate_fn: Callable[[], object],
    play_hotkey: str = DEFAULT_PLAY_HOTKEY,
    calibrate_hotkey: str = DEFAULT_CALIBRATE_HOTKEY,
) -> None:
    if platform.system().lower() != "windows":
        raise RuntimeError("Hotkey runtime is supported on Windows only")

    try:
        import keyboard
    except ImportError as exc:
        raise RuntimeError("keyboard dependency is required for hotkey mode") from exc

    runtime = DualHotkeyRuntime(play_once_fn=play_once_fn, calibrate_fn=calibrate_fn)
    keyboard.add_hotkey(play_hotkey, runtime.trigger_play)
    keyboard.add_hotkey(calibrate_hotkey, runtime.trigger_calibrate)
    print(
        f"[autoplay-v2] Listening hotkeys: play='{play_hotkey}', calibrate='{calibrate_hotkey}'. "
        "Press Ctrl+C to stop."
    )
    keyboard.wait()
