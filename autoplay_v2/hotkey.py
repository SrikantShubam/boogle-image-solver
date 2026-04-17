from __future__ import annotations

import threading
from typing import Callable

from autoplay_v2.config import DEFAULT_HOTKEY


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


def run_hotkey_loop(run_once_fn: Callable[[], object], hotkey: str = DEFAULT_HOTKEY) -> None:
    try:
        import keyboard
    except ImportError as exc:
        raise RuntimeError("keyboard dependency is required for hotkey mode") from exc

    trigger = HotkeyTrigger(run_once_fn)
    keyboard.add_hotkey(hotkey, trigger.trigger)
    print(f"[autoplay-v2] Listening on hotkey '{hotkey}'. Press Ctrl+C to stop.")
    keyboard.wait()
