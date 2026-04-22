"""Floating always-on-top GUI with Start / Stop control.

Launch with::

    python -m autoplay_v2.gui          # live mode
    python -m autoplay_v2.gui --dry    # dry-run mode (no swipes)
"""
from __future__ import annotations

import sys
import threading
import tkinter as tk
from tkinter import scrolledtext

from autoplay_v2.session import auto_play_loop


class AutoplayGUI:
    """Minimal floating control panel for the Plato Crosswords bot."""

    WIDTH = 460
    HEIGHT = 620
    BG = "#1e1e2e"
    FG = "#cdd6f4"
    ACCENT = "#a6e3a1"
    STOP_COLOUR = "#f38ba8"
    BTN_FONT = ("Segoe UI", 14, "bold")
    LOG_FONT = ("Consolas", 9)

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.stop_event = threading.Event()
        self.running = False
        self._worker: threading.Thread | None = None

        self.root = tk.Tk()
        self.root.title("Crossword Bot")
        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        hdr = tk.Frame(self.root, bg=self.BG)
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(
            hdr,
            text="Plato Crosswords Bot",
            bg=self.BG,
            fg=self.FG,
            font=("Segoe UI", 13, "bold"),
        ).pack(side="left")
        mode_text = "DRY RUN" if self.dry_run else "LIVE"
        mode_colour = "#fab387" if self.dry_run else self.ACCENT
        tk.Label(
            hdr,
            text=mode_text,
            bg=self.BG,
            fg=mode_colour,
            font=("Segoe UI", 10, "bold"),
        ).pack(side="right")

        self._build_options()

        self.btn = tk.Button(
            self.root,
            text="START",
            command=self._toggle,
            bg=self.ACCENT,
            fg="#1e1e2e",
            activebackground="#94e2d5",
            activeforeground="#1e1e2e",
            font=self.BTN_FONT,
            relief="flat",
            cursor="hand2",
            width=18,
            height=2,
        )
        self.btn.pack(pady=10)

        self.log = scrolledtext.ScrolledText(
            self.root,
            wrap="word",
            bg="#181825",
            fg=self.FG,
            insertbackground=self.FG,
            font=self.LOG_FONT,
            relief="flat",
            borderwidth=0,
            state="disabled",
            height=20,
        )
        self.log.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        self.log.tag_configure("info", foreground=self.FG)
        self.log.tag_configure("ok", foreground=self.ACCENT)
        self.log.tag_configure("warn", foreground="#fab387")
        self.log.tag_configure("err", foreground=self.STOP_COLOUR)

    def _build_options(self) -> None:
        grid_frame = tk.Frame(self.root, bg=self.BG)
        grid_frame.pack(fill="x", padx=12, pady=(6, 4))

        tk.Label(
            grid_frame,
            text="Grid Size:",
            bg=self.BG,
            fg=self.FG,
            font=("Segoe UI", 10),
        ).pack(side="left")

        self.grid_size_var = tk.StringVar(value="Auto")
        for opt in ["Auto", "4x4", "5x5"]:
            tk.Radiobutton(
                grid_frame,
                text=opt,
                variable=self.grid_size_var,
                value=opt,
                bg=self.BG,
                fg=self.FG,
                selectcolor="#181825",
                activebackground=self.BG,
                activeforeground=self.FG,
                font=("Segoe UI", 10),
            ).pack(side="left", padx=8)

        strategy_frame = tk.Frame(self.root, bg=self.BG)
        strategy_frame.pack(fill="x", padx=12, pady=(0, 6))

        tk.Label(
            strategy_frame,
            text="Strategy:",
            bg=self.BG,
            fg=self.FG,
            font=("Segoe UI", 10),
        ).pack(side="left")

        self.strategy_var = tk.StringVar(value="speed")
        for label, value in [("Speed", "speed"), ("Balanced", "balanced")]:
            tk.Radiobutton(
                strategy_frame,
                text=label,
                variable=self.strategy_var,
                value=value,
                bg=self.BG,
                fg=self.FG,
                selectcolor="#181825",
                activebackground=self.BG,
                activeforeground=self.FG,
                font=("Segoe UI", 10),
            ).pack(side="left", padx=8)

        dur_frame = tk.Frame(self.root, bg=self.BG)
        dur_frame.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(
            dur_frame,
            text="Game Duration (s):",
            bg=self.BG,
            fg=self.FG,
            font=("Segoe UI", 10),
        ).pack(side="left")

        self.duration_var = tk.StringVar(value="60")
        tk.Entry(
            dur_frame,
            textvariable=self.duration_var,
            width=7,
            bg="#181825",
            fg=self.FG,
            insertbackground=self.FG,
            relief="flat",
            font=("Segoe UI", 10),
        ).pack(side="left", padx=(8, 0))

    def run(self) -> None:
        self._log("Ready. Click START to begin.\n", tag="info")
        self.root.mainloop()

    def _toggle(self) -> None:
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        self.running = True
        self.stop_event.clear()
        self.btn.configure(
            text="STOP",
            bg=self.STOP_COLOUR,
            activebackground="#eba0ac",
        )
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.running = False
        self.btn.configure(
            text="START",
            bg=self.ACCENT,
            activebackground="#94e2d5",
        )

    def _run_worker(self) -> None:
        try:
            val = self.grid_size_var.get()
            force_size = 5 if val == "5x5" else (4 if val == "4x4" else None)
            strategy = self.strategy_var.get().strip().lower() or "speed"

            try:
                game_duration_s = float(self.duration_var.get().strip())
                if game_duration_s <= 0:
                    raise ValueError("duration must be positive")
            except Exception:
                game_duration_s = 60.0
                self._status("Invalid duration; using default 60s", tag="warn")

            self._status(
                f"Running strategy={strategy}, duration={game_duration_s:.1f}s",
                tag="info",
            )

            auto_play_loop(
                stop_event=self.stop_event,
                status=self._status,
                dry_run=self.dry_run,
                force_grid_size=force_size,
                strategy=strategy,
                game_duration_s=game_duration_s,
                speed_solve_budget_s=2.0 if strategy == "speed" else None,
                speed_max_word_len=7,
                speed_max_candidates=120 if strategy == "speed" else None,
            )
        except Exception as exc:
            self._status(f"Error: {exc}", tag="err")
        finally:
            self.root.after(0, self._on_worker_done)

    def _on_worker_done(self) -> None:
        self.running = False
        self.btn.configure(
            text="START",
            bg=self.ACCENT,
            activebackground="#94e2d5",
        )

    def _status(self, msg: str, tag: str = "info") -> None:
        if tag == "info":
            if msg.startswith("Error") or msg.startswith("FAIL"):
                tag = "err"
            elif msg.startswith("WARN"):
                tag = "warn"
            elif msg.startswith("OK"):
                tag = "ok"

        def _append() -> None:
            self.log.configure(state="normal")
            self.log.insert("end", msg + "\n", tag)
            self.log.see("end")
            self.log.configure(state="disabled")

        self.root.after(0, _append)

    def _log(self, text: str, tag: str = "info") -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text, tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        self.stop_event.set()
        self.root.destroy()


def main() -> None:
    dry_run = "--dry" in sys.argv or "--dry-run" in sys.argv
    gui = AutoplayGUI(dry_run=dry_run)
    gui.run()


if __name__ == "__main__":
    main()
