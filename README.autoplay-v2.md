# Boggle Autoplay V2 (Local MVP)

This subsystem is a local Python workflow for Android Studio emulator autoplay:

1. Load persistent calibration (`config/calibration.json`)
2. Capture calibrated ROI
3. OCR per tile (supports digraph tokens like `Th`, `He`, `Qu`)
4. Solve with exact path-aware DFS (no tile reuse per word path)
5. Rank words (score first, then length)
6. Dry-run or live ADB swipe playback
7. Append feedback (`data/dictionary_feedback.jsonl`)

## 1) Install

```bash
python -m pip install -r requirements-v2.txt
```

## 2) ADB prerequisites (for live mode)

- Android Studio emulator running
- `adb` available in `PATH`
- Device visible:

```bash
adb devices
```

## 3) Calibrate once

```bash
python -m autoplay_v2.cli calibrate --left 100 --top 200 --width 500 --height 500 --grid-size 5 --tile-padding 8 --emulator-label android-studio --hotkey Shift
```

This writes `config/calibration.json`.

## 4) Play once

Dry-run against live capture:

```bash
python -m autoplay_v2.cli play-once
```

Dry-run against fixture screenshot:

```bash
python -m autoplay_v2.cli play-once --fixture 1.jpeg
```

Live ADB swipe mode:

```bash
python -m autoplay_v2.cli play-once --live
```

## 5) Hotkey runtime

Default hotkey is `Shift`.

```bash
python -m autoplay_v2.cli hotkey
```

Live ADB hotkey runtime:

```bash
python -m autoplay_v2.cli hotkey --live
```

## 6) Review latest run

```bash
python -m autoplay_v2.cli review-last --show-feedback
```

## Artifact/feedback locations

- Run artifacts: `runs/<run_id>/run.json` and debug images
- Calibration: `config/calibration.json`
- Feedback log: `data/dictionary_feedback.jsonl`

Rejected playback attempts are always logged to feedback (append-only).
