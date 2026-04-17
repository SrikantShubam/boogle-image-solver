# Boggle Autoplay V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Windows-only Android Studio emulator autoplay tool that captures a calibrated board ROI, OCRs each tile, solves the board with exact paths, auto-swipes ranked words, and logs failed words for dictionary refinement.

**Architecture:** V2 will be a separate local Python package inside this repo, independent from the deployed Vercel app. It will reuse the existing Python solver foundation where useful, add calibration/capture/OCR/input modules around it, and persist calibration plus run artifacts locally so the user can calibrate once and play repeatedly from a hotkey. The initial default trigger is the `Shift` key, stored in config so it can be changed later without code edits.

**Tech Stack:** Python 3.10+, `opencv-python`, `mss`, `pillow`, `keyboard` or `pynput`, subprocess/`adb`, existing repo Python solver code, JSON/JSONL local storage, `pytest`

---

## File Structure

### New package

- `autoplay_v2/__init__.py`
  - package marker
- `autoplay_v2/models.py`
  - typed dataclasses for calibration, OCR result, solved word, swipe attempt, and run artifacts
- `autoplay_v2/config.py`
  - config paths, load/save helpers, default runtime settings
- `autoplay_v2/calibration.py`
  - calibration creation, ROI math, tile-center generation, calibration validation
- `autoplay_v2/capture.py`
  - Windows screen capture for calibrated ROI and debug frames
- `autoplay_v2/ocr.py`
  - per-tile crop extraction, token OCR, token normalization, debug overlays
- `autoplay_v2/solver.py`
  - path-aware trie DFS, ranking inputs, board/token helpers
- `autoplay_v2/ranking.py`
  - ranking policy for longest/highest-scoring-first ordering
- `autoplay_v2/input_driver.py`
  - ADB/emulator swipe generation and dry-run playback
- `autoplay_v2/feedback.py`
  - append-only failed/accepted/unknown word logging
- `autoplay_v2/session.py`
  - orchestration for one hotkey-triggered run and artifact persistence
- `autoplay_v2/hotkey.py`
  - manual hotkey listener and dispatch
- `autoplay_v2/cli.py`
  - CLI entrypoints for `calibrate`, `play-once`, `hotkey`, `review-last`

### Reused or modified existing files

- Modify: `boggle_winner.py`
  - extract or mirror exact solver logic needed for path-returning DFS
- Modify: `.gitignore`
  - ignore generated V2 run artifacts, calibration outputs, and caches
- Modify: `package.json`
  - optional convenience scripts for launching the local Python V2 tool if useful
- Create: `requirements-v2.txt`
  - explicit Python deps for V2

### Config and data paths

- Create: `config/calibration.json`
- Create: `data/dictionary_feedback.jsonl`
- Create: `runs/.gitkeep`

### Tests

- `tests/autoplay_v2/test_models.py`
- `tests/autoplay_v2/test_calibration.py`
- `tests/autoplay_v2/test_ocr.py`
- `tests/autoplay_v2/test_solver.py`
- `tests/autoplay_v2/test_ranking.py`
- `tests/autoplay_v2/test_feedback.py`
- `tests/autoplay_v2/test_session.py`

---

### Task 1: Scaffold the V2 package and runtime configuration

**Files:**
- Create: `autoplay_v2/__init__.py`
- Create: `autoplay_v2/models.py`
- Create: `autoplay_v2/config.py`
- Create: `requirements-v2.txt`
- Create: `tests/autoplay_v2/test_models.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing tests for config and models**

Add tests for:
- calibration path resolution
- default run/data directories
- dataclass round-trip serialization for calibration and solved-word records

Run:
```bash
python -m pytest tests/autoplay_v2/test_models.py -q
```

Expected: fail because package and models do not exist yet.

- [ ] **Step 2: Implement the minimal V2 package skeleton**

Implement:
- `CalibrationConfig`
- `TileCenter`
- `OCRTileResult`
- `SolvedWord`
- `RunArtifact`
- `load_json_file()` / `save_json_file()` helpers
- constants for `config/`, `data/`, `runs/`

Key requirement:
- all filesystem locations must remain inside the repo root
- default hotkey config value must be `Shift`

- [ ] **Step 3: Add dependency manifest and ignore generated state**

`requirements-v2.txt` should include only the initial runtime/test dependencies required by V2:
- `opencv-python`
- `mss`
- `pillow`
- `keyboard` or `pynput`
- `pytest`

`.gitignore` should ignore:
- `runs/*`
- `config/calibration.json`
- OCR debug outputs

- [ ] **Step 4: Run the new tests**

Run:
```bash
python -m pytest tests/autoplay_v2/test_models.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add autoplay_v2/__init__.py autoplay_v2/models.py autoplay_v2/config.py requirements-v2.txt tests/autoplay_v2/test_models.py .gitignore
git commit -m "feat: scaffold autoplay v2 package"
```

---

### Task 2: Implement persistent calibration and tile geometry

**Files:**
- Create: `autoplay_v2/calibration.py`
- Create: `tests/autoplay_v2/test_calibration.py`
- Modify: `autoplay_v2/models.py`
- Modify: `autoplay_v2/config.py`

- [ ] **Step 1: Write failing calibration tests**

Cover:
- ROI rectangle persistence
- generated tile centers for `4x4` and `5x5`
- stable center ordering row-major from top-left to bottom-right
- ability to reload calibration from disk

Run:
```bash
python -m pytest tests/autoplay_v2/test_calibration.py -q
```

Expected: fail because calibration logic does not exist.

- [ ] **Step 2: Implement calibration math**

Implement:
- calibration save/load
- ROI validation against positive width/height
- tile center generation from ROI and grid size
- per-tile crop rectangles with padding

Key requirement:
- calibration must not depend on live OCR to succeed

- [ ] **Step 3: Add a CLI-facing calibration creation function**

Implement a function that accepts:
- ROI bounds
- grid size
- optional tile padding
- emulator label

and writes `config/calibration.json`.

- [ ] **Step 4: Run calibration tests**

Run:
```bash
python -m pytest tests/autoplay_v2/test_calibration.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add autoplay_v2/calibration.py autoplay_v2/models.py autoplay_v2/config.py tests/autoplay_v2/test_calibration.py
git commit -m "feat: add autoplay calibration model"
```

---

### Task 3: Add ROI capture from the emulator layout

**Files:**
- Create: `autoplay_v2/capture.py`
- Create: `tests/autoplay_v2/test_session_capture.py`
- Modify: `autoplay_v2/models.py`

- [ ] **Step 1: Write failing capture tests against fixture images**

Use simple image fixtures or generated images to verify:
- ROI extraction by saved bounds
- debug-frame save path generation
- return metadata includes timestamp and calibration id

Run:
```bash
python -m pytest tests/autoplay_v2/test_session_capture.py -q
```

Expected: fail because capture module is missing.

- [ ] **Step 2: Implement capture abstraction**

Implement:
- `capture_roi(calibration)` using `mss`
- optional dry-run fixture capture path for tests
- `save_debug_capture(...)`

Key requirement:
- live capture path and test path must share the same output contract

- [ ] **Step 3: Run capture tests**

Run:
```bash
python -m pytest tests/autoplay_v2/test_session_capture.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add autoplay_v2/capture.py tests/autoplay_v2/test_session_capture.py autoplay_v2/models.py
git commit -m "feat: add autoplay ROI capture"
```

---

### Task 4: Implement per-tile OCR and token normalization

**Files:**
- Create: `autoplay_v2/ocr.py`
- Create: `tests/autoplay_v2/test_ocr.py`
- Modify: `autoplay_v2/models.py`
- Modify: `autoplay_v2/calibration.py`

- [ ] **Step 1: Write failing OCR regression tests using screenshot fixtures**

Use the screenshot corpus to test:
- tile slicing from saved calibration
- token normalization
- support for one-letter tiles and digraph-style tiles like `Th`, `He`, `Qu`
- `5x5` default and `4x4` support

Run:
```bash
python -m pytest tests/autoplay_v2/test_ocr.py -q
```

Expected: fail because OCR module does not exist.

- [ ] **Step 2: Implement tile extraction and OCR contract**

Implement:
- per-tile crop generation from calibration
- OCR per tile
- raw token preservation
- normalized token output for solver use
- debug overlay generation showing tile index and recognized token

Key requirement:
- OCR must read only the circle content and must preserve visible digraph tiles instead of collapsing them to one character

- [ ] **Step 3: Add low-confidence handling**

If OCR returns unusable tokens:
- mark the tile low-confidence
- abort autoplay
- persist debug artifacts for review

- [ ] **Step 4: Run OCR tests**

Run:
```bash
python -m pytest tests/autoplay_v2/test_ocr.py -q
```

Expected: pass against the local fixture set.

- [ ] **Step 5: Commit**

```bash
git add autoplay_v2/ocr.py autoplay_v2/models.py autoplay_v2/calibration.py tests/autoplay_v2/test_ocr.py
git commit -m "feat: add per-tile OCR for autoplay"
```

---

### Task 5: Upgrade the solver to return exact tile paths

**Files:**
- Create: `autoplay_v2/solver.py`
- Create: `tests/autoplay_v2/test_solver.py`
- Modify: `boggle_winner.py`

- [ ] **Step 1: Write failing path-aware solver tests**

Cover:
- solver returns words plus exact tile paths
- digraph tiles contribute both letters
- no tile reused in one word path
- deterministic ordering for identical score/length cases

Run:
```bash
python -m pytest tests/autoplay_v2/test_solver.py -q
```

Expected: fail because path-returning solver does not exist.

- [ ] **Step 2: Implement path-aware trie DFS**

Implement:
- exact DFS over neighbors
- path tracking as tile indices
- conversion helpers between board coordinates and flat indices
- score computation matching current game logic

Key requirement:
- keep the exact search behavior, only add path reconstruction

- [ ] **Step 3: Add ranked solved-word objects**

Each solved record must include:
- `word`
- `path`
- `score`
- `length`
- `token_count`

- [ ] **Step 4: Run solver tests**

Run:
```bash
python -m pytest tests/autoplay_v2/test_solver.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add autoplay_v2/solver.py tests/autoplay_v2/test_solver.py boggle_winner.py
git commit -m "feat: add path-aware autoplay solver"
```

---

### Task 6: Implement word ranking and failed-word feedback storage

**Files:**
- Create: `autoplay_v2/ranking.py`
- Create: `autoplay_v2/feedback.py`
- Create: `tests/autoplay_v2/test_ranking.py`
- Create: `tests/autoplay_v2/test_feedback.py`

- [ ] **Step 1: Write failing tests for ranking and feedback**

Cover:
- highest score first
- longer words before shorter words on equal score
- stable tie-break ordering
- append-only feedback logging for `accepted`, `rejected`, `unknown`

Run:
```bash
python -m pytest tests/autoplay_v2/test_ranking.py tests/autoplay_v2/test_feedback.py -q
```

Expected: fail because ranking and feedback modules do not exist.

- [ ] **Step 2: Implement ranking policy**

Implement:
- score-first sort
- length-second sort
- optional path-cost tie-break
- stable final lexical tie-break

- [ ] **Step 3: Implement feedback repository**

Implement:
- append-only `data/dictionary_feedback.jsonl`
- helper to record one attempt with board signature and timestamp
- helper to read recent feedback entries for review mode

Key requirement:
- failed words must always be written, never silently dropped

- [ ] **Step 4: Run ranking/feedback tests**

Run:
```bash
python -m pytest tests/autoplay_v2/test_ranking.py tests/autoplay_v2/test_feedback.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add autoplay_v2/ranking.py autoplay_v2/feedback.py tests/autoplay_v2/test_ranking.py tests/autoplay_v2/test_feedback.py
git commit -m "feat: add word ranking and feedback logging"
```

---

### Task 7: Implement emulator swipe playback with dry-run mode

**Files:**
- Create: `autoplay_v2/input_driver.py`
- Create: `tests/autoplay_v2/test_input_driver.py`
- Modify: `autoplay_v2/models.py`
- Modify: `autoplay_v2/calibration.py`

- [ ] **Step 1: Write failing playback tests**

Cover:
- conversion from tile path to screen coordinates
- drag path generation from tile centers
- dry-run mode emits commands without sending input

Run:
```bash
python -m pytest tests/autoplay_v2/test_input_driver.py -q
```

Expected: fail because input driver does not exist.

- [ ] **Step 2: Implement swipe command generation**

Implement:
- tile path -> coordinate path
- swipe timing model
- ADB command generation
- dry-run mode for test and debug

Key requirement:
- coordinate generation must be deterministic and entirely derived from calibration

- [ ] **Step 3: Add live playback adapter**

Implement subprocess-driven ADB dispatch behind a narrow adapter so tests can mock it easily.

- [ ] **Step 4: Run playback tests**

Run:
```bash
python -m pytest tests/autoplay_v2/test_input_driver.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add autoplay_v2/input_driver.py tests/autoplay_v2/test_input_driver.py autoplay_v2/models.py autoplay_v2/calibration.py
git commit -m "feat: add emulator swipe playback"
```

---

### Task 8: Wire the session orchestrator and manual hotkey runtime

**Files:**
- Create: `autoplay_v2/session.py`
- Create: `autoplay_v2/hotkey.py`
- Create: `autoplay_v2/cli.py`
- Create: `tests/autoplay_v2/test_session.py`
- Modify: `requirements-v2.txt`

- [ ] **Step 1: Write failing integration-style session tests**

Cover:
- `play-once` orchestration order: capture -> OCR -> solve -> rank -> playback -> log
- abort on OCR failure
- failed-word logging occurs on playback rejection path
- hotkey handler calls the session runner exactly once

Run:
```bash
python -m pytest tests/autoplay_v2/test_session.py -q
```

Expected: fail because session and hotkey modules do not exist.

- [ ] **Step 2: Implement session orchestration**

Implement:
- `run_once()`
- artifact folder creation under `runs/<timestamp>/`
- OCR debug writes
- ranked word playback loop
- feedback writes

- [ ] **Step 3: Implement CLI and hotkey entrypoints**

CLI commands:
- `calibrate`
- `play-once`
- `hotkey`
- `review-last`

Hotkey runtime:
- register manual trigger using `Shift` by default
- call `run_once()`
- prevent overlapping runs

- [ ] **Step 4: Run session tests**

Run:
```bash
python -m pytest tests/autoplay_v2/test_session.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add autoplay_v2/session.py autoplay_v2/hotkey.py autoplay_v2/cli.py tests/autoplay_v2/test_session.py requirements-v2.txt
git commit -m "feat: wire autoplay session runtime"
```

---

### Task 9: Add end-to-end fixture verification and operator documentation

**Files:**
- Create: `tests/autoplay_v2/test_e2e_fixture.py`
- Create: `README.autoplay-v2.md`
- Modify: `package.json`

- [ ] **Step 1: Write a fixture-driven end-to-end test**

Cover:
- load saved calibration fixture
- capture from static fixture image
- OCR into board
- solve into ranked words with paths
- generate dry-run swipes
- persist a session artifact

Run:
```bash
python -m pytest tests/autoplay_v2/test_e2e_fixture.py -q
```

Expected: fail until all orchestration pieces are wired together.

- [ ] **Step 2: Implement any missing glue for the fixture flow**

Keep this step narrow:
- only fix integration issues uncovered by the end-to-end test
- do not add new features beyond plan scope

- [ ] **Step 3: Write operator docs**

Document:
- dependency install
- ADB prerequisites
- calibration flow
- hotkey usage
- dry-run mode
- where failed words are logged

- [ ] **Step 4: Run the full V2 test suite**

Run:
```bash
python -m pytest tests/autoplay_v2 -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tests/autoplay_v2/test_e2e_fixture.py README.autoplay-v2.md package.json
git commit -m "docs: add autoplay v2 operator guide"
```

---

## Plan Self-Review

### Spec coverage

- Manual hotkey start: covered in Task 8
- Default `Shift` trigger: covered in Task 1 config defaults and Task 8 hotkey runtime
- Persistent calibration: covered in Task 2
- Emulator-only capture: covered in Task 3
- Per-tile OCR: covered in Task 4
- Exact word paths: covered in Task 5
- Longest/highest-scoring ranking: covered in Task 6
- Auto-swipe playback: covered in Task 7
- Failed-word repository: covered in Task 6 and Task 8
- Screenshot corpus for calibration/OCR tests: covered in Task 4 and Task 9

### Placeholder scan

No `TODO`, `TBD`, or “implement later” placeholders remain. Each task identifies concrete files, tests, commands, and outputs.

### Type consistency

Core shared types used consistently across tasks:
- `CalibrationConfig`
- `OCRTileResult`
- `SolvedWord`
- `RunArtifact`

### Scope check

This plan stays within one subsystem: a local Python autoplay tool. It does not mix in Vercel deployment work, physical device support, or cloud OCR dependence.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-17-boggle-autoplay-v2.md`.
