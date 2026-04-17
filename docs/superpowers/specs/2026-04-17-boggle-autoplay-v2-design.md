# Boggle Autoplay V2 Design

**Goal:** Build a Windows-only Android Studio emulator auto-player that reads a calibrated board region, OCRs each tile, solves the board with exact tile paths, auto-swipes ranked words into the emulator, and records failed words to refine the dictionary over time.

**Status:** Approved high-level design, ready for implementation planning after user review.

## Scope

V2 is explicitly limited to:

- Windows host machine
- Android Studio emulator only
- Manual hotkey to trigger each run
- Reusable saved board calibration
- OCR from a calibrated board region
- Auto-swipe input back into the emulator
- Longest / highest-scoring words played first
- Persistent logging of failed words for dictionary refinement

V2 does **not** include:

- Physical phone automation
- Browser mirroring support
- Automatic board detection from arbitrary screenshots
- Continuous background watching for new boards
- Cloud OCR as the primary runtime path
- Full production-grade Plato dictionary extraction

## Product Intent

The current app solves uploaded screenshots. V2 pivots to a local automation tool that turns a stable emulator board region into playable actions. The user's only recurring job is to start the game and trigger a hotkey. Everything after that should be handled locally:

1. capture the calibrated board region
2. OCR each tile
3. normalize the board matrix
4. solve with exact paths
5. rank playable words
6. auto-swipe them into the emulator
7. record which words likely failed so the lexicon can be refined

## User Experience

### Calibration Mode

One-time setup for a stable emulator layout.

User flow:

1. Launch Android Studio emulator with the target game visible.
2. Open V2 calibration mode.
3. Select grid size default (`5x5` or `4x4`).
4. Mark the board region of interest in the emulator window.
5. Confirm or adjust tile grid overlay.
6. Save calibration.

Saved calibration stores:

- emulator window identity
- board ROI rectangle
- grid size default
- tile center coordinates
- swipe timing defaults
- optional per-tile padding and OCR crop parameters

### Play Mode

Normal runtime mode.

User flow:

1. Start a game round in the emulator.
2. Press the configured hotkey.
3. Tool captures the saved ROI.
4. OCR extracts the board matrix.
5. Solver generates valid words plus exact tile paths.
6. Player swipes ranked words into the emulator.
7. Session results are logged for later review.

### Review Mode

Used for debugging and dictionary refinement.

Shows:

- captured ROI image
- OCR result per tile
- final parsed matrix
- played words in ranked order
- words flagged as failed / rejected / uncertain

## Architecture

V2 should be split into focused local modules.

### 1. Capture Layer

Responsibility:

- capture only the calibrated emulator ROI on hotkey
- optionally capture a larger debug frame for investigation

Constraints:

- must be fast enough for round-time play
- must not depend on browser upload flow
- should support stable Windows desktop capture

Output:

- board-region image buffer
- metadata including timestamp and calibration id

### 2. Calibration Layer

Responsibility:

- create and persist board ROI and tile geometry
- provide a deterministic tile map for runtime

Requirements:

- calibration should be manual and explicit
- calibration must be reusable across sessions
- recalibration must be possible without deleting logs

Output:

- `calibration.json` with board geometry and runtime settings

### 3. OCR Layer

Responsibility:

- crop each tile using the saved calibration
- read only the text inside the circle
- support single letters and digraph-style tiles like `Th`, `He`, `Qu`

Runtime assumptions:

- board position is stable after calibration
- tile circles are visually consistent
- OCR runs per tile, not on the whole board

Requirements:

- preserve visible token text rather than hallucinating whole-board structure
- return normalized tokens for solver use
- keep raw OCR output for debugging

Output:

- parsed board matrix
- OCR confidence/debug metadata per tile

## Solver Layer

Responsibility:

- solve the board exactly using trie DFS
- return both valid words and exact tile paths

Requirements:

- support `4x4` and `5x5`
- support mixed single-letter and digraph tiles
- preserve deterministic path ordering
- expose ranking metadata per word

Word result model:

- `word`
- `path` as tile indices or tile coordinates
- `score`
- `length`
- `uses_bonus_tile` if applicable

## Ranking Layer

Responsibility:

- order words for practical gameplay

Primary ranking:

1. higher score first
2. longer words first
3. lower path cost / easier swipe first
4. stable alphabetical tie-break

Future ranking inputs:

- acceptance history
- observed rejection rate
- swipe duration cost

## Player Layer

Responsibility:

- convert tile paths into emulator swipe gestures
- play words back into Android Studio emulator

Requirements:

- use stable screen coordinates derived from calibration
- support drag through adjacent tile centers
- configurable gesture duration and per-segment timing
- avoid replaying duplicate words in a single run

Output:

- attempted word log with timestamps and played path coordinates

## Feedback Layer

Responsibility:

- keep a repository of words that failed to score
- allow iterative dictionary refinement

Core states:

- `accepted`
- `rejected`
- `unknown`

Minimum logged data per attempt:

- word
- parsed board id
- timestamp
- source dictionary status
- OCR board snapshot reference
- attempt result state

V2 rule:

- failed words must never be silently dropped
- every failed candidate must be persisted for review

## Data Storage

### `config/calibration.json`

Stores:

- emulator target
- ROI bounds
- grid size default
- tile centers
- OCR crop tuning
- swipe timing parameters
- hotkey configuration

### `runs/<timestamp>/`

Stores per run:

- raw ROI image
- optional debug image
- OCR board JSON
- played words JSON
- runtime diagnostics

### `data/dictionary_feedback.jsonl`

Append-only log of:

- word
- state
- board signature
- timestamp
- notes / reason

JSONL keeps the feedback store grep-friendly and easy to diff.

## Data Flow

1. User presses hotkey.
2. Capture layer grabs calibrated ROI.
3. OCR layer reads per-tile tokens.
4. Parser normalizes tokens into a board matrix.
5. Solver returns `word + path + score`.
6. Ranking orders words for play.
7. Player sends swipe gestures into emulator.
8. Session logger stores artifacts.
9. Feedback layer records failed or uncertain words for later dictionary tuning.

## Error Handling

### OCR Failure

If OCR cannot produce a valid matrix:

- abort the play run
- save ROI and OCR debug artifacts
- show the parsed partial matrix if available
- do not attempt swipes on low-confidence garbage

### Calibration Mismatch

If the ROI capture no longer matches expected board geometry:

- stop the run
- surface a recalibration-required error
- preserve debug image for inspection

### Swipe Failure

If input automation fails:

- stop remaining word playback
- save the failed word and path
- preserve the captured board and ranked word list

### Dictionary Uncertainty

If a word's acceptance result cannot be determined:

- log it as `unknown`
- do not treat it as accepted or rejected automatically

## Performance Requirements

V2 should optimize for local speed.

Targets:

- ROI capture should be near-instant
- OCR should run on per-tile crops only
- solving should remain exact and fast
- playback should begin quickly enough to be useful within a live round

Design implication:

- runtime should prefer deterministic local OCR and geometry over remote multimodal calls
- heavy whole-image model inference should not be on the critical play path

## Testing Strategy

### Calibration Tests

- verify saved ROI loads correctly
- verify tile centers are stable across sessions

### OCR Tests

- use the existing screenshot corpus as regression fixtures
- verify token extraction for letters and digraph tiles
- verify `4x4` and `5x5` parsing paths

### Solver Tests

- verify exact word generation with returned paths
- verify digraph handling
- verify ranking order

### Player Tests

- dry-run mode to emit intended swipe coordinates without input
- emulator sandbox test to confirm swipe paths hit expected tiles

### Integration Tests

- calibration -> OCR -> solve -> ranked playback on a fixed fixture
- artifact logging for failed words and OCR misreads

## Security and Safety

- keep API keys out of the frontend
- prefer fully local runtime for V2 play mode
- require explicit manual hotkey start
- never run continuous background autoplay without user action

## Open Decisions Deferred Past V2

- exact Plato lexicon acquisition
- automatic score delta detection from UI
- continuous board watching
- humanization of swipe motion
- multi-layout calibration profiles
- automatic acceptance detection from on-screen feedback

These should stay out of initial V2 implementation unless they become necessary blockers.

## Recommended Implementation Sequence

1. Persistent calibration and ROI capture
2. Per-tile OCR from saved geometry
3. Solver upgrade to return exact tile paths
4. Word ranking
5. Emulator swipe playback
6. Session artifact logging
7. Failed-word feedback repository

## Acceptance Criteria

V2 is successful when:

- user calibrates once for the emulator layout
- pressing the hotkey captures the board ROI
- OCR returns a valid `4x4` or `5x5` matrix from the calibrated board
- solver returns playable words with exact paths
- player auto-swipes ranked words into the emulator
- failed words are persisted for later dictionary refinement
- the user no longer needs to upload screenshots manually during gameplay
