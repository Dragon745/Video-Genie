# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-07-15 +05:30
### Added
- **Linux support**: replaced hardcoded Windows (`d:\GitHub\...`) paths with paths derived from the project location in `transliteration_script.py` and `batch_romanize.py`.
- **Automatic hardware selection** in `whisper_script.py`: device and compute precision are chosen from the machine's real GPU capability (CTranslate2 requires CUDA compute capability 7.0+), otherwise falling back to CPU.
- **Stop button** in the GUI to cancel a running pipeline task (Whisper/Translate/Romanize/Trim).
- **ffmpeg frame-preview Visual Trimmer**, replacing the Qt media player that froze the app on Wayland. Includes a slider, ±1s/±10s nudge buttons, and a background frame-extraction worker so the UI never blocks.
- **"Watch from here" button** that opens the video in an external system player (VLC/mpv/ffplay) at the current scrub position, launched with a clean environment.
- The GUI now runs under XWayland (`xcb`) when the dependency is present, to avoid Qt multimedia hangs on Wayland.
- Comprehensive `README.md` with features, installation instructions, technical architecture overview, and project branding.

### Changed
- **Simplified Whisper controls** to Model, Language, Timestamp granularity, VAD Filter, VAD Silence Split, and Prompt; device and compute precision are now automatic.
- **Default CPU compute type is now `float32`** (full precision, matching the original Windows behaviour) instead of the faster-but-quantized `int8`.
- **Rewrote the romanizer** (`transliteration_script.py`) to use `gpt-4o` with JSON mode, id-keyed requests (robust to model spelling corrections), deduplicated context sentences, local honorific-ligature expansion, retries for dropped words, and per-request response id / token / cost logging.
- **Translation** now loads NLLB via safetensors (avoids the `torch.load` restriction on torch < 2.6) and detects the source language by majority vote across the whole SRT rather than the first few lines.
- `trim_video.py` gained a non-interactive `exact` CLI subcommand so the GUI's trim action works instead of blocking on `input()`.

### Fixed
- Empty or partial SRT files left by a crashed transcription no longer poison later runs: output is written to a temporary file and only committed on success, and an existing empty SRT is regenerated instead of trusted.
- Restored the **"Cond. on Prev"** toggle (removed during the UI simplification) so long files that loop or drift can be fixed.
- Language auto-detection no longer mislabels Urdu lectures as Arabic (from the Arabic opening lines); Arabic is now mapped to `arb_Arab` instead of silently defaulting to an English source.

## [1.0.0] - 2026-03-12 23:10 +05:30
### Changed
- Rewrote `transliteration_script.py` to use `uroman` for fast, lightweight, offline transliteration of Urdu/Hindi SRTs, removing the heavy PyTorch dependency from the `ai4bharat` wrapper.
- Replaced the global `Clear SRT Cache` button with a dynamic, per-file list in the "File Status" card. Generated SRTs now feature a dedicated `❌` button to delete individual files cleanly.
- Modified the VAD silence split slider in `gui.py` to stick to 500ms increments for a more predictable user experience.
- Added a "No Punctuation" checkbox to the GUI and a corresponding `--no_punctuation` flag to `whisper_script.py` to strip all punctuation marks from transcribed subtitles.
- Updated the default Whisper prompt in `gui.py` to be punctuation-free to encourage the model to follow a cleaner transcription style.
- **Professional UI Overhaul**: Redesigned `gui.py` using a clean 2-panel Top/Bottom architecture (Navigation/Actions on Top, Console on Bottom).
  - Implemented a custom Native QPalette (Light Mode) for a flat, modern aesthetic.
  - Replaced GroupBoxes with styled borderless Cards (`QFrame`).
  - Added dedicated visual Status Badges (✅ / ❌) for SRT translation states.
  - Replaced basic text inputs with Unicode Icons across all buttons.
  - Fixed a visibility bug where `QComboBox` dropdown items were becoming invisible on hover.
  - Added an active `QProgressBar` for background tasks that reads `[XX.X%]` regex from stdout.
  - GUI Console now explicitly records the total elapsed execution time (`Execution Time: Xm Ys`).
  - Migrated internal background task execution from `QThread` to OS-level `Subprocesses` to prevent C-level memory access crashes.
- **NLLB Translation Accuracy Upgrade**: Upgraded the translation backend from `facebook/nllb-200-distilled-600M` to `facebook/nllb-200-3.3B` for near human-level grammar mapping.
- **Whisper Transcription Accuracy Upgrades**: Enhanced the `whisper_script.py` engine with `vad_filter`, explicit device targeting, and optimized compute types.
- Refactored `scripts/trim_video.py` to allow direct functional execution and implemented an in-place replace of the original video. Added `trim_video_exact` for millisecond-precise scrubbing output.

### Added
- Upgraded the GUI to expose Advanced Whisper Settings (Compute Precision, Timestamp Granularity, VAD Filter, Custom Initial Prompts).
- Upgraded `transliteration_script.py` to act as an offline-first caching system with bulk OpenAI API fallback for missing words.
- Created `batch_romanize.py` script to use OpenAI's Batch API for translating offline dictionaries.
- Added explicit model size and file selector dropdowns to the GUI for Translation and Transliteration steps.
- Created a PyQt6 graphical interface (`gui.py`) to manage the video processing pipeline.
- Created `scripts` directory to modularize the application.
- Added `scripts/utils.py` for shared helper functions.
- **Visual Video Trimmer**: Added `VisualTrimmerDialog` with multimedia playback to visually set In/Out points using a timeline scrubber.
- **File Management**: Added capabilities to Import/Delete videos and associated cache.
- **Double-Click Shortcut**: Double-clicking a video in the list instantly opens it in the Visual Trimmer.
- **App Icon & Shortcut**: Created a custom `icon.ico`, a silent `launch_genie.vbs` runner, and a native Windows desktop shortcut.

### Fixed
- Fixed an interaction issue where `gui.py` attempted to launch `transliteration_script.py` with an unsupported `"all"` scanning argument.
- Fixed a bug where closing the Visual Trimmer while a video was playing would cause the application to hang.
- Fixed a potential crash when closing the main application while an AI pipeline thread was actively running.

### Removed
- Removed the old Manual Minute Trimmer ("Front Trim" and "Last Trim" inputs). Users will exclusively use the Visual Trimmer moving forward.
- Deleted `pipeline.py` as it was replaced by the PyQt6 interface (`gui.py`).
