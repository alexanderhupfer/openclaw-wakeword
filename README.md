# OpenClaw Wakeword (macOS)

Self-contained first proof-of-concept for local wakeword detection using:
- handcrafted frontend features (mel/mfcc stats)
- distilled sklearn model (`wakeword_model.pkl`)
- live mic + speaker selection
- local shadow-mode capture (`events.jsonl` + clips)

## Contents
- `wakeword_mac.py` - live detector + "Hey There" response
- `models/wakeword_model.pkl` - default model artifact (200MB class)
- `profiles/jabra_profile.json` - example profile
- `scripts/label_shadow_events.py` - manual TP/FP hotkey labeling
- `scripts/autolabel_shadow_with_openai.py` - automatic labeling via OpenAI STT
- `scripts/summarize_shadow_log.py` - shadow-run summary report
- `shadow_sample/events.jsonl` - labeled sample trigger dataset
- `shadow_sample/clips/*.wav` - sample trigger clips
- `shadow_sample/shadow_summary_labeled.json` - sample summary

## Quick Start
From this folder:

```bash
uv run --with librosa --with sounddevice --with soundfile --with numpy \
  python wakeword_mac.py --list-devices
```

Run with Jabra profile defaults:

```bash
uv run --with librosa --with sounddevice --with soundfile --with numpy \
  python wakeword_mac.py --profile profiles/jabra_profile.json --show-rms
```

Explicit device run example:

```bash
uv run --with librosa --with sounddevice --with soundfile --with numpy \
  python wakeword_mac.py \
  --input-device 5 \
  --output-device 4 \
  --threshold 0.34 \
  --input-gain 20 \
  --vad-rms-threshold 0.001 \
  --vote-window 3 \
  --vote-required 2 \
  --no-adaptive-threshold \
  --show-rms
```

## Labeling Workflow
Manual TP/FP labeling:

```bash
python3 scripts/label_shadow_events.py --shadow-dir shadow_sample --only-unknown
```

Automatic labeling with OpenAI (requires `OPENAI_API_KEY` in `.env`):

```bash
uv run --with openai python scripts/autolabel_shadow_with_openai.py \
  --shadow-dir shadow_sample \
  --only-unknown
```

Generate summary after labeling:

```bash
uv run --with numpy python scripts/summarize_shadow_log.py --shadow-dir shadow_sample
```

## Mining and Training Summary
This repository package contains an inference-ready model, but the model was produced through a multi-stage mining and training workflow:

1. Data Mining (Positives/Negatives)
- Source episodes were harvested and segmented into short clips.
- Positive clips contain spoken variants of `OpenClaw`.
- Negative clips include:
  - random background speech/audio windows
  - hard negatives (near-miss phrases, confusable pronunciations)
  - long-form podcast/background corpus windows for FAR stress testing.

2. Feature Frontend
- Runtime and training use the same handcrafted feature frontend:
  - `16kHz` mono audio
  - `1.2s` analysis window
  - mel spectrogram + MFCC + delta summaries
- The `.pkl` model is backend-only and expects these feature vectors (not raw PCM directly).

3. Distillation / Size-Constrained Selection
- A stronger teacher configuration was distilled into smaller student candidates.
- Candidates were compared by:
  - held-out AUC / AP
  - transfer behavior at fixed FAR targets (threshold chosen on validation, applied to test)
  - model size constraints.
- The packaged model corresponds to the selected 200MB-class operating point.

4. Shadow-Mode Hard-Negative Loop
- During live local shadow mode, each trigger stores:
  - event metadata (`events.jsonl`)
  - short trigger-centered clip (`clips/*.wav`)
- Events are labeled `tp` / `fp` manually or auto-labeled via OpenAI transcription.
- Confirmed `fp` events are fed back as hard negatives for retraining and threshold refinement.

## Source Material Stats
- Source episode pool:
  - `1251` episodes
- Corpus-negative generation snapshot (`v3 negscale` run):
  - `48,000` negative clips
  - `3s` per clip
  - `40.0h` generated negatives total (`20h train / 10h val / 10h test`)
- Final negative-hour coverage in the training split files:
  - train: `~25.82h`
  - val: `~10.13h`
  - test: `~10.21h`
- Packaged shadow sample in this repo:
  - `28` trigger clips
  - current labels: `7 tp / 14 fp / 7 unknown`

## Notes
- This is a first local POC (not production-hardened).
- It currently does **not** integrate with OpenClaw runtime yet; that integration is planned.
- Default runtime model is the 200MB variant.
- Publishing to GitHub with the 200MB model requires Git LFS (see `.gitattributes`).

## License and Attribution
- Code and project-owned artifacts are licensed under **Apache-2.0**:
  - `LICENSE`
- Attribution and notices:
  - `NOTICE`
