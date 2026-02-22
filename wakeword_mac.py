#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
TARGET_SECONDS = 1.2
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_SECONDS)

DEFAULTS: dict[str, Any] = {
    "threshold": 0.36,
    "input_gain": 20.0,
    "cooldown_sec": 3.0,
    "check_interval_sec": 0.2,
    "vad_rms_threshold": 0.004,
    "vad_hangover_sec": 0.35,
    "vote_window": 4,
    "vote_required": 2,
    "adaptive_threshold": True,
    "adaptive_percentile": 95.0,
    "adaptive_margin": 0.04,
    "adaptive_min": 0.28,
    "adaptive_max": 0.62,
    "adaptive_warmup_sec": 20.0,
    "adaptive_min_samples": 40,
    "adaptive_window": 400,
    "pre_roll_sec": 1.0,
    "post_roll_sec": 2.0,
}


def featurize_window(y: np.ndarray) -> np.ndarray:
    if y.shape[0] < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - y.shape[0]))
    else:
        y = y[:TARGET_SAMPLES]

    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64, n_fft=400, hop_length=160, power=2.0)
    lmel = librosa.power_to_db(mel + 1e-10)

    mfcc = librosa.feature.mfcc(S=lmel, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    parts = [
        lmel.mean(axis=1),
        lmel.std(axis=1),
        np.percentile(lmel, 10, axis=1),
        np.percentile(lmel, 90, axis=1),
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        d1.mean(axis=1),
        d1.std(axis=1),
        d2.mean(axis=1),
        d2.std(axis=1),
    ]
    return np.concatenate(parts).astype(np.float32)


def _normalize_score_like_train_path(raw_scores: np.ndarray) -> np.ndarray:
    raw_scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
    mn = float(np.min(raw_scores))
    mx = float(np.max(raw_scores))
    if mx <= mn:
        return np.zeros_like(raw_scores)
    return (raw_scores - mn) / (mx - mn + 1e-12)


def predict_score(model, feats: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(feats)
        if p.ndim == 2 and p.shape[1] > 1:
            return float(p[0, 1])
        return float(p.reshape(-1)[0])
    if hasattr(model, "decision_function"):
        d = model.decision_function(feats)
        return float(_normalize_score_like_train_path(d)[0])
    return float(model.predict(feats).reshape(-1)[0])


def list_audio_devices() -> tuple[list[tuple[int, dict]], list[tuple[int, dict]]]:
    devices = sd.query_devices()
    inputs: list[tuple[int, dict]] = []
    outputs: list[tuple[int, dict]] = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            inputs.append((idx, d))
        if d.get("max_output_channels", 0) > 0:
            outputs.append((idx, d))
    return inputs, outputs


def choose_device(kind: str, devices: list[tuple[int, dict]], preferred: str | None) -> int:
    if not devices:
        raise RuntimeError(f"No {kind} devices found")

    if preferred:
        p = preferred.lower()
        for idx, d in devices:
            if p in str(d.get("name", "")).lower():
                print(f"Using {kind} device by name match: {idx} -> {d['name']}")
                return idx

    print(f"\nAvailable {kind} devices:")
    for idx, d in devices:
        sr = int(d.get("default_samplerate", 0))
        print(f"  [{idx}] {d['name']} (default_sr={sr})")

    while True:
        raw = input(f"Select {kind} device index: ").strip()
        try:
            chosen = int(raw)
        except ValueError:
            print("Please enter a numeric index")
            continue
        if any(idx == chosen for idx, _ in devices):
            return chosen
        print("Invalid device index")


def synthesize_hey_there(temp_dir: Path, voice: str | None = None) -> tuple[np.ndarray, int]:
    out_path = temp_dir / "hey_there.aiff"
    cmd = ["say", "-o", str(out_path)]
    if voice:
        cmd.extend(["-v", voice])
    cmd.append("Hey There")
    subprocess.run(cmd, check=True)

    audio, sr = sf.read(str(out_path), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio[:, 0]
    return np.asarray(audio, dtype=np.float32), int(sr)


def play_audio_on_device(audio: np.ndarray, sr: int, output_device: int) -> None:
    sd.play(audio, samplerate=sr, device=output_device, blocking=True)


def shift_append(buf: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    n = len(chunk)
    if n >= len(buf):
        return chunk[-len(buf) :].copy()
    buf[:-n] = buf[n:]
    buf[-n:] = chunk
    return buf


def load_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_profile(args: argparse.Namespace, profile: dict[str, Any], parser: argparse.ArgumentParser) -> None:
    defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}  # noqa: SLF001
    for k, v in profile.items():
        if hasattr(args, k):
            current = getattr(args, k)
            if current == defaults.get(k):
                setattr(args, k, v)


def save_profile(path: Path, args: argparse.Namespace, input_device: int | None = None, output_device: int | None = None) -> None:
    payload = {
        "threshold": float(args.threshold),
        "input_gain": float(args.input_gain),
        "cooldown_sec": float(args.cooldown_sec),
        "check_interval_sec": float(args.check_interval_sec),
        "vad_rms_threshold": float(args.vad_rms_threshold),
        "vad_hangover_sec": float(args.vad_hangover_sec),
        "vote_window": int(args.vote_window),
        "vote_required": int(args.vote_required),
        "adaptive_threshold": bool(args.adaptive_threshold),
        "adaptive_percentile": float(args.adaptive_percentile),
        "adaptive_margin": float(args.adaptive_margin),
        "adaptive_min": float(args.adaptive_min),
        "adaptive_max": float(args.adaptive_max),
        "adaptive_warmup_sec": float(args.adaptive_warmup_sec),
        "adaptive_min_samples": int(args.adaptive_min_samples),
        "adaptive_window": int(args.adaptive_window),
        "pre_roll_sec": float(args.pre_roll_sec),
        "post_roll_sec": float(args.post_roll_sec),
    }
    if input_device is not None:
        payload["input_device"] = int(input_device)
    if output_device is not None:
        payload["output_device"] = int(output_device)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-time distilled wakeword demo for macOS")
    ap.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/wakeword_model.pkl"),
    )
    ap.add_argument("--threshold", type=float, default=DEFAULTS["threshold"])
    ap.add_argument("--input-device", type=str, default=None, help="Input index or case-insensitive name substring")
    ap.add_argument("--output-device", type=str, default=None, help="Output index or case-insensitive name substring")
    ap.add_argument("--cooldown-sec", type=float, default=DEFAULTS["cooldown_sec"])
    ap.add_argument("--check-interval-sec", type=float, default=DEFAULTS["check_interval_sec"])
    ap.add_argument("--voice", type=str, default=None, help="macOS say voice name")
    ap.add_argument("--input-gain", type=float, default=DEFAULTS["input_gain"], help="Multiply input waveform before featurization")
    ap.add_argument("--show-rms", action="store_true", help="Show input RMS meter in console")
    ap.add_argument("--vad-rms-threshold", type=float, default=DEFAULTS["vad_rms_threshold"])
    ap.add_argument("--vad-hangover-sec", type=float, default=DEFAULTS["vad_hangover_sec"])
    ap.add_argument("--vote-window", type=int, default=DEFAULTS["vote_window"])
    ap.add_argument("--vote-required", type=int, default=DEFAULTS["vote_required"])
    ap.add_argument("--adaptive-threshold", action=argparse.BooleanOptionalAction, default=DEFAULTS["adaptive_threshold"])
    ap.add_argument("--adaptive-percentile", type=float, default=DEFAULTS["adaptive_percentile"])
    ap.add_argument("--adaptive-margin", type=float, default=DEFAULTS["adaptive_margin"])
    ap.add_argument("--adaptive-min", type=float, default=DEFAULTS["adaptive_min"])
    ap.add_argument("--adaptive-max", type=float, default=DEFAULTS["adaptive_max"])
    ap.add_argument("--adaptive-warmup-sec", type=float, default=DEFAULTS["adaptive_warmup_sec"])
    ap.add_argument("--adaptive-min-samples", type=int, default=DEFAULTS["adaptive_min_samples"])
    ap.add_argument("--adaptive-window", type=int, default=DEFAULTS["adaptive_window"])
    ap.add_argument("--pre-roll-sec", type=float, default=DEFAULTS["pre_roll_sec"])
    ap.add_argument("--post-roll-sec", type=float, default=DEFAULTS["post_roll_sec"])
    ap.add_argument("--shadow-dir", type=Path, default=Path("shadow_sample/runtime"))
    ap.add_argument("--disable-clip-capture", action="store_true")
    ap.add_argument("--profile", type=Path, default=Path("profiles/jabra_profile.json"))
    ap.add_argument("--write-profile", action="store_true", help="Write active runtime settings to --profile")
    ap.add_argument("--list-devices", action="store_true", help="List input/output devices and exit")
    args = ap.parse_args()

    if args.profile:
        profile = load_profile(args.profile)
        apply_profile(args, profile, ap)
        if args.input_device is None and "input_device" in profile:
            args.input_device = str(profile["input_device"])
        if args.output_device is None and "output_device" in profile:
            args.output_device = str(profile["output_device"])

    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with model_path.open("rb") as f:
        model = pickle.load(f)

    inputs, outputs = list_audio_devices()
    if args.list_devices:
        print("Input devices:")
        for idx, d in inputs:
            print(f"  [{idx}] {d['name']}")
        print("Output devices:")
        for idx, d in outputs:
            print(f"  [{idx}] {d['name']}")
        return 0

    def parse_or_choose(kind: str, devices: list[tuple[int, dict]], value: str | None) -> int:
        if value is None:
            return choose_device(kind, devices, None)
        try:
            idx = int(value)
            if any(didx == idx for didx, _ in devices):
                return idx
        except ValueError:
            pass
        return choose_device(kind, devices, value)

    input_device = parse_or_choose("input", inputs, args.input_device)
    output_device = parse_or_choose("output", outputs, args.output_device)

    if args.write_profile:
        save_profile(args.profile, args, input_device=input_device, output_device=output_device)
        print(f"Wrote profile: {args.profile}")

    print(f"\nInput device:  {input_device} -> {sd.query_devices(input_device)['name']}")
    print(f"Output device: {output_device} -> {sd.query_devices(output_device)['name']}")
    print(f"Model: {model_path}")
    print(f"Base threshold: {args.threshold:.6f}")
    print(f"VAD threshold: {args.vad_rms_threshold:.6f}")

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="oww_demo_")
    temp_dir = Path(temp_dir_obj.name)
    reply_audio, reply_sr = synthesize_hey_there(temp_dir, voice=args.voice)

    shadow_dir = args.shadow_dir.resolve()
    clip_dir = shadow_dir / "clips"
    event_log_path = shadow_dir / "events.jsonl"
    if not args.disable_clip_capture:
        clip_dir.mkdir(parents=True, exist_ok=True)
        shadow_dir.mkdir(parents=True, exist_ok=True)

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
    stop_flag = threading.Event()

    def audio_callback(indata: np.ndarray, frames: int, t, status) -> None:  # noqa: ANN001
        if status:
            print(f"[audio status] {status}", file=sys.stderr)
        mono = indata[:, 0].copy()
        try:
            q.put_nowait(mono)
        except queue.Full:
            pass

    rolling = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    pre_roll_samples = max(1, int(args.pre_roll_sec * SAMPLE_RATE))
    post_roll_samples = max(1, int(args.post_roll_sec * SAMPLE_RATE))
    pre_context = np.zeros(pre_roll_samples, dtype=np.float32)

    capture_active = False
    capture_chunks: list[np.ndarray] = []
    capture_remaining = 0
    capture_event: dict[str, Any] | None = None

    vote_hist: deque[int] = deque(maxlen=max(1, int(args.vote_window)))
    adapt_scores: deque[float] = deque(maxlen=max(10, int(args.adaptive_window)))

    start_ts = time.time()
    last_check = 0.0
    last_trigger = 0.0
    last_speech_ts = 0.0

    print("\nListening... press Ctrl+C to stop.")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=input_device,
            callback=audio_callback,
            blocksize=1024,
        ):
            while not stop_flag.is_set():
                try:
                    chunk = q.get(timeout=0.2)
                except queue.Empty:
                    continue

                n = len(chunk)
                rolling = shift_append(rolling, chunk)
                pre_context = shift_append(pre_context, chunk)

                if capture_active:
                    capture_chunks.append(chunk.copy())
                    capture_remaining -= n
                    if capture_remaining <= 0:
                        capture_active = False
                        if capture_event is not None and not args.disable_clip_capture:
                            wav = np.concatenate(capture_chunks).astype(np.float32)
                            ts_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime(capture_event["trigger_time"]))
                            clip_path = clip_dir / f"trigger_{ts_tag}_{int(capture_event['trigger_time']*1000)%1000:03d}.wav"
                            sf.write(str(clip_path), wav, SAMPLE_RATE, subtype="PCM_16")
                            capture_event["clip_path"] = str(clip_path)
                            with event_log_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(capture_event) + "\n")
                        capture_chunks = []
                        capture_event = None

                now = time.time()
                if (now - last_check) < args.check_interval_sec:
                    continue
                last_check = now

                rms_raw = float(np.sqrt(np.mean(np.square(rolling))))
                if rms_raw >= args.vad_rms_threshold:
                    last_speech_ts = now
                speech_active = (now - last_speech_ts) <= float(args.vad_hangover_sec)

                base_threshold = float(args.threshold)
                dyn_threshold = base_threshold
                if args.adaptive_threshold and (now - start_ts) >= float(args.adaptive_warmup_sec):
                    if (not speech_active) and (len(adapt_scores) >= int(args.adaptive_min_samples)):
                        p = float(np.percentile(np.asarray(adapt_scores, dtype=np.float32), float(args.adaptive_percentile)))
                        dyn_threshold = max(base_threshold, p + float(args.adaptive_margin))
                        dyn_threshold = min(max(dyn_threshold, float(args.adaptive_min)), float(args.adaptive_max))

                score = -1.0
                vote = 0
                if speech_active:
                    x = np.clip(rolling * float(args.input_gain), -1.0, 1.0).astype(np.float32)
                    feats = featurize_window(x).reshape(1, -1)
                    score = predict_score(model, feats)
                    vote = int(score >= dyn_threshold)
                    vote_hist.append(vote)
                else:
                    vote_hist.clear()

                # Update adaptive background model with non-speech checks only.
                if (not speech_active) and score >= 0.0:
                    adapt_scores.append(float(score))

                if args.show_rms:
                    sys.stdout.write(
                        f"\rscore={score:.4f} thr={dyn_threshold:.4f} base={base_threshold:.4f} rms={rms_raw:.5f} "
                        f"speech={int(speech_active)} votes={sum(vote_hist)}/{len(vote_hist)}   "
                    )
                else:
                    sys.stdout.write(f"\rscore={score:.4f} thr={dyn_threshold:.4f} votes={sum(vote_hist)}/{len(vote_hist)}   ")
                sys.stdout.flush()

                trigger_ready = (
                    speech_active
                    and len(vote_hist) >= max(1, int(args.vote_window))
                    and int(sum(vote_hist)) >= max(1, int(args.vote_required))
                    and (now - last_trigger) >= float(args.cooldown_sec)
                )

                if trigger_ready:
                    last_trigger = now
                    vote_hist.clear()

                    if not args.disable_clip_capture:
                        capture_active = True
                        capture_chunks = [pre_context.copy()]
                        capture_remaining = post_roll_samples
                        capture_event = {
                            "trigger_time": now,
                            "score": float(score),
                            "dynamic_threshold": float(dyn_threshold),
                            "base_threshold": float(base_threshold),
                            "rms": float(rms_raw),
                            "speech_active": bool(speech_active),
                            "input_device": int(input_device),
                            "output_device": int(output_device),
                            "input_gain": float(args.input_gain),
                            "vote_window": int(args.vote_window),
                            "vote_required": int(args.vote_required),
                            "decision": "triggered",
                            "label": "unknown",
                        }

                    print("\nWakeword detected -> speaking: Hey There")
                    play_audio_on_device(reply_audio, reply_sr, output_device)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        temp_dir_obj.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
