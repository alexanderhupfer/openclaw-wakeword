#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import select
import subprocess
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path


def normalize_label(raw: str) -> str | None:
    v = raw.strip().lower()
    if v in {"t", "tp", "true", "true_positive", "positive"}:
        return "tp"
    if v in {"f", "fp", "false", "false_positive", "negative"}:
        return "fp"
    if v in {"u", "unk", "unknown", "?"}:
        return "unknown"
    if v in {"s", "skip", ""}:
        return None
    return None


@contextmanager
def cbreak_stdin():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key_nonblocking(timeout_s: float = 0.05) -> str | None:
    r, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not r:
        return None
    ch = sys.stdin.read(1)
    if not ch:
        return None
    return ch.lower()


def save_events(path: Path, events: list[dict]) -> None:
    out = [json.dumps(obj, ensure_ascii=True) for obj in events]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive TP/FP labeling for shadow-mode events")
    ap.add_argument("--shadow-dir", type=Path, default=Path("shadow_mode/jabra"))
    ap.add_argument("--start", type=int, default=0, help="Start index in events list")
    ap.add_argument("--only-unknown", action="store_true", help="Only review events labeled unknown")
    ap.add_argument("--no-play", action="store_true", help="Do not auto-play clip")
    args = ap.parse_args()

    events_path = args.shadow_dir.resolve() / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events log: {events_path}")

    lines = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]

    print(f"Loaded {len(events)} events from {events_path}")
    print("Hotkeys: [t]=tp, [f]=fp, [u]=unknown, [s]=skip, [p]=replay, [q]=quit")
    print("Single-key input (no Enter). You can press keys during playback.")

    i = max(0, int(args.start))
    changed = False

    with cbreak_stdin():
        while i < len(events):
            ev = events[i]
            label = str(ev.get("label", "unknown")).lower()
            if args.only_unknown and label != "unknown":
                i += 1
                continue

            clip = Path(str(ev.get("clip_path", "")))
            print("\n" + "-" * 72)
            print(f"[{i+1}/{len(events)}] label={label} score={ev.get('score')} thr={ev.get('dynamic_threshold')}")
            print(f"clip: {clip}")
            print("press key now: t/f/u/s/p/q")

            player: subprocess.Popen[str] | None = None
            if clip.exists() and not args.no_play:
                player = subprocess.Popen(["afplay", str(clip)])
            elif not clip.exists():
                print("clip missing")

            while True:
                key = read_key_nonblocking(timeout_s=0.05)
                if key is None:
                    continue

                if key == "q":
                    if player and player.poll() is None:
                        player.terminate()
                    save_events(events_path, events)
                    print(f"\nSaved and quit: {events_path}")
                    return 0

                if key == "p":
                    if player and player.poll() is None:
                        player.terminate()
                    if clip.exists():
                        player = subprocess.Popen(["afplay", str(clip)])
                    else:
                        print("\nclip missing")
                    continue

                normalized = normalize_label(key)
                if normalized is None:
                    # skip current sample
                    if key in {"s", "\n", "\r", " "}:
                        if player and player.poll() is None:
                            player.terminate()
                        break
                    continue

                if player and player.poll() is None:
                    player.terminate()

                ev["label"] = normalized
                ev["labeled_at"] = int(time.time())
                changed = True
                break

            i += 1

    if changed:
        save_events(events_path, events)
        print(f"Saved labels: {events_path}")
    else:
        print("No label changes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
