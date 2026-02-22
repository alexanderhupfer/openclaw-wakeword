#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("\"").strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def classify_transcript(text: str) -> tuple[str, str]:
    t = normalize_text(text)

    if re.search(r"\bopen\s*-?\s*claw\b", t) or "openclaw" in t:
        return "tp", "contains_openclaw"

    toks = t.split()
    for i, tok in enumerate(toks):
        if tok == "open":
            span = toks[i + 1 : i + 4]
            if "claw" in span:
                return "tp", "open_near_claw"

    if "claw" in toks:
        return "fp", "claw_without_open"

    return "fp", "no_wakeword_match"


def extract_confidence(resp: Any) -> float | None:
    # Best-effort extraction across possible response payload shapes.
    try:
        data = resp.model_dump()
    except Exception:
        try:
            data = dict(resp)
        except Exception:
            return None

    # Word confidence path
    words = data.get("words")
    if isinstance(words, list) and words:
        vals = [float(w.get("confidence")) for w in words if isinstance(w, dict) and w.get("confidence") is not None]
        if vals:
            return float(sum(vals) / len(vals))

    # Segment avg_logprob path (Whisper-like)
    segs = data.get("segments")
    if isinstance(segs, list) and segs:
        vals = [float(s.get("avg_logprob")) for s in segs if isinstance(s, dict) and s.get("avg_logprob") is not None]
        if vals:
            import math

            probs = [max(0.0, min(1.0, math.exp(v))) for v in vals]
            return float(sum(probs) / len(probs))

    # Token logprobs path
    lps = data.get("logprobs")
    if isinstance(lps, list) and lps:
        vals = []
        for lp in lps:
            if isinstance(lp, dict):
                v = lp.get("logprob")
                if v is not None:
                    vals.append(float(v))
        if vals:
            import math

            probs = [max(0.0, min(1.0, math.exp(v))) for v in vals]
            return float(sum(probs) / len(probs))

    return None


def extract_text(resp: Any) -> str:
    txt = getattr(resp, "text", None)
    if isinstance(txt, str):
        return txt
    try:
        data = resp.model_dump()
        if isinstance(data.get("text"), str):
            return data["text"]
    except Exception:
        pass
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-label shadow trigger clips using OpenAI transcription")
    ap.add_argument("--shadow-dir", type=Path, default=Path("shadow_mode/jabra"))
    ap.add_argument("--model", type=str, default="gpt-4o-mini-transcribe-2025-12-15")
    ap.add_argument("--min-confidence", type=float, default=0.55)
    ap.add_argument("--only-unknown", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep-sec", type=float, default=0.0)
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    args = ap.parse_args()

    load_dotenv(args.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (or not found in .env)")

    events_path = args.shadow_dir.resolve() / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events log: {events_path}")

    lines = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]

    client = OpenAI()

    processed = 0
    updated = 0
    reviewed = 0

    for ev in events:
        cur = str(ev.get("label", "unknown")).lower()
        if args.only_unknown and cur != "unknown":
            continue
        if args.limit > 0 and processed >= args.limit:
            break

        clip = Path(str(ev.get("clip_path", "")))
        if not clip.exists():
            continue

        with clip.open("rb") as f:
            resp = client.audio.transcriptions.create(
                model=args.model,
                file=f,
                response_format="json",
                include=["logprobs"],
                prompt="The wakeword of interest is OpenClaw.",
                temperature=0,
            )

        text = extract_text(resp)
        conf = extract_confidence(resp)
        auto_label, reason = classify_transcript(text)

        final_label = auto_label
        if conf is not None and conf < float(args.min_confidence):
            final_label = "review"
            reviewed += 1

        ev["asr_model"] = args.model
        ev["asr_text"] = text
        ev["asr_confidence"] = conf
        ev["auto_label"] = auto_label
        ev["auto_label_reason"] = reason
        ev["auto_labeled_at"] = int(time.time())
        ev["auto_label_final"] = final_label

        if not args.dry_run:
            ev["label"] = final_label
            updated += 1

        processed += 1
        print(
            f"[{processed}] {clip.name} -> text={text!r} conf={conf} auto={auto_label} final={final_label} reason={reason}"
        )

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    if not args.dry_run:
        out = [json.dumps(obj, ensure_ascii=True) for obj in events]
        events_path.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"Saved {updated} updates -> {events_path}")
    else:
        print("Dry run; no file changes")

    print(f"Processed={processed} Updated={updated} Review={reviewed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
