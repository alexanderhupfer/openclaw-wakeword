#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize shadow-mode trigger logs")
    ap.add_argument("--shadow-dir", type=Path, default=Path("shadow_mode/jabra"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    events_path = args.shadow_dir.resolve() / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events log: {events_path}")

    rows = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not rows:
        raise RuntimeError("No parsable events")

    ts = np.asarray([float(r.get("trigger_time", 0.0)) for r in rows], dtype=np.float64)
    scores = np.asarray([float(r.get("score", 0.0)) for r in rows], dtype=np.float32)
    dyn = np.asarray([float(r.get("dynamic_threshold", 0.0)) for r in rows], dtype=np.float32)

    tmin = float(np.min(ts))
    tmax = float(np.max(ts))
    elapsed_h = max((tmax - tmin) / 3600.0, 1e-6)
    triggers = int(len(rows))

    labels = [str(r.get("label", "unknown")).lower() for r in rows]
    fp = sum(1 for l in labels if l in {"fp", "false_positive", "negative"})
    tp = sum(1 for l in labels if l in {"tp", "true_positive", "positive"})
    unk = triggers - fp - tp

    report = {
        "events_path": str(events_path),
        "trigger_count": triggers,
        "window_hours": elapsed_h,
        "triggers_per_hour": float(triggers / elapsed_h),
        "score_p50": float(np.percentile(scores, 50)),
        "score_p90": float(np.percentile(scores, 90)),
        "score_p99": float(np.percentile(scores, 99)),
        "dynamic_threshold_p50": float(np.percentile(dyn, 50)),
        "labels": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "unknown": int(unk),
        },
    }

    out = args.out.resolve() if args.out else (args.shadow_dir.resolve() / "shadow_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
