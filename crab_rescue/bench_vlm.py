"""python -m crab_rescue.bench_vlm [--dry-run] [--fake] [--frames N] [--out DIR]

Sends the same N ground-truthed frames through each candidate VLM and reports localisation
accuracy (fire IoU, figurine and exit error), latency and cost per call.
Real candidates are wired in at M4; until then --fake runs the two stand-in tiers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .clients import FakeVLM
from .config import load_config
from .evaluator import perception_accuracy
from .executor import SensorView, World
from .scene import generate_room


def make_frames(n: int, cfg: dict[str, Any]) -> list[tuple[World, bytes, int]]:
    out = []
    gw, gh = cfg["sim"]["grid"]
    for i in range(n):
        room = generate_room(1000 + i, gw, gh, cfg["sim"]["cell_m"])
        world = World(room, cfg, 1000 + i)
        world.fire.advance_to((i % 6) * 8.0)          # early fire ... large fire
        png, fid = SensorView(world).frame()
        out.append((world, png, fid))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crab_rescue.bench_vlm")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--out", default=None, help="run directory to write bench_vlm.json into")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    est = cfg["estimates"]
    cands = ["vlm_primary", "vlm_cheap"]
    if a.dry_run or not a.fake:
        calls = a.frames * len(cands)
        print(f"DRY RUN: {a.frames} frames x {len(cands)} candidate roles = {calls} VLM calls")
        print(f"  est. tokens: {calls * est['vlm_in']} in / {calls * est['vlm_out']} out")
        unknown = [r for r in cands if cfg["models"][r]["price_in"] is None]
        print("  est. cost: UNKNOWN - set ids and prices in config.yaml (never guessed)" if unknown else "  est. cost: see config prices")
        if not a.fake:
            print("Real VLM clients arrive at M4; use --fake for the stand-in tiers.")
        return 0

    frames = make_frames(a.frames, cfg)
    models = []
    for role in cands:
        vlm = FakeVLM(tuple(cfg["sim"]["grid"]), seed=7)
        ents, lat = [], []
        for world, png, fid in frames:
            t0 = time.time()
            res = vlm.perceive(role, png, {})
            lat.append(time.time() - t0)
            ents.append({"valid": True, "frame_id": fid, "parsed": json.loads(res.text), "role": role, "_w": world})
        accs = [perception_accuracy(w, [{k: v for k, v in e.items() if k != "_w"}])
                for (w, _, _), e in zip(frames, ents)]
        pick = lambda k: round(float(np.nanmean([x[k] if x[k] is not None else np.nan for x in accs])), 4)  # noqa: E731
        models.append({"name": f"stand-in {role} (fake)", "fire_iou": pick("fire_iou"),
                       "figurine_err_m": pick("figurine_err_m"), "exit_err_m": pick("exit_err_m"),
                       "latency_s": round(float(np.mean(lat)), 4), "cost_per_call": 0.0})
    result = {"note": f"STAND-IN models on {a.frames} synthetic frames; real candidates are benchmarked at M4.",
              "frames": a.frames, "models": models}
    print(json.dumps(result, indent=2))
    if a.out:
        (Path(a.out) / "bench_vlm.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
