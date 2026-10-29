"""python -m crab_rescue.run --seeds S [--dry-run] [--fake]

Runs S seeded rooms for each method (agent, blind, oracle) and writes
runs/<ts>/ep_<seed>/<method>/{plans.json,perception.json,metrics.json,trace.json,video.mp4,frames/},
plus ledger.jsonl, summary.json and report.md at the top.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .agent import run_agent
from .baselines import run_blind, run_oracle
from .budget import Budget
from .clients import DiskCache, FakePlanner, FakeVLM
from .config import ROLES, load_config
from .evaluator import episode_metrics
from .executor import SensorView, World
from .nebius import ChatTransport, NebiusPlanner, NebiusVLM, env_credentials
from .render import VideoWriter, render_world
from .router import Router
from .scene import generate_room

METHODS = ("agent", "blind", "oracle")


def dry_run_report(cfg: dict[str, Any], seeds: int, methods: tuple[str, ...]) -> str:
    caps, est, models = cfg["caps"], cfg["estimates"], cfg["models"]
    n = seeds if "agent" in methods else 0
    vlm_calls = n * caps["max_vlm_calls_per_episode"]
    plan_calls = n * caps["max_replans_per_episode"]
    rows = [f"DRY RUN: {seeds} seeds x methods {list(methods)}; no model call is made, $0 spent.",
            f"  worst-case VLM calls : {vlm_calls}  (cap {caps['max_vlm_calls_per_episode']}/episode)",
            f"  worst-case plan calls: {plan_calls}  (cap {caps['max_replans_per_episode']}/episode, "
            f"+ up to {caps['max_repairs']} repairs each)"]
    tin = vlm_calls * est["vlm_in"] + plan_calls * est["planner_in"]
    tout = vlm_calls * est["vlm_out"] + plan_calls * est["planner_out"]
    rows.append(f"  est. tokens          : {tin} in / {tout} out (assumptions in config.yaml: estimates)")
    unknown = [r for r, m in models.items() if m["price_in"] is None or m["price_out"] is None]
    if unknown:
        rows.append(f"  est. cost            : UNKNOWN - no price set for {unknown}; fill in config.yaml "
                    "from the Token Factory console (prices are never guessed)")
    else:
        cost = (vlm_calls * (est["vlm_in"] * models["vlm_primary"]["price_in"]
                             + est["vlm_out"] * models["vlm_primary"]["price_out"])
                + plan_calls * (est["planner_in"] * models["planner"]["price_in"]
                                + est["planner_out"] * models["planner"]["price_out"])) / 1e6
        rows.append(f"  est. cost (upper)    : ${cost:.4f} vs cap ${caps['max_total_usd']}")
    return "\n".join(rows)


def run_episode(seed: int, method: str, cfg: dict[str, Any], out: Path, budget: Budget, router: Router,
                fake: bool, video: bool, clients: tuple[Any, Any] | None = None) -> dict[str, Any]:
    gw, gh = cfg["sim"]["grid"]
    room = generate_room(seed, gw, gh, cfg["sim"]["cell_m"])
    world = World(room, cfg, seed)
    edir = out / f"ep_{seed}" / method
    edir.mkdir(parents=True, exist_ok=True)
    vw = None
    if video:
        scale = 45
        vw = VideoWriter(edir / "video.mp4", (room.w * scale, room.h * scale), fps=10)
        every = max(1, int(0.25 / cfg["sim"]["dt"]))
        counter = {"n": 0}

        def rec(w: World) -> None:
            counter["n"] += 1
            if counter["n"] % every:
                return
            hud = (f"{method}  seed {seed}  t={w.t:5.1f}s  fire cells={w.fire.area}"
                   f"{'  CARRYING' if w.carried else ''}")
            vw.add(render_world(room, w.fire.burning, tuple(w.pose), tuple(w.fig), scale=scale,
                                trail=w.trail, hud=hud))

        world.observers.append(rec)

    budget.new_episode(f"ep{seed}/{method}")
    router.new_episode()
    log: dict[str, Any] | None = None
    if method == "agent":
        vlm, planner = clients if clients else (FakeVLM((gw, gh), seed=seed), FakePlanner(seed=seed))
        log = run_agent(SensorView(world), vlm, planner, budget, router, cfg, frame_dir=edir / "frames")
    elif method == "blind":
        run_blind(world, cfg)
    else:
        run_oracle(world, cfg)
    if vw:
        vw.add(render_world(room, world.fire.burning, tuple(world.pose), tuple(world.fig), scale=45,
                            trail=world.trail, hud=f"{method} seed {seed}  RESULT: {world.outcome}"))
        vw.close()
    metrics = episode_metrics(world, method, seed, budget if log else None, log,
                              uses_ground_truth=method != "agent")
    metrics["fake_models"] = fake
    (edir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if log:
        (edir / "plans.json").write_text(json.dumps(log["plans"], indent=2))
        (edir / "perception.json").write_text(json.dumps(log["perceptions"], indent=2))
        (edir / "trace.json").write_text(json.dumps(log["events"], indent=2))
    # scene facts for the site (the layout of the room, not used by the agent)
    (edir / "scene.json").write_text(json.dumps({
        "seed": seed, "grid": room.grid.tolist(), "cell_m": room.cell, "start": room.start,
        "figurine": room.figurine, "exit_xy": room.exit_xy, "fire_origin": room.fire_origin}))
    return metrics


def write_report(out: Path, rows: list[dict[str, Any]], cfg: dict[str, Any], fake: bool, seeds: int) -> None:
    def agg(method: str) -> dict[str, Any]:
        r = [x for x in rows if x["method"] == method]
        if not r:
            return {}
        resc = [x for x in r if x["rescued"]]
        mean = lambda k, src=r: round(float(np.mean([x[k] for x in src])), 2) if src else None  # noqa: E731
        d = {"n": len(r), "rescued": len(resc), "rescue_rate": round(len(resc) / len(r), 3),
             "time_to_rescue_s": mean("time_to_rescue_s", resc), "fire_exposure_s": mean("fire_exposure_s"),
             "fire_entries": mean("fire_entries"), "collisions": mean("collisions")}
        if method == "agent":
            d.update(replans=mean("replans"), vlm_calls=mean("vlm_calls"), invalid_plans=mean("invalid_plans"),
                     tokens_in=mean("tokens_in"), tokens_out=mean("tokens_out"), cost_usd=mean("cost_usd"),
                     fire_iou=round(float(np.nanmean([x["perception"]["fire_iou"] or np.nan for x in r])), 3))
        return d

    summary = {"methods": {m: agg(m) for m in METHODS}, "episodes": rows, "fake_models": fake,
               "seeds": seeds, "config_caps": cfg["caps"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    L = ["# Crab Rescue run report", "",
         f"- seeds: {seeds}; models: **{'FAKE (deterministic stubs, $0)' if fake else 'real'}**",
         "- executor: kinematic stand-in for the locomotion policy; fire = simulated hazard grid; "
         "carry = weld-on-contact. See the limitations section of the site.", "",
         "| method | rescued | rate | mean time (s) | fire exposure (s) | fire entries | collisions |",
         "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        a = summary["methods"][m]
        if a:
            L.append(f"| {m} | {a['rescued']}/{a['n']} | {a['rescue_rate']} | {a['time_to_rescue_s']} | "
                     f"{a['fire_exposure_s']} | {a['fire_entries']} | {a['collisions']} |")
    ag = summary["methods"]["agent"]
    if ag:
        L += ["", f"Agent: replans {ag['replans']}, VLM calls {ag['vlm_calls']}, invalid plans "
                  f"{ag['invalid_plans']}, tokens {ag['tokens_in']}/{ag['tokens_out']}, cost "
                  f"${ag['cost_usd']}, mean fire IoU {ag['fire_iou']}"]
    lost = [x for x in rows if x["method"] == "agent" and not x["rescued"]]
    L += ["", "## Seeds where the agent failed", ""]
    L += [f"- seed {x['seed']}: {x['outcome']}" for x in lost] or ["- none"]
    (out / "report.md").write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crab_rescue.run")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="deterministic fake VLM/planner, $0")
    ap.add_argument("--dry-run", action="store_true", help="print planned calls/cost, spend nothing")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    methods = tuple(m for m in args.methods.split(",") if m)
    root = Path(args.out)
    marker = root / ".dryrun_done"
    if not args.fake and not args.dry_run and not marker.exists():
        print("First real invocation defaults to --dry-run (nothing is spent). Re-run to proceed.\n")
        args.dry_run = True
    if args.dry_run:
        print(dry_run_report(cfg, args.seeds, methods))
        if not args.fake:
            root.mkdir(parents=True, exist_ok=True)
            marker.touch()
        return 0
    clients = None
    if not args.fake:
        missing = [r for r in ROLES if any(cfg["models"][r].get(k) is None for k in ("id", "price_in", "price_out"))]
        if missing:
            print(f"config.yaml models.{missing} need id, price_in and price_out.\n"
                  "Ids come only from `python -m crab_rescue.discover --yes`; prices only from the "
                  "Token Factory console. Nothing was sent.", file=sys.stderr)
            return 2
        key, base = env_credentials()
        transport = ChatTransport(base, key)
        ids = {r: cfg["models"][r]["id"] for r in ROLES}
        cache = DiskCache(root / ".cache")
        mt = cfg["caps"]["max_tokens_per_call"]
        clients = (NebiusVLM(transport, ids, mt, cache), NebiusPlanner(transport, ids, mt, cache))

    out = root / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(cfg, ledger_path=out / "ledger.jsonl", fake=args.fake)
    router = Router(cfg["caps"]["max_escalations"], cfg["agent"]["low_confidence"])
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    rows = []
    try:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for method in methods:
                t0 = time.time()
                m = run_episode(seed, method, cfg, out, budget, router, args.fake, not args.no_video, clients)
                rows.append(m)
                print(f"seed {seed:3d} {method:6s} {m['outcome']:24s} sim {m['sim_time_s']:6.1f}s "
                      f"wall {time.time() - t0:5.1f}s")
    except KeyboardInterrupt:
        print("interrupted: saving what exists")
    write_report(out, rows, cfg, args.fake, args.seeds)
    print(f"\nrun saved to {out}\n  total spend ${budget.total_usd:.4f} ({budget.total_calls} calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
