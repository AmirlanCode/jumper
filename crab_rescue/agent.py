"""The VLM + LLM rescue agent.

It sees the world ONLY through `SensorView` (camera frame, odometry, velocity command,
grasp/release). This module must never touch the simulator's ground truth; the isolation test in
`tests/test_pipeline.py` enforces that.

Loop: look (VLM) -> plan (LLM) -> validate -> follow -> monitor -> replan.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .budget import Budget, BudgetExceeded, PriceUnknown
from .clients import PlannerClient, VLMClient
from .follower import WaypointFollower
from .router import Router
from .schemas import Perception, Plan, parse_perception, parse_plan
from .validator import PerceivedMap, build_map, validate_plan


def run_agent(sensor, vlm: VLMClient, planner: PlannerClient, budget: Budget, router: Router,
              cfg: dict[str, Any], frame_dir: Path | None = None) -> dict[str, Any]:
    W, H, gw, gh, cell = sensor.bounds()
    a, lim = cfg["agent"], cfg["limits"]
    follower = WaypointFollower(tuple(lim["cmd_max"]))
    log: dict[str, Any] = {"perceptions": [], "plans": [], "events": []}

    perception: Perception | None = None
    pmap: PerceivedMap | None = None
    last_valid: Plan | None = None
    last_vlm_t = -1e9
    last_plan_t = -1e9
    fire_hist: list[tuple[float, int]] = []
    fire_event = False
    stuck_ref = (sensor.pose()[0], sensor.pose()[1], 0.0)
    pending_reason: str | None = "initial"

    def event(kind: str, **kw: Any) -> None:
        log["events"].append({"t": round(sensor.time, 2), "type": kind, **kw})

    # ------------------------------------------------------------------ perception
    def do_perceive() -> None:
        nonlocal perception, pmap, last_vlm_t, fire_event, pending_reason
        t = sensor.time
        last_vlm_t = t
        role = router.pick_vlm(first=perception is None, fire_event=fire_event,
                               confidence=perception.confidence if perception else None)
        repair = None
        for attempt in range(2):     # one repair retry on invalid JSON
            try:
                budget.check("vlm", role, est_in=1500, est_out=300)
            except (BudgetExceeded, PriceUnknown) as e:
                event("vlm_blocked", reason=str(e))
                return
            png, fid = sensor.frame()
            t0 = time.time()
            res = vlm.perceive(role, png, {"t": t, "repair": repair, "grid": (gw, gh)})
            budget.record("vlm", role, res.model, res.tokens_in, res.tokens_out,
                          res.latency_s or (time.time() - t0), t_sim=round(t, 2), attempt=attempt)
            fname = None
            if frame_dir is not None:
                frame_dir.mkdir(parents=True, exist_ok=True)
                fname = f"f{fid:04d}.png"
                (frame_dir / fname).write_bytes(png)
            entry = {"t": round(t, 2), "role": role, "model": res.model, "frame_id": fid,
                     "frame_file": fname, "raw": res.text, "attempt": attempt}
            try:
                new = parse_perception(res.text)
            except (ValueError, ValidationError) as e:
                entry.update(valid=False, error=str(e)[:300])
                log["perceptions"].append(entry)
                repair = f"previous reply was invalid: {str(e)[:200]}"
                continue
            entry.update(valid=True, parsed=new.model_dump())
            log["perceptions"].append(entry)
            perception, pmap = new, build_map(new, gw, gh, cell)
            area = int(pmap.fire.sum())
            prev = fire_hist[-1][1] if fire_hist else area
            fire_hist.append((t, area))
            fire_event = (area - prev) >= a["fire_spread_event_cells"]
            if fire_event:
                event("fire_spread", cells_added=area - prev, area=area)
            # does the plan we are following still look safe on the new map?
            if follower.plan is not None and not follower.finished:
                rest = Plan(waypoints=follower.plan.waypoints[follower.idx:], rationale="rest")
                bad = validate_plan(rest, pmap, sensor.pose()[:2], lim["wall_margin_m"],
                                    lim["fire_margin_m"], lim["plan_speed_max"])
                if bad:
                    pending_reason = "path_blocked"
                    event("path_blocked", reasons=bad[:2])
            return

    # ------------------------------------------------------------------ planning
    def trend() -> float:
        if len(fire_hist) < 2:
            return 0.0
        (t0, a0), (t1, a1) = fire_hist[-2], fire_hist[-1]
        return (a1 - a0) / max(t1 - t0, 1e-6)

    def request_plan(reason: str) -> Plan | None:
        nonlocal last_valid, last_plan_t
        t = sensor.time
        last_plan_t = t
        if perception is None or pmap is None:
            event("plan_skipped", reason="no perception yet")
            return None
        pose = sensor.pose()
        near_fire = bool(pmap.fire.any()) and min(
            math.hypot((x + 0.5) * cell - pose[0], (y + 0.5) * cell - pose[1])
            for y, x in zip(*pmap.fire.nonzero())) < 2.0
        pressure = fire_event or near_fire
        ctx = {"pose": list(pose), "t": round(t, 2), "carrying": sensor.carrying,
               "perception": perception.model_dump(), "grid": [gw, gh, cell],
               "figurine": list(pmap.figurine) if pmap.figurine else None,
               "fire_cells_per_s": round(trend(), 3), "time_pressure": pressure, "reason": reason}
        failed = False
        for rnd in range(2):          # normal round, then (maybe) one escalation round
            role = router.pick_planner(time_pressure=pressure, failed_before=failed)
            if rnd == 1 and role != "planner_escalation":
                break
            feedback: list[str] = []
            for att in range(1 + cfg["caps"]["max_repairs"]):
                kind = "plan" if (rnd == 0 and att == 0) else "repair"
                try:
                    budget.check(kind, role, est_in=len(json.dumps(ctx)) // 4, est_out=400)
                except (BudgetExceeded, PriceUnknown) as e:
                    event("plan_blocked", reason=str(e), kept_last_valid=last_valid is not None)
                    return None
                t0 = time.time()
                res = planner.plan(role, ctx, feedback)
                budget.record(kind, role, res.model, res.tokens_in, res.tokens_out,
                              res.latency_s or (time.time() - t0), t_sim=round(t, 2), reason=reason)
                entry = {"t": round(t, 2), "reason": reason, "role": role, "model": res.model,
                         "round": rnd, "attempt": att, "raw": res.text}
                try:
                    plan = parse_plan(res.text, (W, H))
                except (ValueError, ValidationError) as e:
                    feedback = [f"your reply was not valid plan JSON: {str(e)[:240]}"]
                    entry.update(accepted=False, rejected_by="schema", reasons=feedback)
                    log["plans"].append(entry)
                    continue
                bad = validate_plan(plan, pmap, pose[:2], lim["wall_margin_m"], lim["fire_margin_m"],
                                    lim["plan_speed_max"])
                if bad:
                    feedback = bad[:4]
                    entry.update(accepted=False, rejected_by="validator", reasons=bad[:4])
                    log["plans"].append(entry)
                    continue
                entry.update(accepted=True, plan=plan.model_dump())
                log["plans"].append(entry)
                last_valid = plan
                return plan
            failed = True
        event("plan_failed", reason=reason, fallback="last_valid" if last_valid else "hold")
        return None

    # ------------------------------------------------------------------ main loop
    while not sensor.done:
        t = sensor.time
        if perception is None or t - last_vlm_t >= a["vlm_period_s"]:
            do_perceive()
        if pending_reason is None:
            if follower.finished and t - last_plan_t >= 1.0:
                pending_reason = "plan_exhausted"
            elif t - last_plan_t >= a["replan_period_s"]:
                pending_reason = "periodic"
        if pending_reason is not None and (perception is not None or t - last_plan_t >= 1.0):
            plan = request_plan(pending_reason)
            if plan is not None:
                follower.set_plan(plan)
                event("plan_accepted", reason=pending_reason, waypoints=len(plan.waypoints))
            pending_reason = None
        cmd, action = follower.command(sensor.pose())
        if action == "pickup":
            ok = sensor.grasp()
            event("pickup", ok=ok)
            if not ok:
                pending_reason = "pickup_failed"
        elif action == "release":
            sensor.release()
            event("release")
        # stuck monitor: commanding motion for 3 s without moving
        px, py, t_ref = stuck_ref
        if t - t_ref >= 3.0:
            moved = math.hypot(sensor.pose()[0] - px, sensor.pose()[1] - py)
            if moved < 0.1 and not follower.finished:
                event("stuck", moved=round(moved, 3))
                pending_reason = "stuck"
            stuck_ref = (sensor.pose()[0], sensor.pose()[1], t)
        sensor.actuate(cmd)
    event("episode_end")
    return log
