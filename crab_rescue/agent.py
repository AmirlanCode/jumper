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
from .clients import ModelCallError, PlannerClient, VLMClient
from .follower import WaypointFollower
from .image_perception import cross_check_perception, refine_from_frame
from .pathfinding import rescue_fallback
from .router import Router
from .schemas import Perception, Plan, parse_model_perception, parse_plan
from .validator import PerceivedMap, build_map, validate_plan


def run_agent(sensor, vlm: VLMClient, planner: PlannerClient, budget: Budget, router: Router,
              cfg: dict[str, Any], frame_dir: Path | None = None,
              log: dict[str, Any] | None = None) -> dict[str, Any]:
    W, H, gw, gh, cell = sensor.bounds()
    a, lim = cfg["agent"], cfg["limits"]
    follower = WaypointFollower(tuple(lim["cmd_max"]))
    log = log if log is not None else {}
    for k in ("perceptions", "plans", "events"):
        log.setdefault(k, [])

    perception: Perception | None = None
    pmap: PerceivedMap | None = None
    last_valid: Plan | None = None
    last_vlm_t = -1e9
    last_map_t = -1e9
    last_plan_t = -1e9
    fire_hist: list[tuple[float, int]] = []
    fire_event = False
    fatal_model_error = False
    stuck_ref = (sensor.pose()[0], sensor.pose()[1], 0.0)
    pending_reason: str | None = "initial"

    def event(kind: str, **kw: Any) -> None:
        log["events"].append({"t": round(sensor.time, 2), "type": kind, **kw})

    # ------------------------------------------------------------------ perception
    def do_perceive() -> None:
        nonlocal perception, pmap, last_vlm_t, last_map_t, fire_event, pending_reason, fatal_model_error
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
            sem = sensor.semantic_png() if hasattr(sensor, "semantic_png") else None
            meta = {"t": t, "repair": repair, "grid": (gw, gh)}
            if sem is not None:
                meta["semantic_png"] = sem     # only the offline stand-in reads it
            t0 = time.time()
            try:
                res = vlm.perceive(role, png, meta)
            except ModelCallError as e:
                event("vlm_error", role=role, error=str(e)[:200])
                fatal_model_error = True
                return
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
                new = parse_model_perception(res.text)
            except (ValueError, ValidationError) as e:
                entry.update(valid=False, error=str(e)[:300])
                log["perceptions"].append(entry)
                repair = f"previous reply was invalid: {str(e)[:200]}"
                continue
            if sem is not None:
                # the model's targets are used when the segmentation channel confirms them
                raw_model = new.model_dump()
                new, grounding = cross_check_perception(new, sem, gw, gh)
                entry["grounding"] = grounding
                entry["model_reported"] = {"figurine": raw_model.get("figurine"), "exit": raw_model.get("exit")}
            else:
                new = refine_from_frame(new, png, gw, gh)
            entry.update(valid=True, parsed=new.model_dump())
            log["perceptions"].append(entry)
            perception, pmap = new, build_map(new, gw, gh, cell)
            last_map_t = t
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

    def refresh_camera_map() -> None:
        """Update the dynamic hazard grid from camera pixels without another API call."""
        nonlocal perception, pmap, last_map_t, fire_event, pending_reason
        if perception is None:
            return
        t = sensor.time
        png, fid = sensor.semantic_frame() if hasattr(sensor, "semantic_frame") else sensor.frame()
        old_area = int(pmap.fire.sum()) if pmap is not None else 0
        perception = refine_from_frame(perception, png, gw, gh)
        pmap = build_map(perception, gw, gh, cell)
        area = int(pmap.fire.sum())
        fire_event = area - old_area >= a["fire_spread_event_cells"]
        last_map_t = t
        event("camera_map_updated", frame_id=fid, fire_cells=area)
        if follower.plan is not None and not follower.finished:
            rest = Plan(waypoints=follower.plan.waypoints[follower.idx:], rationale="rest")
            carried_margin = lim["fire_margin_m"] + (
                cfg["sim"]["carry_offset_m"] if sensor.carrying else 0.0)
            bad = validate_plan(rest, pmap, sensor.pose()[:2], lim["wall_margin_m"],
                                carried_margin, lim["plan_speed_max"])
            if bad:
                pending_reason = "path_blocked"
                event("path_blocked", reasons=bad[:2])

    # ------------------------------------------------------------------ planning
    def trend() -> float:
        if len(fire_hist) < 2:
            return 0.0
        (t0, a0), (t1, a1) = fire_hist[-2], fire_hist[-1]
        return (a1 - a0) / max(t1 - t0, 1e-6)

    def safety_fallback(reason: str, pose: tuple[float, float, float]) -> Plan | None:
        if pmap is None:
            return None
        carried_fire_margin = lim["fire_margin_m"] + (
            cfg["sim"]["carry_offset_m"] if sensor.carrying else 0.0)
        plan = rescue_fallback(pose[:2], sensor.carrying, pmap, lim["wall_margin_m"],
                               carried_fire_margin, lim["plan_speed_max"],
                               cfg["sim"]["carry_offset_m"])
        if plan is None:
            event("safety_fallback_unavailable", reason=reason)
            return None
        bad = validate_plan(plan, pmap, pose[:2], lim["wall_margin_m"], carried_fire_margin,
                            lim["plan_speed_max"])
        if bad:
            event("safety_fallback_rejected", reason=reason, reasons=bad[:4])
            return None
        log["plans"].append({"t": round(sensor.time, 2), "reason": reason,
                             "role": "safety_astar", "accepted": True,
                             "repaired_model_route": True, "plan": plan.model_dump()})
        event("safety_fallback_accepted", reason=reason, waypoints=len(plan.waypoints))
        return plan

    def request_plan(reason: str) -> Plan | None:
        nonlocal last_valid, last_plan_t, fatal_model_error
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
        grid = [["." for _ in range(gw)] for _ in range(gh)]
        for y in range(gh):
            for x in range(gw):
                if pmap.walls[y, x]:
                    grid[y][x] = "#"
                elif pmap.fire[y, x]:
                    grid[y][x] = "F"
        def mark(point: tuple[float, float] | None, marker: str) -> None:
            if point is None:
                return
            x, y = int(point[0] / cell), int(point[1] / cell)
            if 0 <= x < gw and 0 <= y < gh and grid[y][x] == ".":
                grid[y][x] = marker
        mark(pmap.figurine, "V")
        mark(pmap.exit_xy, "E")
        mark(pose[:2], "R")
        map_ascii = "\n".join("".join(row) for row in grid)
        ctx = {"pose": list(pose), "t": round(t, 2), "carrying": sensor.carrying,
               "perception": perception.model_dump(), "grid": [gw, gh, cell],
               "figurine": list(pmap.figurine) if pmap.figurine else None,
               "exit": list(pmap.exit_xy) if pmap.exit_xy else None, "map_ascii": map_ascii,
               "fire_cells_per_s": round(trend(), 3), "time_pressure": pressure, "reason": reason}
        failed = False
        blocked = False
        api_error = False
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
                    blocked = True
                    break
                t0 = time.time()
                try:
                    res = planner.plan(role, ctx, feedback)
                except ModelCallError as e:
                    event("plan_error", role=role, error=str(e)[:200])
                    api_error = True
                    break
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
                actions = [wp.action for wp in plan.waypoints]
                semantic_errors = []
                if not sensor.carrying and ("pickup" not in actions or "release" not in actions
                                             or actions.index("pickup") > actions.index("release")):
                    semantic_errors.append("plan must pick up the figurine before releasing at the exit")
                if sensor.carrying and "release" not in actions:
                    semantic_errors.append("carrying plan must release at the exit")
                for wp in plan.waypoints:
                    if wp.action == "pickup" and pmap.figurine and math.dist(
                            (wp.x, wp.y), pmap.figurine) > 0.4:
                        semantic_errors.append("pickup waypoint must be within grasp range of figurine")
                    if wp.action == "release" and pmap.exit_xy and math.dist(
                            (wp.x, wp.y), pmap.exit_xy) > 0.5:
                        semantic_errors.append("release waypoint must be next to the perceived exit")
                if semantic_errors:
                    feedback = semantic_errors[:4]
                    entry.update(accepted=False, rejected_by="task_contract", reasons=feedback)
                    log["plans"].append(entry)
                    continue
                carried_fire_margin = lim["fire_margin_m"] + (
                    cfg["sim"]["carry_offset_m"] if sensor.carrying else 0.0)
                bad = validate_plan(plan, pmap, pose[:2], lim["wall_margin_m"],
                                    carried_fire_margin, lim["plan_speed_max"])
                if bad:
                    feedback = bad[:4]
                    entry.update(accepted=False, rejected_by="validator", reasons=bad[:4])
                    log["plans"].append(entry)
                    continue
                entry.update(accepted=True, plan=plan.model_dump())
                log["plans"].append(entry)
                last_valid = plan
                return plan
            if blocked or api_error:
                break
            failed = True
        if blocked or api_error or failed:
            safe = safety_fallback(reason, pose)
            if safe is not None:
                last_valid = safe
                return safe
        event("plan_failed", reason=reason, fallback="last_valid" if last_valid else "hold")
        return None

    # ------------------------------------------------------------------ main loop
    while not sensor.done:
        t = sensor.time
        if (perception is None and last_vlm_t < 0) or t - last_vlm_t >= a["vlm_period_s"]:
            do_perceive()
        elif perception is not None and t - last_map_t >= cfg["sim"]["fire"]["step_s"]:
            refresh_camera_map()
        if fatal_model_error:
            event("episode_aborted", reason="model_api_error")
            break
        if pending_reason is None and follower.finished and t - last_plan_t >= a["replan_period_s"]:
            pending_reason = "plan_exhausted"
        if pending_reason is not None and (
                perception is not None or t - last_plan_t >= a["replan_period_s"]):
            plan = request_plan(pending_reason)
            if plan is not None:
                follower.set_plan(plan)
                event("plan_accepted", reason=pending_reason, waypoints=len(plan.waypoints))
            pending_reason = None
        if fatal_model_error:
            event("episode_aborted", reason="model_api_error")
            break
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
