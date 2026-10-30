"""The model drives: NVIDIA Nemotron-3.5-Super-VL moves the crab one step at a time.

    python -m crab_rescue.model_drive --seed 191          # needs NEBIUS_API_KEY in the environment

Each turn the model sees the overhead camera frame and returns ONE command (move_to <= 1.5 m,
pickup, release). Code only executes it (turn + straight line) and refuses a step that would
cross a wall or fire. No path planner runs, and the figurine's position is never given: the
model finds it in the image. This produced the recorded rescue on seed 191 (41.0 s, 32 calls,
$0.043, 2026-10-30). It reuses mission3d's scene, logging and site, replacing only the agent.
"""
# ruff: noqa: E702  (the step loop is kept as compact as when it drove the recorded run)
import base64
import json
import math
import time
import crab_rescue.mission3d as M
from crab_rescue.image_perception import refine_from_frame
from crab_rescue.nebius import _result, ApiError
from crab_rescue.schemas import Perception, Plan, Waypoint, extract_json
from crab_rescue.validator import build_map, validate_plan
from crab_rescue.prompts import vlm_user_text
from crab_rescue.schemas import parse_perception

MODEL = "nvidia/Nemotron-3.5-Super-VL"
MAX_CALLS, STEP_MAX, SPEED = 40, 1.55, 0.45  # 1.5 m told to the model; 5 cm slack for rounding

_load = M.load_config
def load_config():
    cfg = _load()
    for r in ("vlm_primary", "vlm_cheap", "planner", "planner_fast", "planner_escalation"):
        cfg["models"][r]["id"] = MODEL
    cfg["sim"]["fire"]["p_spread"] = 0.0          # the fire stays as it starts
    cfg["sim"]["max_episode_s"] = 600.0
    return cfg
M.load_config = load_config

class Budget(M.Budget):
    def __init__(self, cfg, *a, **k):
        cfg["caps"].update(max_replans_per_episode=500, max_total_usd=0.50)
        super().__init__(cfg, *a, **k)
M.Budget = Budget

SYSTEM = """You drive a small crab robot, one step at a time, in a room seen by an overhead camera.
Image: the whole 10 m x 8 m room seen from above. Image fraction (fx, fy) = metres (10*fx, 8*fy); x right, y down, origin top-left.
In the image: dark blue-grey blocks = walls, brown boxes = crates, orange flames = FIRE, green doorway + green floor on the LEFT wall = EXIT / safe zone, small blue human-shaped doll with a white head lying on the floor = the FIGURINE, the red crab = the robot (you).
You also get `map_ascii` from the same camera (`#` wall or crate, `F` fire, `.` free, `R` you, `E` exit; one character = 0.5 m; rows/columns numbered; cell (col,row) centre = ((col+0.5)*0.5, (row+0.5)*0.5) m), `ranges_m` (free distance in 8 directions from you, like a lidar), your `pose`, and your last steps with their results.
`my_vision` is where YOU located the figurine and the exit in this frame a moment ago (metres); trust it and head for it.
Walls can only be passed through gaps (`.` cells in a wall's column or row). Plan a few steps ahead: if the figurine is behind a wall, first go to the gap.

Reply with ONE JSON object and nothing else:
{"figurine_seen": {"x": metres, "y": metres} or null, "action": "move_to" | "pickup" | "release", "x": metres, "y": metres, "reason": "one short sentence"}
- move_to: walk in a straight line toward (x, y); a target farther than 1.5 m is walked toward for 1.5 m this step. The straight line must stay on free cells, at least 0.2 m from walls and away from fire. A refused step costs a turn.
- pickup: grab the figurine; it works only when you are within 0.35 m of it.
- release: put the figurine down; do it when you stand on the green EXIT cells."""

DIRS = [("E", 0), ("SE", 45), ("S", 90), ("SW", 135), ("W", 180), ("NW", 225), ("N", 270), ("NE", 315)]

def numbered(grid):
    w = len(grid[0])
    head = ["    " + "".join(str(c // 10) for c in range(w)), "    " + "".join(str(c % 10) for c in range(w))]
    return "\n".join(head + [f"{r:3d} " + "".join(row) for r, row in enumerate(grid)])

def drive(sensor, vlm, planner, budget, router, cfg, frame_dir=None, log=None):
    log = log if log is not None else {}
    for k in ("perceptions", "plans", "events"):
        log.setdefault(k, [])
    W, H, gw, gh, cell = sensor.bounds()
    lim = cfg["limits"]
    hist = []
    def event(kind, **kw):
        log["events"].append({"t": round(sensor.time, 2), "type": kind, **kw})

    for n in range(MAX_CALLS):
        if sensor.done:
            break
        sem = sensor.semantic_png()
        pmap = build_map(refine_from_frame(Perception(confidence=0.5), sem, gw, gh), gw, gh, cell)
        x, y, th = sensor.pose()
        grid = [["#" if pmap.walls[r, c] else ("F" if pmap.fire[r, c] else ".") for c in range(gw)] for r in range(gh)]
        if pmap.exit_xy:
            for r in range(gh):
                for c in range(gw):
                    if grid[r][c] == "." and math.hypot((c + .5) * cell - pmap.exit_xy[0], (r + .5) * cell - pmap.exit_xy[1]) < 0.6:
                        grid[r][c] = "E"
        grid[int(y / cell)][int(x / cell)] = "R"
        def rng(deg):
            a = math.radians(deg); d = 0.0
            while d < 4.0:
                d += 0.1; cx, cy = int((x + d * math.cos(a)) / cell), int((y + d * math.sin(a)) / cell)
                if not (0 <= cx < gw and 0 <= cy < gh) or pmap.walls[cy, cx] or pmap.fire[cy, cx]:
                    return round(d - 0.1, 1)
            return 4.0
        carrying = sensor.carrying
        goal = ("Carry the figurine to the green EXIT on the left wall and release it there." if carrying
                else "Find the figurine in the image, walk to it and pick it up.")
        ctx = {"goal": goal, "pose": {"x": round(x, 2), "y": round(y, 2), "heading_deg": round(math.degrees(th) % 360)},
               "carrying": carrying, "ranges_m": {k: rng(d) for k, d in DIRS}, "map_ascii": numbered(grid),
               "last_steps": hist[-6:]}
        png, fid = sensor.frame()
        if frame_dir is not None:
            frame_dir.mkdir(parents=True, exist_ok=True)
            (frame_dir / f"step{n:03d}.png").write_bytes(png)
        # stage 1, LOOK: the same model with its vision-only prompt (it found the figurine 3/3 this
        # way on seed 191, but 0/40 when looking and driving were asked in one call)
        b64 = "data:image/png;base64," + base64.b64encode(png).decode()
        t0 = time.time()
        try:
            look = _result(vlm.t.chat(MODEL, [{"role": "system", "content": M.VLM_SYSTEM_3D}, {"role": "user", "content": [
                {"type": "text", "text": vlm_user_text({})}, {"type": "image_url", "image_url": {"url": b64}}]}], 300, json_mode=True),
                MODEL, time.time() - t0, est_in=1500)
        except ApiError as e:
            event("vlm_error", error=str(e)[:200]); break
        budget.record("vlm", "vlm_primary", look.model, look.tokens_in, look.tokens_out, look.latency_s, t_sim=round(sensor.time, 2), reason="look")
        pentry = {"t": round(sensor.time, 2), "role": "vlm_primary", "model": MODEL, "raw": look.text, "frame_file": f"step{n:03d}.png",
                  "grounding": {"figurine": "model_only", "exit": "model_only"}}
        seen_fig = seen_exit = None
        try:
            per = parse_perception(look.text)
            if per.figurine:
                seen_fig = [round(per.figurine.x * W, 2), round(per.figurine.y * H, 2)]
            if per.exit:
                seen_exit = [round(per.exit.x * W, 2), round(per.exit.y * H, 2)]
            pentry.update(valid=True, parsed=per.model_dump())
        except Exception as e:  # noqa: BLE001
            pentry.update(valid=False, error=str(e)[:200], parsed=None)
        log["perceptions"].append(pentry)
        ctx["my_vision"] = {"figurine_m": seen_fig, "exit_m": seen_exit,
                            "note": "what you just saw in this frame, converted to metres"}
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
            {"type": "text", "text": json.dumps(ctx)},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}]
        t0 = time.time()
        try:
            res = _result(vlm.t.chat(MODEL, msgs, 600, json_mode=True), MODEL, time.time() - t0, est_in=1500)
        except ApiError as e:
            event("plan_error", error=str(e)[:200]); break
        budget.record("plan", "planner", res.model, res.tokens_in, res.tokens_out, res.latency_s, t_sim=round(sensor.time, 2), reason="step")
        entry = {"t": round(sensor.time, 2), "reason": "step", "role": "planner", "model": MODEL, "round": 0, "attempt": n, "raw": res.text}
        try:
            cmd = json.loads(extract_json(res.text)); act = cmd.get("action")
        except Exception as e:  # noqa: BLE001
            entry.update(accepted=False, rejected_by="schema", reasons=[f"not valid JSON: {str(e)[:120]}"])
            log["plans"].append(entry); hist.append({"cmd": "?", "result": "invalid JSON"}); continue
        seen = cmd.get("figurine_seen"); why = str(cmd.get("reason", ""))[:200]
        short = {"action": act, "x": cmd.get("x"), "y": cmd.get("y")}
        if act == "move_to":
            try:
                tx, ty = float(cmd["x"]), float(cmd["y"])
            except Exception:  # noqa: BLE001
                tx = ty = None
            bad = ["move_to needs numeric x and y"] if tx is None else []
            if not bad and math.hypot(tx - x, ty - y) > STEP_MAX:
                k = 1.5 / math.hypot(tx - x, ty - y)      # a far target is walked toward, 1.5 m per step
                tx, ty = x + (tx - x) * k, y + (ty - y) * k
            if not bad:
                plan = Plan(waypoints=[Waypoint(x=tx, y=ty, speed=SPEED, action="move")], rationale=why or "step")
                bad = validate_plan(plan, pmap, (x, y), lim["wall_margin_m"], lim["fire_margin_m"] + (cfg["sim"]["carry_offset_m"] if carrying else 0), lim["plan_speed_max"])
            if bad:
                entry.update(accepted=False, rejected_by="validator", reasons=bad[:3])
                log["plans"].append(entry); hist.append({"cmd": short, "result": "REFUSED: " + "; ".join(bad[:2])}); continue
            entry.update(accepted=True, plan={"waypoints": [{"x": tx, "y": ty, "speed": SPEED, "action": "move"}], "rationale": why})
            log["plans"].append(entry)
            x0, y0 = x, y
            for _ in range(int(60 / cfg["sim"]["dt"])):
                if sensor.done:
                    break
                px, py, pth = sensor.pose(); d = math.hypot(tx - px, ty - py)
                if d < 0.05:
                    break
                err = math.atan2(math.sin(math.atan2(ty - py, tx - px) - pth), math.cos(math.atan2(ty - py, tx - px) - pth))
                sensor.actuate((SPEED * min(1, d / 0.3) if abs(err) < 0.35 else 0.0, 0.0, max(-2.0, min(2.0, 3.0 * err))))
            px, py, _ = sensor.pose()
            hist.append({"cmd": short, "result": f"moved {math.hypot(px - x0, py - y0):.2f} m, now at ({px:.2f}, {py:.2f})", "figurine_seen": seen})
        elif act in ("pickup", "release"):
            if act == "pickup":
                ok = sensor.grasp(); event("pickup", ok=ok)
                res_txt = "picked up the figurine" if ok else "FAILED: the figurine is not within 0.35 m"
            else:
                ok = carrying
                if ok:
                    sensor.release(); event("release")
                res_txt = "released" if ok else "FAILED: you are not carrying anything"
            entry.update(accepted=ok, rejected_by=None if ok else "task_contract", reasons=[] if ok else [res_txt],
                         plan={"waypoints": [{"x": x, "y": y, "speed": 0.1, "action": act}], "rationale": why} if ok else None)
            log["plans"].append(entry); hist.append({"cmd": short, "result": res_txt, "figurine_seen": seen})
            for _ in range(3):
                sensor.actuate((0.0, 0.0, 0.0))
        else:
            entry.update(accepted=False, rejected_by="schema", reasons=[f"unknown action {act!r}"])
            log["plans"].append(entry); hist.append({"cmd": short, "result": "unknown action"})
        print(f"  step {n:2d} t={sensor.time:5.1f} {act:8s} {short.get('x')},{short.get('y')} -> {hist[-1]['result'][:70]}", flush=True)
    if sensor.carrying and not sensor.done:     # truthful end: it puts the figurine down where it stands
        sensor.release(); event("release")
        for _ in range(3):
            sensor.actuate((0.0, 0.0, 0.0))
    event("episode_end")
    return log
M.run_agent = drive



def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=191)
    ap.add_argument("--out", default="runs3d/model_drive")
    ap.add_argument("--max-usd", default="0.50")
    a = ap.parse_args(argv)
    M.main(["--live", "--seed", str(a.seed), "--out", a.out, "--max-usd", a.max_usd,
            "--max-vlm-calls", "4", "--vlm-period", "1000"])


if __name__ == "__main__":
    main()
