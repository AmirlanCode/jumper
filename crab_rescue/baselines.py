"""Baselines. Unlike the agent they read ground truth by design (labelled in metrics):

* blind  - straight line to the figurine, then straight to the exit. Knows where the targets
           are but sees no walls and no fire (no VLM, no planner).
* oracle - A* on the TRUE map and TRUE fire, replanned every 2 s. An upper bound.
"""

from __future__ import annotations

from typing import Any

from .clients import _simplify, plan_path_xy
from .follower import WaypointFollower
from .schemas import Plan, Waypoint
from .executor import World


def _wp(x: float, y: float, speed: float, action: str = "move") -> Waypoint:
    return Waypoint(x=x, y=y, speed=speed, action=action)  # type: ignore[arg-type]


def run_blind(world: World, cfg: dict[str, Any]) -> dict[str, Any]:
    ex, ey = world.room.exit_xy
    plan = Plan(waypoints=[_wp(*world.room.figurine, 0.4, "pickup"), _wp(ex + 0.05, ey, 0.4, "release")],
                rationale="straight line to the figurine, then straight to the exit")
    return _drive(world, cfg, lambda: plan, replan_period=1e9, label="blind")


def run_oracle(world: World, cfg: dict[str, Any]) -> dict[str, Any]:
    cell = world.room.cell
    walls = world.room.blocked_mask()

    def make() -> Plan | None:
        x, y, _ = world.pose
        fire = world.fire.burning
        wps = []
        cur = (x, y)
        if not world.carried:
            path = plan_path_xy(walls, fire, cell, cur, tuple(world.fig))
            if path is None:
                return None
            for px, py in _simplify(path, walls, fire, cell)[1:-1]:
                wps.append(_wp(px, py, 0.5))
            wps.append(_wp(world.fig[0], world.fig[1], 0.3, "pickup"))
            cur = tuple(world.fig)
        ex = (world.room.exit_xy[0] + 0.05, world.room.exit_xy[1])
        path = plan_path_xy(walls, fire, cell, cur, ex)
        if path is None:
            return None
        for px, py in _simplify(path, walls, fire, cell)[1:-1]:
            wps.append(_wp(px, py, 0.5))
        wps.append(_wp(ex[0], ex[1], 0.3, "release"))
        return Plan(waypoints=wps[:16], rationale="A* on the true map and true fire")

    return _drive(world, cfg, make, replan_period=2.0, label="oracle")


def _drive(world: World, cfg: dict[str, Any], make_plan, replan_period: float, label: str) -> dict[str, Any]:
    fol = WaypointFollower(tuple(cfg["limits"]["cmd_max"]))
    last = -1e9
    events: list[dict[str, Any]] = []
    plans = 0
    while not world.done:
        if world.t - last >= replan_period or fol.finished:
            if world.t - last >= 0.5 or last < 0:
                last = world.t
                p = make_plan()
                if p is not None:
                    fol.set_plan(p)
                    plans += 1
        cmd, action = fol.command(tuple(world.pose))
        if action == "pickup":
            world.grasp()
        elif action == "release":
            world.release()
        world.step(cmd)
    return {"label": label, "plans": plans, "events": events}
