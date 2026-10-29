"""Waypoint follower: turns a plan into (vx, vy, yaw-rate) commands for the locomotion policy."""

from __future__ import annotations

import math

from .schemas import Plan


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class WaypointFollower:
    def __init__(self, cmd_max: tuple[float, float, float], reach_m: float = 0.2,
                 action_reach_m: float = 0.3, k_yaw: float = 3.0):
        self.cmd_max = cmd_max
        self.reach = reach_m
        self.action_reach = action_reach_m
        self.k_yaw = k_yaw
        self.plan: Plan | None = None
        self.idx = 0

    def set_plan(self, plan: Plan | None) -> None:
        self.plan, self.idx = plan, 0

    @property
    def finished(self) -> bool:
        return self.plan is None or self.idx >= len(self.plan.waypoints)

    def command(self, pose: tuple[float, float, float]) -> tuple[tuple[float, float, float], str | None]:
        """Return (cmd, action-to-run-now). The action fires when a waypoint is reached."""
        if self.finished:
            return (0.0, 0.0, 0.0), None
        wp = self.plan.waypoints[self.idx]  # type: ignore[union-attr]
        x, y, th = pose
        dist = math.hypot(wp.x - x, wp.y - y)
        tol = self.action_reach if wp.action != "move" else self.reach
        if dist <= tol:
            self.idx += 1
            return (0.0, 0.0, 0.0), (wp.action if wp.action != "move" else None)
        err = _wrap(math.atan2(wp.y - y, wp.x - x) - th)
        wz = max(-self.cmd_max[2], min(self.cmd_max[2], self.k_yaw * err))
        speed = min(wp.speed, self.cmd_max[0], dist / 0.5 + 0.1)
        vx = speed * max(0.0, math.cos(err)) ** 2
        return (vx, 0.0, wz), None
