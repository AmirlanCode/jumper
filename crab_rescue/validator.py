"""Programmatic safety validator: the only thing between a model's JSON and the robot.

It works on the PERCEIVED map (what the VLM reported), never on simulator truth. A plan is
rejected when it leaves the room, crosses a perceived wall/obstacle/fire cell, or breaks
speed limits. Reasons are returned as text so they can be fed back to the planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import near_blocked
from .schemas import Box, Perception, Plan


@dataclass
class PerceivedMap:
    w: int
    h: int
    cell: float
    walls: np.ndarray        # [h, w] bool: ring + perceived obstacles
    fire: np.ndarray         # [h, w] bool
    figurine: tuple[float, float] | None
    exit_xy: tuple[float, float] | None

    @property
    def width_m(self) -> float:
        return self.w * self.cell

    @property
    def height_m(self) -> float:
        return self.h * self.cell


def rasterize(boxes: list[Box], w: int, h: int, min_overlap: float = 0.3) -> np.ndarray:
    """Cells whose area is covered at least `min_overlap` by one of the boxes."""
    out = np.zeros((h, w), dtype=bool)
    for b in boxes:
        for cy in range(max(0, int(b.y0 * h) - 1), min(h, int(b.y1 * h) + 2)):
            for cx in range(max(0, int(b.x0 * w) - 1), min(w, int(b.x1 * w) + 2)):
                ox = max(0.0, min(b.x1 * w, cx + 1) - max(b.x0 * w, cx))
                oy = max(0.0, min(b.y1 * h, cy + 1) - max(b.y0 * h, cy))
                if ox * oy >= min_overlap:
                    out[cy, cx] = True
    return out


def build_map(perception: Perception, w: int, h: int, cell: float) -> PerceivedMap:
    W, H = w * cell, h * cell
    walls = rasterize(perception.obstacles, w, h)
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    exit_xy = None
    if perception.exit is not None:
        exit_xy = (perception.exit.x * W, perception.exit.y * H)
        for cy in range(h):
            for cx in range(w):
                if ring[cy, cx] and math.hypot(
                        (cx + 0.5) * cell - exit_xy[0], (cy + 0.5) * cell - exit_xy[1]) <= 0.6:
                    ring[cy, cx] = False
    figurine = ((perception.figurine.x * W, perception.figurine.y * H)
                if perception.figurine is not None else None)
    fire = rasterize(perception.fire, w, h)
    return PerceivedMap(w, h, cell, walls | ring, fire, figurine, exit_xy)


def validate_plan(plan: Plan, pmap: PerceivedMap, start_xy: tuple[float, float],
                  wall_margin: float = 0.2, fire_margin: float = 0.15,
                  max_speed: float = 0.6, step: float = 0.1) -> list[str]:
    """Return a list of human-readable rejection reasons; empty means the plan is safe."""
    reasons: list[str] = []
    W, H = pmap.width_m, pmap.height_m
    pts = [start_xy] + [(wp.x, wp.y) for wp in plan.waypoints]
    for i, wp in enumerate(plan.waypoints):
        if not (0 <= wp.x <= W and 0 <= wp.y <= H):
            reasons.append(f"waypoint {i} ({wp.x:.2f},{wp.y:.2f}) is outside the room")
        if wp.speed > max_speed:
            reasons.append(f"waypoint {i} speed {wp.speed:.2f} exceeds {max_speed} m/s")
    if reasons:
        return reasons
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        hit_wall = hit_fire = None
        for k in range(n + 1):
            t = k / n
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            if k > 0 and hit_wall is None and near_blocked(pmap.walls, pmap.cell, x, y, wall_margin):
                hit_wall = (x, y)
            # the first 0.3 m is exempt for fire: a crab already beside the fire must be able to leave
            leaving = i == 0 and math.hypot(x - x0, y - y0) < 0.3
            if k > 0 and hit_fire is None and not leaving and near_blocked(
                    pmap.fire, pmap.cell, x, y, fire_margin, outside_blocks=False):
                hit_fire = (x, y)
        if hit_wall:
            reasons.append(f"segment {i}->{i + 1} crosses a wall/obstacle near "
                           f"({hit_wall[0]:.2f},{hit_wall[1]:.2f})")
        if hit_fire:
            reasons.append(f"segment {i}->{i + 1} crosses fire near "
                           f"({hit_fire[0]:.2f},{hit_fire[1]:.2f})")
    return reasons
