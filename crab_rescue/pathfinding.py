"""Camera-map grid routing used as a safety repair for model-generated plans."""

from __future__ import annotations

import heapq
import math

from .geometry import cell_of, center_of, near_blocked
from .schemas import Plan, Waypoint
from .validator import PerceivedMap


def _grid_path(start: tuple[float, float], goal: tuple[float, float], pmap: PerceivedMap,
               wall_margin: float, fire_margin: float) -> list[tuple[float, float]] | None:
    """Find a four-connected path through perceived free cells, including clearance margins."""
    sx, sy = cell_of(*start, pmap.cell)
    gx, gy = cell_of(*goal, pmap.cell)
    if not (0 <= sx < pmap.w and 0 <= sy < pmap.h and 0 <= gx < pmap.w and 0 <= gy < pmap.h):
        return None

    def free(c: tuple[int, int]) -> bool:
        x, y = center_of(*c, pmap.cell)
        return (not near_blocked(pmap.walls, pmap.cell, x, y, wall_margin)
                and not near_blocked(pmap.fire, pmap.cell, x, y, fire_margin,
                                     outside_blocks=False))

    source, target = (sx, sy), (gx, gy)
    if source != target and (not free(target) or (not free(source) and
                                                   math.dist(start, center_of(*source, pmap.cell)) > 0.3)):
        return None
    queue: list[tuple[float, int, tuple[int, int]]] = []
    serial = 0
    heapq.heappush(queue, (abs(gx - sx) + abs(gy - sy), serial, source))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {source: None}
    cost = {source: 0}
    while queue:
        _, _, current = heapq.heappop(queue)
        if current == target:
            break
        x, y = current
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nxt[0] < pmap.w and 0 <= nxt[1] < pmap.h):
                continue
            if nxt != target and not free(nxt):
                continue
            if nxt == target and not free(nxt):
                continue
            new_cost = cost[current] + 1
            if new_cost >= cost.get(nxt, math.inf):
                continue
            cost[nxt] = new_cost
            came_from[nxt] = current
            serial += 1
            priority = new_cost + abs(gx - nxt[0]) + abs(gy - nxt[1])
            heapq.heappush(queue, (priority, serial, nxt))
    if target not in came_from:
        return None

    cells = []
    current = target
    while current != source:
        cells.append(current)
        current = came_from[current]  # type: ignore[assignment]
    cells.reverse()
    points = [center_of(*c, pmap.cell) for c in cells]
    # Keep only turns, then end at the exact requested semantic waypoint.
    turns: list[tuple[float, float]] = []
    previous = (sx, sy)
    old_dir = None
    for i, cell_xy in enumerate(cells):
        direction = (cell_xy[0] - previous[0], cell_xy[1] - previous[1])
        if old_dir is not None and direction != old_dir:
            turns.append(points[i - 1])
        old_dir = direction
        previous = cell_xy
    if points and (not turns or turns[-1] != points[-1]):
        turns.append(points[-1])
    if not turns or math.dist(turns[-1], goal) > 1e-6:
        turns.append(goal)
    # Drop redundant initial cell-center points within the follower's reach radius.
    while len(turns) > 1 and math.dist(start, turns[0]) <= 0.2:
        turns.pop(0)
    return turns


def route_targets(start: tuple[float, float], targets: list[Waypoint], pmap: PerceivedMap,
                  wall_margin: float, fire_margin: float, speed: float) -> Plan | None:
    """Route semantic waypoints around camera-perceived walls and fire; caller revalidates."""
    pose = start
    routed: list[Waypoint] = []
    for target in targets:
        path = _grid_path(pose, (target.x, target.y), pmap, wall_margin, fire_margin)
        if path is None:
            return None
        for x, y in path[:-1]:
            routed.append(Waypoint(x=x, y=y, speed=min(speed, 0.6), action="move"))
        routed.append(target)
        pose = (target.x, target.y)
    if not routed or len(routed) > 16:
        return None
    return Plan(waypoints=routed, rationale="Safety-routed semantic targets on the perceived grid.",
                replan=True)


def rescue_fallback(start: tuple[float, float], carrying: bool, pmap: PerceivedMap,
                    wall_margin: float, fire_margin: float, speed: float,
                    carry_offset: float = 0.3) -> Plan | None:
    """Build a conservative rescue route when the language model cannot supply a safe one."""
    targets: list[Waypoint] = []
    if not carrying:
        if pmap.figurine is None:
            return None
        targets.append(Waypoint(x=pmap.figurine[0], y=pmap.figurine[1], speed=speed,
                                action="pickup"))
    if pmap.exit_xy is None:
        return None
    # The figurine is carried in front of the robot. Approach the exit facing inward so
    # the carried figurine, rather than the robot's base, reaches the exit cells.
    release_x = min(pmap.width_m - 0.2, pmap.exit_xy[0] + carry_offset)
    targets.append(Waypoint(x=release_x, y=pmap.exit_xy[1], speed=speed, action="release"))
    return route_targets(start, targets, pmap, wall_margin, fire_margin, speed)
