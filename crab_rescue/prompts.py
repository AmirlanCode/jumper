"""Prompts for the real models. They state the output contract exactly; the code still
validates every reply against the pydantic schemas, so a model that ignores them is caught."""

from __future__ import annotations

import json
from typing import Any

VLM_SYSTEM = """You are the vision system of a rescue robot. You see ONE top-down image of a room.
Legend: green on the left wall = exit; blue circle = figurine; magenta triangle = robot (ignore it).
Locate only the centre of the blue figurine and the green exit. Coordinates are fractions of the
image: x right, y down, (0,0) top-left, (1,1) bottom-right. If an item is not visible, use null.

Reply with one JSON object and nothing else, with exactly this shape:
{"figurine":{"x":0.8,"y":0.5},"exit":{"x":0.02,"y":0.5},"fire":[],"obstacles":[],"confidence":0.9,"notes":""}
Replace example coordinates with what you see. Keep fire and obstacles as empty arrays; they are
refined from the same camera image. Always include all six keys and a confidence from 0 to 1."""

PLAN_SYSTEM = """You plan the path of a rescue robot. Reply with ONE JSON object and nothing else:
{"waypoints": [{"x": metres, "y": metres, "speed": 0.1..0.6, "action": "move"|"pickup"|"release"}],
 "rationale": "one or two sentences",
 "replan": true|false}
Frame: metres, origin at the top-left of the room, x right, y down. 1 to 16 waypoints, each inside
the room. The robot starts at "pose" (x, y, heading). `map_ascii` is the camera-derived grid:
`#` wall/obstacle, `F` fire, `.` free, with `R` robot, `V` figurine, `E` exit markers.
One character is one grid cell; rows and columns are numbered, and cell (col, row) has its centre
at x = (col + 0.5) * cell_m, y = (row + 0.5) * cell_m. Route through `.` cells and do not cross
`#` or `F` cells: a wall can only be passed where its column or row has a `.` gap.
The robot drives in a straight line between consecutive waypoints, so put a waypoint only where
the route turns, never one per cell: a whole rescue fits in 4 to 10 waypoints.
Use the exact `figurine_m` and `exit_m` targets. Reach the figurine and use action `pickup` there,
then carry it to the exit and use `release` there. Every plan must END with the `release` waypoint
at `exit_m`; if `carrying` is true, skip pickup. Keep at
least one cell away from fire when possible. Use only the three action names. If `feedback` lists
problems with your previous attempt, fix exactly those."""


def vlm_user_text(meta: dict[str, Any]) -> str:
    extra = f" Your previous reply was invalid ({meta['repair']}); answer with valid JSON only." \
        if meta.get("repair") else ""
    return "Describe this frame as the JSON object specified." + extra


def _numbered(map_ascii: str | None) -> str | None:
    """Label rows and columns: the planners miscounted columns of the bare grid (live, 2026-10-30)."""
    if not map_ascii:
        return map_ascii
    rows = map_ascii.split("\n")
    w = len(rows[0])
    head = ["    " + "".join(str(c // 10) for c in range(w)), "    " + "".join(str(c % 10) for c in range(w))]
    return "\n".join(head + [f"{r:3d} {row}" for r, row in enumerate(rows)])


def plan_user_text(context: dict[str, Any], feedback: list[str]) -> str:
    w, h, cell = context["grid"]
    body = {
        "room_size_m": [w * cell, h * cell], "cell_m": cell, "pose": context["pose"],
        "time_s": context["t"], "carrying": context["carrying"],
        "figurine_m": context.get("figurine"), "exit_m": context.get("exit"),
        "map_ascii": _numbered(context.get("map_ascii")), "perception": context["perception"],
        "fire_cells_per_s": context.get("fire_cells_per_s", 0.0),
        "time_pressure": context.get("time_pressure", False), "reason": context.get("reason"),
        "feedback": feedback,
    }
    return json.dumps(body)
