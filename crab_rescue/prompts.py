"""Prompts for the real models. They state the output contract exactly; the code still
validates every reply against the pydantic schemas, so a model that ignores them is caught."""

from __future__ import annotations

import json
from typing import Any

VLM_SYSTEM = """You are the vision system of a rescue robot. You see ONE top-down image of a room.
Legend: dark grey = walls; brown squares = obstacles; green cells on the left wall = the EXIT;
orange/red cells = FIRE; light grey tint = smoke (not fire); blue circle = the figurine to rescue;
magenta triangle = the robot (ignore it).

Reply with ONE JSON object and nothing else, using exactly these keys:
{"figurine": {"x": 0..1, "y": 0..1} or null,
 "exit": {"x": 0..1, "y": 0..1} or null,
 "fire": [{"x0":..,"y0":..,"x1":..,"y1":..}, ...],
 "obstacles": [{"x0":..,"y0":..,"x1":..,"y1":..}, ...],
 "confidence": 0..1,
 "notes": "short text"}
Coordinates are fractions of the image: x to the right, y downward, (0,0) top-left, (1,1)
bottom-right. Boxes need x0<x1 and y0<y1. "fire" boxes must cover every burning cell (use several
rectangles for irregular shapes). "obstacles" are the brown squares and every INTERIOR wall; do not
list the outer wall ring. The exit point is the centre of the green opening."""

PLAN_SYSTEM = """You plan the path of a rescue robot. Reply with ONE JSON object and nothing else:
{"waypoints": [{"x": metres, "y": metres, "speed": 0.1..0.6, "action": "move"|"pickup"|"release"}],
 "rationale": "one or two sentences",
 "replan": true|false}
Frame: metres, origin at the top-left of the room, x right, y down. 1 to 16 waypoints, each inside
the room. The robot starts at "pose" (x, y, heading). Goal: reach the figurine, use action "pickup"
at its position, carry it to the exit point and use action "release" there. If "carrying" is true,
skip the pickup. Never route through fire cells, walls or obstacles; keep at least one cell away
from fire when you can, because fire spreads (see fire_cells_per_s). Use only the three action
names. If "feedback" lists problems with your previous attempt, fix exactly those."""


def vlm_user_text(meta: dict[str, Any]) -> str:
    extra = f" Your previous reply was invalid ({meta['repair']}); answer with valid JSON only." \
        if meta.get("repair") else ""
    return "Describe this frame as the JSON object specified." + extra


def plan_user_text(context: dict[str, Any], feedback: list[str]) -> str:
    w, h, cell = context["grid"]
    body = {
        "room_size_m": [w * cell, h * cell], "cell_m": cell, "pose": context["pose"],
        "time_s": context["t"], "carrying": context["carrying"],
        "figurine_m": context.get("figurine"), "perception": context["perception"],
        "fire_cells_per_s": context.get("fire_cells_per_s", 0.0),
        "time_pressure": context.get("time_pressure", False), "reason": context.get("reason"),
        "feedback": feedback,
    }
    return json.dumps(body)
