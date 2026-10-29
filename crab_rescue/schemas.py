"""Strict JSON schemas for what the models may emit.

The LLM never emits executable code: only JSON that validates here. Coordinates in a
Perception are normalised image coordinates (x right, y down, both in [0, 1]) over the
whole room; waypoints in a Plan are metres in the same frame (origin top-left of the room).
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

PLAN_SPEED_MAX = 0.6
MAX_WAYPOINTS = 16
ActionName = Literal["move", "pickup", "release"]


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class Box(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "Box":
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError("box must have x0<x1 and y0<y1")
        return self


class Perception(BaseModel):
    model_config = ConfigDict(extra="forbid")
    figurine: Point | None = None
    exit: Point | None = None
    fire: list[Box] = Field(default_factory=list, max_length=60)
    obstacles: list[Box] = Field(default_factory=list, max_length=60)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = Field(default="", max_length=400)


class Waypoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    speed: float = Field(default=0.4, gt=0.0, le=PLAN_SPEED_MAX)
    action: ActionName = "move"


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    waypoints: list[Waypoint] = Field(min_length=1, max_length=MAX_WAYPOINTS)
    rationale: str = Field(max_length=600)
    replan: bool = False

    @field_validator("waypoints")
    @classmethod
    def _inside_room(cls, v: list[Waypoint], info: ValidationInfo) -> list[Waypoint]:
        bounds = (info.context or {}).get("bounds")
        if bounds:
            w, h = bounds
            for i, wp in enumerate(v):
                if not (0.0 <= wp.x <= w and 0.0 <= wp.y <= h):
                    raise ValueError(f"waypoint {i} ({wp.x:.2f},{wp.y:.2f}) is outside the room "
                                     f"0..{w} x 0..{h} m")
        return v


def extract_json(text: str) -> str:
    """Pull the first JSON object out of a model reply (fences and chatter tolerated)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unterminated JSON object in reply")


def parse_perception(text: str) -> Perception:
    return Perception.model_validate(json.loads(extract_json(text)))


def parse_plan(text: str, bounds: tuple[float, float]) -> Plan:
    return Plan.model_validate(json.loads(extract_json(text)), context={"bounds": bounds})
