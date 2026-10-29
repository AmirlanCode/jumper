"""Simulation world and the sensor view the agent is allowed to see.

`World` holds ground truth and is read only by the evaluator and the baselines.
`SensorView` is the agent's whole interface: a camera frame, odometry, a velocity command,
and grasp/release. The agent module must never import or touch `World`.

HONEST SIMPLIFICATIONS (labelled on the site):
* The executor here is a KINEMATIC stand-in (holonomic body-frame integration with wall
  collisions). It exposes the same (vx, vy, yaw-rate) command the real locomotion policy takes,
  so a MuJoCo + ONNX executor can replace it behind `Executor` without touching the agent.
* Fire is a simulated hazard grid, not fire physics.
* "Carry" is weld-on-contact: the figurine is attached in front of the crab when within
  `grasp_radius_m`. No manipulation is simulated.
* The simulation is paused while a model call is in flight.
"""

from __future__ import annotations

import math
from typing import Any, Protocol


from .fire import FireModel
from .geometry import cell_of, near_blocked
from .render import png_bytes, render_world
from .scene import EXIT, Room


class Executor(Protocol):
    def step(self, cmd: tuple[float, float, float], dt: float) -> None: ...


class World:
    def __init__(self, room: Room, cfg: dict[str, Any], seed: int):
        self.room = room
        self.cfg = cfg
        s = cfg["sim"]
        self.dt = s["dt"]
        self.radius = s["robot_radius_m"]
        self.grasp_r = s["grasp_radius_m"]
        self.carry_off = s["carry_offset_m"]
        self.max_t = s["max_episode_s"]
        self.burn_limit = s["fire"]["figurine_burn_s"]
        self.fire = FireModel(room, seed, s["fire"]["p_spread"], s["fire"]["step_s"])
        self.blocked = room.blocked_mask()
        self.pose = list(room.start)
        self.fig = list(room.figurine)
        self.carried = False
        self.t = 0.0
        self.done = False
        self.outcome = "running"
        self.rescued_t: float | None = None
        # ground-truth counters (evaluator only)
        self.fire_exposure_s = 0.0
        self.fire_entries = 0
        self.collisions = 0
        self._in_fire = False
        self._in_contact = False
        self._fig_burn_s = 0.0
        self.trail: list[tuple[float, float]] = [(self.pose[0], self.pose[1])]
        self.gt_frames: list[dict[str, Any]] = []   # GT snapshot per camera frame
        self.observers: list[Any] = []              # called after every step (video recorder)

    # ---- physics -----------------------------------------------------------------
    def step(self, cmd: tuple[float, float, float], dt: float | None = None) -> None:
        if self.done:
            return
        dt = dt or self.dt
        vx, vy, wz = cmd
        cmax = self.cfg["limits"]["cmd_max"]
        vx = max(-cmax[0], min(cmax[0], vx))
        vy = max(-cmax[1], min(cmax[1], vy))
        wz = max(-cmax[2], min(cmax[2], wz))
        x, y, th = self.pose
        th += wz * dt
        dx = (vx * math.cos(th) - vy * math.sin(th)) * dt
        dy = (vx * math.sin(th) + vy * math.cos(th)) * dt
        blocked_now = False
        for nx, ny in ((x + dx, y + dy), (x + dx, y), (x, y + dy)):
            if not near_blocked(self.blocked, self.room.cell, nx, ny, self.radius):
                x, y = nx, ny
                break
            blocked_now = True
        else:
            pass
        if blocked_now and not self._in_contact:
            self.collisions += 1
        self._in_contact = blocked_now
        self.pose = [x, y, th]
        if self.carried:
            self.fig = [x + self.carry_off * math.cos(th), y + self.carry_off * math.sin(th)]
        self.t += dt
        self.fire.advance_to(self.t)
        self._account()
        for cb in self.observers:
            cb(self)
        if len(self.trail) == 0 or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.05:
            self.trail.append((x, y))

    def _account(self) -> None:
        c = self.room.cell
        cx, cy = cell_of(self.pose[0], self.pose[1], c)
        in_fire = bool(0 <= cx < self.room.w and 0 <= cy < self.room.h and self.fire.burning[cy, cx])
        if in_fire:
            self.fire_exposure_s += self.dt
            if not self._in_fire:
                self.fire_entries += 1
        self._in_fire = in_fire
        fx, fy = cell_of(self.fig[0], self.fig[1], c)
        if 0 <= fx < self.room.w and 0 <= fy < self.room.h:
            if self.room.grid[fy, fx] == EXIT:
                self.done, self.outcome, self.rescued_t = True, "rescued", self.t
                return
            if self.fire.burning[fy, fx]:
                self._fig_burn_s += self.dt
                if self._fig_burn_s >= self.burn_limit:
                    self.done, self.outcome = True, "figurine_lost_to_fire"
                    return
        if self.t >= self.max_t:
            self.done, self.outcome = True, "timeout"

    def grasp(self) -> bool:
        if math.hypot(self.fig[0] - self.pose[0], self.fig[1] - self.pose[1]) <= self.grasp_r:
            self.carried = True
            return True
        return False

    def release(self) -> None:
        self.carried = False

    # ---- ground truth snapshot (evaluator only) ----------------------------------------
    def truth(self) -> dict[str, Any]:
        return {"t": self.t, "figurine": tuple(self.fig), "exit_xy": self.room.exit_xy,
                "fire": self.fire.burning.copy(), "pose": tuple(self.pose)}


class SensorView:
    """What the agent may use. Nothing here returns ground truth."""

    def __init__(self, world: World, scale: int = 24):
        self._w = world
        self._scale = scale

    @property
    def time(self) -> float:
        return self._w.t

    @property
    def done(self) -> bool:
        return self._w.done

    def bounds(self) -> tuple[float, float, int, int, float]:
        r = self._w.room
        return r.width_m, r.height_m, r.w, r.h, r.cell

    def pose(self) -> tuple[float, float, float]:
        return tuple(self._w.pose)  # type: ignore[return-value]

    def frame(self) -> tuple[bytes, int]:
        """A top-down camera image (PNG) and an opaque frame id."""
        w = self._w
        img = render_world(w.room, w.fire.burning, tuple(w.pose), tuple(w.fig), scale=self._scale)
        w.gt_frames.append(w.truth())
        return png_bytes(img), len(w.gt_frames) - 1

    def actuate(self, cmd: tuple[float, float, float]) -> None:
        self._w.step(cmd)

    def grasp(self) -> bool:
        return self._w.grasp()

    def release(self) -> None:
        self._w.release()

    @property
    def carrying(self) -> bool:
        return self._w.carried
