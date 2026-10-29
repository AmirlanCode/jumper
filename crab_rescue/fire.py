"""Cellular-automaton hazard grid. A simulated hazard region, NOT fire physics."""

from __future__ import annotations

import numpy as np

from .scene import FREE, Room


class FireModel:
    def __init__(self, room: Room, seed: int, p_spread: float = 0.2, step_s: float = 1.0):
        self.room = room
        self.p = p_spread
        self.step_s = step_s
        self.rng = np.random.default_rng(seed * 7919 + 1)
        self.burning = np.zeros((room.h, room.w), dtype=bool)
        self.burn_time = np.zeros((room.h, room.w), dtype=float)
        ox, oy = room.fire_origin
        self.burning[oy, ox] = True
        self._t = 0.0
        self._passable = room.grid == FREE   # walls, obstacles and the exit never burn

    @property
    def area(self) -> int:
        return int(self.burning.sum())

    def _step_once(self) -> None:
        b = self.burning
        n = np.zeros(b.shape, dtype=int)
        n[1:, :] += b[:-1, :]
        n[:-1, :] += b[1:, :]
        n[:, 1:] += b[:, :-1]
        n[:, :-1] += b[:, 1:]
        prob = 1.0 - (1.0 - self.p) ** n
        ignite = (self.rng.random(b.shape) < prob) & ~b & self._passable
        self.burning = b | ignite

    def advance_to(self, t: float) -> None:
        while self._t + self.step_s <= t + 1e-9:
            self._step_once()
            self._t += self.step_s
        # burn_time is bookkeeping for display only
        self.burn_time[self.burning] = np.maximum(self.burn_time[self.burning], 1.0)

    def copy_mask(self) -> np.ndarray:
        return self.burning.copy()
