"""Grid geometry shared by the simulator, the validator and the planner."""

from __future__ import annotations

import math

import numpy as np


def rect_distance(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def near_blocked(mask: np.ndarray, cell: float, x: float, y: float, margin: float,
                 outside_blocks: bool = True) -> bool:
    """True when (x, y) is within `margin` metres of any True cell of `mask` ([h, w])."""
    h, w = mask.shape
    ix0 = math.floor((x - margin) / cell)
    ix1 = math.floor((x + margin) / cell)
    iy0 = math.floor((y - margin) / cell)
    iy1 = math.floor((y + margin) / cell)
    for iy in range(iy0, iy1 + 1):
        for ix in range(ix0, ix1 + 1):
            inside = 0 <= ix < w and 0 <= iy < h
            if not inside:
                if outside_blocks and rect_distance(
                        x, y, ix * cell, iy * cell, (ix + 1) * cell, (iy + 1) * cell) <= margin:
                    return True
                continue
            if mask[iy, ix] and rect_distance(
                    x, y, ix * cell, iy * cell, (ix + 1) * cell, (iy + 1) * cell) <= margin:
                return True
    return False


def cell_of(x: float, y: float, cell: float) -> tuple[int, int]:
    return int(math.floor(x / cell)), int(math.floor(y / cell))


def center_of(cx: int, cy: int, cell: float) -> tuple[float, float]:
    return (cx + 0.5) * cell, (cy + 0.5) * cell
