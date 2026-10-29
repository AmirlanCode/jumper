"""Parametric room generated from a seed.

Layout: outer wall ring, an exit opening (2 cells) in the west wall, one interior wall
with a 2-cell door, a few obstacles. The crab starts next to the exit; the figurine is in
the far (east) half, so the crab must pass the door, pick it up and bring it back.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

FREE, WALL, OBST, EXIT = 0, 1, 2, 3


@dataclass
class Room:
    seed: int
    w: int
    h: int
    cell: float
    grid: np.ndarray                      # [h, w] uint8
    start: tuple[float, float, float]     # x, y, heading (m, m, rad)
    figurine: tuple[float, float]
    exit_cells: list[tuple[int, int]]
    exit_xy: tuple[float, float]          # centre of the opening on the west wall line
    door_cells: list[tuple[int, int]]
    fire_origin: tuple[int, int]
    obstacles: list[tuple[int, int]] = field(default_factory=list)

    @property
    def width_m(self) -> float:
        return self.w * self.cell

    @property
    def height_m(self) -> float:
        return self.h * self.cell

    def blocked_mask(self) -> np.ndarray:
        return (self.grid == WALL) | (self.grid == OBST)


def _reachable(blocked: np.ndarray, src: tuple[int, int]) -> np.ndarray:
    h, w = blocked.shape
    seen = np.zeros_like(blocked, dtype=bool)
    q = deque([src])
    seen[src[1], src[0]] = True
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not blocked[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((nx, ny))
    return seen


def generate_room(seed: int, w: int = 20, h: int = 16, cell: float = 0.5) -> Room:
    rng = np.random.default_rng(seed)
    for _attempt in range(200):
        grid = np.zeros((h, w), dtype=np.uint8)
        grid[0, :] = grid[-1, :] = WALL
        grid[:, 0] = grid[:, -1] = WALL
        exit_row = int(rng.integers(3, h - 5))
        exit_cells = [(0, exit_row), (0, exit_row + 1)]
        for cx, cy in exit_cells:
            grid[cy, cx] = EXIT
        wall_x = int(rng.integers(9, 12))
        grid[1:h - 1, wall_x] = WALL
        door_row = int(rng.integers(2, h - 4))
        door_cells = [(wall_x, door_row), (wall_x, door_row + 1)]
        for cx, cy in door_cells:
            grid[cy, cx] = FREE

        start_cell = (2, exit_row)
        fx = int(rng.integers(wall_x + 3, w - 2))
        fy = int(rng.integers(2, h - 2))
        keep = [start_cell, (fx, fy), (1, exit_row), (1, exit_row + 1), *door_cells]

        obstacles: list[tuple[int, int]] = []
        for _ in range(int(rng.integers(4, 8))):
            ox, oy = int(rng.integers(2, w - 2)), int(rng.integers(2, h - 2))
            if ox == wall_x:
                continue
            if any(abs(ox - kx) <= 1 and abs(oy - ky) <= 1 for kx, ky in keep):
                continue
            grid[oy, ox] = OBST
            obstacles.append((ox, oy))

        blocked = (grid == WALL) | (grid == OBST)
        if blocked[fy, fx] or blocked[start_cell[1], start_cell[0]]:
            continue
        seen = _reachable(blocked, start_cell)
        if not (seen[fy, fx] and seen[exit_row, 1]):
            continue

        # Fire origin: free, away from start/figurine, often near the door so it matters.
        cands = [(x, y) for y in range(1, h - 1) for x in range(1, w - 1)
                 if not blocked[y, x] and seen[y, x]
                 and abs(x - start_cell[0]) + abs(y - start_cell[1]) >= 6
                 and abs(x - fx) + abs(y - fy) >= 3 and (x, y) not in exit_cells]
        if not cands:
            continue
        near_door = [c for c in cands
                     if min(abs(c[0] - dx) + abs(c[1] - dy) for dx, dy in door_cells) <= 5]
        pool = near_door if (near_door and rng.random() < 0.6) else cands
        fire_origin = pool[int(rng.integers(0, len(pool)))]

        return Room(
            seed=seed, w=w, h=h, cell=cell, grid=grid,
            start=((start_cell[0] + 0.5) * cell, (start_cell[1] + 1.0) * cell, 0.0),
            figurine=((fx + 0.5) * cell, (fy + 0.5) * cell),
            exit_cells=exit_cells,
            exit_xy=(0.5 * cell, (exit_row + 1.0) * cell),
            door_cells=door_cells, fire_origin=fire_origin, obstacles=obstacles,
        )
    raise RuntimeError(f"could not generate a valid room for seed {seed}")
