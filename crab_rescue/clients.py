"""Model interfaces and deterministic fakes.

`VLMClient` / `PlannerClient` are the only seams to a model. The fakes cost $0, are fully
deterministic, and exercise the same JSON contract the real models will have to meet. The
FakeVLM sees ONLY the PNG pixels (it decodes colours), the FakePlanner sees ONLY the
structured context the agent passes in.
"""

from __future__ import annotations

import hashlib
import heapq
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .render import PALETTE
from .validator import build_map, rasterize  # noqa: F401  (rasterize re-exported for tests)
from .schemas import Box, Perception, Point


class ModelCallError(RuntimeError):
    """A model call failed (HTTP or transport). The agent keeps running on its last valid plan."""


@dataclass
class CallResult:
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float = 0.0
    model: str = "fake"


class VLMClient(Protocol):
    def perceive(self, role: str, image_png: bytes, meta: dict[str, Any]) -> CallResult: ...


class PlannerClient(Protocol):
    def plan(self, role: str, context: dict[str, Any], feedback: list[str]) -> CallResult: ...


# ----------------------------------------------------------------------------------------
# Fake VLM: decodes the rendered image
# ----------------------------------------------------------------------------------------
def _classify(arr: np.ndarray) -> dict[str, np.ndarray]:
    names = list(PALETTE)
    cols = np.array([PALETTE[n] for n in names], dtype=int)
    d = np.abs(arr[:, :, None, :].astype(int) - cols[None, None, :, :]).sum(-1)
    idx = d.argmin(-1)
    ok = d.min(-1) <= 40
    return {n: (idx == i) & ok for i, n in enumerate(names)}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask)
    comps = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not seen[y, x]:
                stack, comp = [(x, y)], []
                seen[y, x] = True
                while stack:
                    cx, cy = stack.pop()
                    comp.append((cx, cy))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))
                comps.append(comp)
    return comps


class FakeVLM:
    """Deterministic perception by colour decoding. `noise` degrades the cheap tier."""

    def __init__(self, grid: tuple[int, int] = (20, 16), cheap_miss_rate: float = 0.15,
                 cheap_jitter: float = 0.02, seed: int = 0):
        self.grid = grid
        self.miss = cheap_miss_rate
        self.jitter = cheap_jitter
        self.seed = seed
        self.n = 0

    def perceive(self, role: str, image_png: bytes, meta: dict[str, Any]) -> CallResult:
        self.n += 1
        rng = np.random.default_rng(self.seed * 1009 + self.n)
        cheap = role == "vlm_cheap"
        img = np.asarray(Image.open(io.BytesIO(image_png)).convert("RGB"))
        H, W = img.shape[:2]
        gw, gh = self.grid
        masks = _classify(img)

        def centroid(name: str) -> Point | None:
            ys, xs = np.nonzero(masks[name])
            if len(xs) == 0:
                return None
            x, y = xs.mean() / W, ys.mean() / H
            if cheap:
                x, y = x + rng.normal(0, self.jitter), y + rng.normal(0, self.jitter)
            return Point(x=float(min(max(x, 0), 1)), y=float(min(max(y, 0), 1)))

        def cell_mask(name: str) -> np.ndarray:
            m = np.zeros((gh, gw), dtype=bool)
            sy, sx = H // gh, W // gw
            for cy in range(gh):
                for cx in range(gw):
                    blk = masks[name][cy * sy:(cy + 1) * sy, cx * sx:(cx + 1) * sx]
                    m[cy, cx] = blk.mean() > 0.5
            return m

        def boxes(m: np.ndarray, drop: float = 0.0, skip_ring: bool = False) -> list[Box]:
            m = m.copy()
            if skip_ring:
                m[0, :] = m[-1, :] = False
                m[:, 0] = m[:, -1] = False
            if drop:
                # the cheap tier misses some cells; keep at least one
                keep = rng.random(m.shape) > drop
                kept = m & keep
                m = kept if kept.any() else m
            return [_bbox_rect(r, gw, gh) for r in _rect_cover(m)][:60]

        fire_m = cell_mask("fire")
        wall_like = cell_mask("wall") | cell_mask("obstacle")
        smoke = bool(masks["smoke"].any())
        perception = Perception(
            figurine=centroid("figurine"),
            exit=_exit_point(masks["exit"], W, H),
            fire=boxes(fire_m, drop=self.miss if cheap else 0.0),
            obstacles=boxes(wall_like, skip_ring=True),
            confidence=0.62 if cheap else 0.93,
            notes=("smoke visible near the fire" if smoke else "no smoke") + f"; tier={role}",
        )
        text = perception.model_dump_json()
        return CallResult(text, tokens_in=1500, tokens_out=len(text) // 4, model=f"fake-{role}")


def _rect_cover(m: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Greedy exact cover of True cells by rectangles (x0, y0, x1, y1 inclusive)."""
    h, w = m.shape
    used = np.zeros_like(m)
    out = []
    for y in range(h):
        for x in range(w):
            if m[y, x] and not used[y, x]:
                x1 = x
                while x1 + 1 < w and m[y, x1 + 1] and not used[y, x1 + 1]:
                    x1 += 1
                y1 = y
                while y1 + 1 < h and m[y1 + 1, x:x1 + 1].all() and not used[y1 + 1, x:x1 + 1].any():
                    y1 += 1
                used[y:y1 + 1, x:x1 + 1] = True
                out.append((x, y, x1, y1))
    return out


def _bbox_rect(r: tuple[int, int, int, int], gw: int, gh: int) -> Box:
    return Box(x0=r[0] / gw, y0=r[1] / gh, x1=(r[2] + 1) / gw, y1=(r[3] + 1) / gh)


def _exit_point(mask: np.ndarray, W: int, H: int) -> Point | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return Point(x=float(xs.mean() / W), y=float(ys.mean() / H))


# ----------------------------------------------------------------------------------------
# A* on the perceived grid (used by FakePlanner and the oracle baseline)
# ----------------------------------------------------------------------------------------
def astar(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    if blocked[goal[1], goal[0]]:
        return None
    pq = [(0.0, start)]
    g = {start: 0.0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in prev:
                cur = prev[cur]
                path.append(cur)
            return path[::-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < w and 0 <= ny < h) or blocked[ny, nx]:
                continue
            if dx and dy and (blocked[cur[1], nx] or blocked[ny, cur[0]]):
                continue
            ng = g[cur] + math.hypot(dx, dy)
            if ng < g.get((nx, ny), 1e9):
                g[(nx, ny)] = ng
                prev[(nx, ny)] = cur
                heapq.heappush(pq, (ng + math.hypot(goal[0] - nx, goal[1] - ny), (nx, ny)))
    return None


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return m.copy()
    out = m.copy()
    ys, xs = np.nonzero(m)
    h, w = m.shape
    for y, x in zip(ys, xs):
        out[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)] = True
    return out


def plan_path_xy(walls: np.ndarray, fire: np.ndarray, cell: float, a: tuple[float, float],
                 b: tuple[float, float], fire_inflate: int = 2) -> list[tuple[float, float]] | None:
    """Cell-centre waypoints from a to b avoiding fire (inflated when possible)."""
    h, w = walls.shape
    sa = (min(max(int(a[0] / cell), 0), w - 1), min(max(int(a[1] / cell), 0), h - 1))
    sb = (min(max(int(b[0] / cell), 0), w - 1), min(max(int(b[1] / cell), 0), h - 1))
    for infl in range(fire_inflate, -1, -1):
        blocked = walls | _dilate(fire, infl)
        blocked[sa[1], sa[0]] = False
        path = astar(blocked, sa, sb)
        if path:
            pts = [((x + 0.5) * cell, (y + 0.5) * cell) for x, y in path]
            pts[0] = a          # start from where the crab really is, not its cell centre
            return pts
    return None


def _simplify(pts: list[tuple[float, float]], walls: np.ndarray, fire: np.ndarray, cell: float,
              max_pts: int = 12) -> list[tuple[float, float]]:
    """Keep corners only; keep a segment straight when the straight line stays clear."""
    from .geometry import near_blocked
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1:
            ok = True
            n = max(1, int(math.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]) / 0.1))
            for k in range(1, n + 1):
                t = k / n
                x = pts[i][0] + t * (pts[j][0] - pts[i][0])
                y = pts[i][1] + t * (pts[j][1] - pts[i][1])
                if near_blocked(walls, cell, x, y, 0.25) or near_blocked(fire, cell, x, y, 0.2, False):
                    ok = False
                    break
            if ok:
                break
            j -= 1
        out.append(pts[j])
        i = j
    if len(out) > max_pts:
        step = math.ceil(len(out) / max_pts)
        out = out[::step] + [out[-1]]
    return out


class FakePlanner:
    """Deterministic planner: A* on the perceived map. Optionally emits bad plans first
    (`bad_first_rate`) so the repair / escalation machinery can be tested."""

    def __init__(self, bad_first_rate: float = 0.0, seed: int = 0, speed: float = 0.5):
        self.bad_rate = bad_first_rate
        self.seed = seed
        self.speed = speed
        self.n = 0

    def plan(self, role: str, context: dict[str, Any], feedback: list[str]) -> CallResult:
        self.n += 1
        rng = np.random.default_rng(self.seed * 31 + self.n)
        bad = (not feedback) and role != "planner_escalation" and rng.random() < self.bad_rate
        if bad:
            kind = int(rng.integers(0, 2))
            text = "I think we should just run" if kind == 0 else json.dumps({
                "waypoints": [{"x": context["pose"][0] + 0.1, "y": context["pose"][1], "speed": 0.4,
                               "action": "move"},
                              {"x": context["figurine"][0] if context.get("figurine") else 5.0,
                               "y": context["figurine"][1] if context.get("figurine") else 4.0,
                               "speed": 0.5, "action": "pickup"}],
                "rationale": "straight line, ignoring obstacles", "replan": False})
            return CallResult(text, 900, len(text) // 4, model=f"fake-{role}")
        text = self._good_plan(context)
        return CallResult(text, 900 + 40 * len(feedback), len(text) // 4, model=f"fake-{role}")

    def _good_plan(self, ctx: dict[str, Any]) -> str:
        per = Perception.model_validate(ctx["perception"])
        w, h, cell = ctx["grid"]
        pm = build_map(per, w, h, cell)
        x, y, _ = ctx["pose"]
        carrying = ctx["carrying"]
        wps: list[dict[str, Any]] = []
        notes = []
        cur = (x, y)
        if not carrying:
            if pm.figurine is None:
                tgt = (pm.width_m * 0.75, pm.height_m * 0.5)
                notes.append("figurine not visible; exploring east")
            else:
                tgt = pm.figurine
            path = plan_path_xy(pm.walls, pm.fire, cell, cur, tgt)
            if path is None:
                return json.dumps({"waypoints": [{"x": x, "y": y, "speed": 0.1, "action": "move"}],
                                   "rationale": "no safe path to the figurine; holding position",
                                   "replan": True})
            pts = _simplify(path, pm.walls, pm.fire, cell)[1:-1] if len(path) > 2 else []
            for px, py in pts:
                wps.append({"x": round(px, 3), "y": round(py, 3), "speed": self.speed, "action": "move"})
            if pm.figurine is not None:
                wps.append({"x": round(pm.figurine[0], 3), "y": round(pm.figurine[1], 3),
                            "speed": 0.3, "action": "pickup"})
                cur = pm.figurine
                notes.append("go to the figurine through the door, avoiding fire cells")
            else:
                wps.append({"x": round(tgt[0], 3), "y": round(tgt[1], 3), "speed": self.speed, "action": "move"})
                cur = tgt
        if pm.exit_xy is not None and (carrying or pm.figurine is not None):
            ex = (pm.exit_xy[0] + 0.05, pm.exit_xy[1])
            path = plan_path_xy(pm.walls, pm.fire, cell, cur, ex)
            if path:
                pts = _simplify(path, pm.walls, pm.fire, cell)[1:-1] if len(path) > 2 else []
                for px, py in pts:
                    wps.append({"x": round(px, 3), "y": round(py, 3), "speed": self.speed, "action": "move"})
                wps.append({"x": round(ex[0], 3), "y": round(ex[1], 3), "speed": 0.3, "action": "release"})
                notes.append("then carry it to the exit, keeping a cell away from fire")
            else:
                notes.append("no safe route to the exit yet; will replan")
        if not wps:
            wps = [{"x": round(x, 3), "y": round(y, 3), "speed": 0.1, "action": "move"}]
        wps = wps[:16]
        trend = ctx.get("fire_cells_per_s", 0.0)
        return json.dumps({"waypoints": wps, "rationale": "; ".join(notes) + f" (fire trend {trend:.2f} cells/s)",
                           "replan": trend > 0.6})


# ----------------------------------------------------------------------------------------
# Disk cache for identical prompts (wraps a real client; fakes do not need it)
# ----------------------------------------------------------------------------------------
class DiskCache:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def key(self, *parts: bytes | str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p if isinstance(p, bytes) else p.encode())
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> CallResult | None:
        f = self.dir / f"{key}.json"
        if f.exists():
            d = json.loads(f.read_text())
            return CallResult(**d)
        return None

    def put(self, key: str, res: CallResult) -> None:
        (self.dir / f"{key}.json").write_text(json.dumps(res.__dict__))
