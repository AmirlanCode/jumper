"""Refine VLM localization from the pixels in the rendered camera frame only."""

from __future__ import annotations

import io
from collections import deque

import numpy as np
from PIL import Image

from .render import PALETTE
from .schemas import Box, Perception, Point


def _cell_mask(rgb: np.ndarray, color: tuple[int, int, int], gw: int, gh: int,
               threshold: float = 0.35) -> np.ndarray:
    h, w, _ = rgb.shape
    pixels = np.all(rgb == color, axis=2)
    out = np.zeros((gh, gw), dtype=bool)
    for cy in range(gh):
        y0, y1 = round(cy * h / gh), round((cy + 1) * h / gh)
        for cx in range(gw):
            x0, x1 = round(cx * w / gw), round((cx + 1) * w / gw)
            tile = pixels[y0:y1, x0:x1]
            out[cy, cx] = float(tile.mean()) >= threshold
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out: list[list[tuple[int, int]]] = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        todo = deque([(int(x), int(y))])
        seen[y, x] = True
        group = []
        while todo:
            cx, cy = todo.popleft()
            group.append((cx, cy))
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    todo.append((nx, ny))
        out.append(group)
    return out


def _boxes(mask: np.ndarray, gw: int, gh: int) -> list[Box]:
    boxes = []
    for cells in _components(mask):
        xs, ys = zip(*cells)
        boxes.append(Box(x0=min(xs) / gw, y0=min(ys) / gh,
                         x1=(max(xs) + 1) / gw, y1=(max(ys) + 1) / gh))
    return boxes


def refine_from_frame(p: Perception, image_png: bytes, gw: int, gh: int) -> Perception:
    """Replace uncertain coordinates with palette detections from the image the agent saw."""
    rgb = np.asarray(Image.open(io.BytesIO(image_png)).convert("RGB"))
    h, w, _ = rgb.shape
    fire = _cell_mask(rgb, PALETTE["fire"], gw, gh)
    walls = _cell_mask(rgb, PALETTE["wall"], gw, gh)
    obstacles = _cell_mask(rgb, PALETTE["obstacle"], gw, gh)
    exit_cells = _cell_mask(rgb, PALETTE["exit"], gw, gh)
    # Outer-wall pixels frame the room; only interior wall cells block paths.
    walls[[0, -1], :] = False
    walls[:, [0, -1]] = False
    wall_obs = _boxes(walls | obstacles, gw, gh)
    fire_boxes = _boxes(fire, gw, gh)

    blue = np.all(rgb == PALETTE["figurine"], axis=2)
    figurine = None
    if blue.any():
        yy, xx = np.nonzero(blue)
        figurine = Point(x=float(xx.mean() / w), y=float(yy.mean() / h))

    exit_point = None
    if exit_cells.any():
        yy, xx = np.nonzero(exit_cells)
        exit_point = Point(x=float((xx.mean() + 0.5) / gw), y=float((yy.mean() + 0.5) / gh))

    return Perception(
        figurine=figurine or p.figurine,
        exit=exit_point or p.exit,
        fire=fire_boxes if fire_boxes else p.fire,
        obstacles=wall_obs if wall_obs else p.obstacles,
        confidence=0.99 if figurine and exit_point else p.confidence,
        notes="VLM report refined from rendered-frame colors" if figurine and exit_point else p.notes,
    )


def cross_check_perception(p: Perception, seg_png: bytes, gw: int, gh: int,
                           tol: float = 0.06) -> tuple[Perception, dict[str, str]]:
    """Keep the model's figurine/exit when the segmentation channel agrees (within `tol` of the
    image size); otherwise use the segmentation point and say so. Fire and walls always come
    from the segmentation channel (the model is told to leave them empty)."""
    import math
    pix = refine_from_frame(Perception(confidence=0.0), seg_png, gw, gh)

    def pick(name: str, model_pt: Point | None, pix_pt: Point | None) -> tuple[Point | None, str]:
        if model_pt is None and pix_pt is None:
            return None, "not_found"
        if model_pt is None:
            return pix_pt, "segmentation_used(model_null)"
        if pix_pt is None:
            return model_pt, "model_unverified"
        d = math.hypot(model_pt.x - pix_pt.x, model_pt.y - pix_pt.y)
        if d <= tol:
            return model_pt, "model_verified"
        return pix_pt, f"segmentation_override(model_off_{d:.2f})"

    fig, fig_src = pick("figurine", p.figurine, pix.figurine)
    ext, ext_src = pick("exit", p.exit, pix.exit)
    out = Perception(figurine=fig, exit=ext, fire=pix.fire or p.fire, obstacles=pix.obstacles or p.obstacles,
                     confidence=p.confidence, notes=(p.notes or "")[:300])
    return out, {"figurine": fig_src, "exit": ext_src}
