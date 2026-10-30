"""Top-down rendering: the frames the VLM sees, and the video frames of a run."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .scene import EXIT, FREE, OBST, WALL, Room

PALETTE = {
    "floor": (232, 228, 216),
    "smoke": (186, 184, 182),
    "wall": (52, 54, 66),
    "obstacle": (126, 84, 48),
    "exit": (40, 190, 90),
    "fire": (238, 78, 24),
    "figurine": (36, 112, 255),
    "crab": (200, 40, 190),
}


def _smoke_mask(burning: np.ndarray, radius: int = 2) -> np.ndarray:
    h, w = burning.shape
    out = np.zeros_like(burning)
    ys, xs = np.nonzero(burning)
    for y, x in zip(ys, xs):
        out[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)] = True
    return out


def render_world(room: Room, burning: np.ndarray, crab: tuple[float, float, float],
                 figurine: tuple[float, float], scale: int = 24,
                 trail: list[tuple[float, float]] | None = None,
                 hud: str | None = None) -> Image.Image:
    """Draw the room. `scale` is pixels per cell."""
    W, H = room.w * scale, room.h * scale
    img = Image.new("RGB", (W, H), PALETTE["floor"])
    d = ImageDraw.Draw(img)
    smoke = _smoke_mask(burning)
    for cy in range(room.h):
        for cx in range(room.w):
            g = room.grid[cy, cx]
            if g == WALL:
                col = PALETTE["wall"]
            elif g == OBST:
                col = PALETTE["obstacle"]
            elif g == EXIT:
                col = PALETTE["exit"]
            elif burning[cy, cx]:
                col = PALETTE["fire"]
            elif g == FREE and smoke[cy, cx]:
                col = PALETTE["smoke"]
            else:
                continue
            d.rectangle([cx * scale, cy * scale, (cx + 1) * scale - 1, (cy + 1) * scale - 1], fill=col)
    k = scale / room.cell
    if trail and len(trail) > 1:
        d.line([(x * k, y * k) for x, y in trail], fill=(120, 20, 110), width=max(1, scale // 12))
    fx, fy = figurine[0] * k, figurine[1] * k
    r = scale * 0.32
    d.ellipse([fx - r, fy - r, fx + r, fy + r], fill=PALETTE["figurine"])
    cx_, cy_, th = crab
    px, py = cx_ * k, cy_ * k
    s = scale * 0.42
    tri = [(px + s * math.cos(th), py + s * math.sin(th)),
           (px + s * math.cos(th + 2.5), py + s * math.sin(th + 2.5)),
           (px + s * math.cos(th - 2.5), py + s * math.sin(th - 2.5))]
    d.polygon(tri, fill=PALETTE["crab"])
    if hud:
        d.rectangle([0, 0, W, 18], fill=(0, 0, 0))
        d.text((6, 3), hud, fill=(255, 255, 255))
    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class VideoWriter:
    """Pipes RGB frames into ffmpeg (H.264, yuv420p). No-op when ffmpeg is missing."""

    def __init__(self, path: Path, size: tuple[int, int], fps: int = 10):
        self.path = path
        self.proc = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                return
        w, h = size[0] - size[0] % 2, size[1] - size[1] % 2
        self.size = (w, h)
        self.proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def add(self, img: Image.Image) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        self.proc.stdin.write(img.crop((0, 0, *self.size)).tobytes())

    def close(self) -> None:
        if self.proc is not None and self.proc.stdin is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=60)
