"""3D rescue mission: the crab agent plays one episode, frames come from the 3D scene.

    python -m crab_rescue.mission3d                      # offline stand-in models, $0, labelled as such
    python -m crab_rescue.mission3d --live --video       # real Nebius models (needs NEBIUS_API_KEY)

What happens
  * the room, fire and crab live in a Three.js scene (scene3d/) that runs in headless Chromium;
  * every camera frame the agent sees is rendered from that scene: a lit overhead frame goes to
    the vision model, a flat segmentation frame of the same render gives hazard cells;
  * the agent loop (look -> plan -> validate -> follow -> monitor) is `crab_rescue.agent`, unchanged
    apart from the cross-check that keeps the model's figurine/exit when the pixels agree;
  * the run is saved as mission.json (every call, its input frame, raw reply, parsed action,
    validation, latency, tokens, cost) and replayed by the same scene for the video.
The API key is read from NEBIUS_API_KEY only and never written anywhere.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import math
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .agent import run_agent
from .budget import Budget
from .clients import CallResult, FakePlanner, FakeVLM
from .config import ROLES, load_config
from .executor import SensorView, World
from .nebius import ApiError, ChatTransport, NebiusPlanner, NebiusVLM
from .router import Router
from .scene import Room, generate_room

HERE = Path(__file__).parent
SCENE_DIR = HERE / "scene3d"
BASE_URL = "https://api.tokenfactory.us-central1.nebius.com/v1/"

HOLD_SCAN, HOLD_PICK, HOLD_REL, HOLD_END, DT = 3.0, 2.4, 2.4, 3.5, 0.1

VLM_SYSTEM_3D = """You are the vision system of a rescue robot. You see ONE overhead camera image of a whole room, looking straight down.
How things look: walls are dark blue-grey blocks with thin blue tops; crates are brown boxes; fire is bright orange flames with an orange glow on the floor and dark smoke; the exit is a glowing green doorway with green floor tiles on the LEFT wall; the figurine is a small blue human-shaped doll lying on the floor with a white head; the robot is a small red crab (ignore it).
Locate only the centre of the blue figurine and the centre of the green exit. Coordinates are fractions of the image: x right, y down, (0,0) top-left, (1,1) bottom-right. If an item is not visible, use null.

Reply with one JSON object and nothing else, with exactly this shape:
{"figurine":{"x":0.8,"y":0.5},"exit":{"x":0.02,"y":0.5},"fire":[],"obstacles":[],"confidence":0.9,"notes":""}
Replace example coordinates with what you see. Keep fire and obstacles as empty arrays; they come from a separate segmentation channel. Always include all six keys and a confidence from 0 to 1."""


# ------------------------------------------------------------------------------------ 3D scene
class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a: Any) -> None:  # noqa: D401
        pass


class Site:
    """Static file server for a run folder (also used by the video recorder)."""

    def __init__(self, root: Path):
        handler = partial(_Quiet, directory=str(root))
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        self.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


def _launch(pw: Any) -> Any:
    return pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                    "--ignore-gpu-blocklist"])


def room_to_json(room: Room) -> dict[str, Any]:
    return {"seed": room.seed, "w": room.w, "h": room.h, "cell": room.cell, "grid": room.grid.tolist(),
            "start": list(room.start), "figurine": list(room.figurine), "exit_xy": list(room.exit_xy),
            "door_cells": [list(c) for c in room.door_cells], "fire_origin": list(room.fire_origin),
            "exit_cells": [list(c) for c in room.exit_cells]}


class Scene3D:
    def __init__(self, site_dir: Path):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise SystemExit("playwright is needed to render the 3D scene:\n"
                             "  pip install playwright && playwright install chromium") from None
        self.site = Site(site_dir)
        self._pw = sync_playwright().start()
        self.browser = _launch(self._pw)
        self.page = self.browser.new_page(viewport={"width": 640, "height": 512})
        self.errors: list[str] = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.goto(f"http://127.0.0.1:{self.site.port}/driver.html")
        self.page.wait_for_function("window.driver && window.driver.ready === true", timeout=90000)
        self.renders = 0

    def load_room(self, room: Room) -> None:
        self.page.evaluate("r => window.driver.loadRoom(r)", room_to_json(room))

    def set_state(self, w: World, ignite: dict[int, float]) -> None:
        st = {"t": w.t, "x": w.pose[0], "y": w.pose[1], "th": w.pose[2], "fx": w.fig[0], "fy": w.fig[1],
              "carried": bool(w.carried)}
        self.page.evaluate("([s, i]) => window.driver.setState(s, i)", [st, [[k, v] for k, v in ignite.items()]])

    def obs(self, kind: str, size: tuple[int, int]) -> bytes:
        url = self.page.evaluate("([k, w, h]) => window.driver.obs(k, w, h)", [kind, size[0], size[1]])
        self.renders += 1
        return base64.b64decode(url.split(",", 1)[1])

    def close(self) -> None:
        for fn in (self.browser.close, self._pw.stop, self.site.close):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass


class IgniteTracker:
    def __init__(self, world: World):
        self.w = world
        self.t: dict[int, float] = {}
        self.seen = np.zeros_like(world.fire.burning)
        self(world)
        for k in self.t:
            self.t[k] = -2.5          # the fire is already burning when the episode starts

    def __call__(self, world: World) -> None:
        new = world.fire.burning & ~self.seen
        for y, x in zip(*np.nonzero(new)):
            self.t[int(y) * world.room.w + int(x)] = round(float(world.t), 2)
        self.seen |= world.fire.burning


class SensorView3D(SensorView):
    """Same interface as SensorView; the frames are rendered from the 3D scene."""

    LIT_PX, SEM_PX = (640, 512), (480, 384)

    def __init__(self, world: World, scene: Scene3D, tracker: IgniteTracker):
        super().__init__(world)
        self.scene, self.tracker = scene, tracker

    def _push(self) -> None:
        self.scene.set_state(self._w, self.tracker.t)

    def frame(self) -> tuple[bytes, int]:
        self._push()
        png = self.scene.obs("lit", self.LIT_PX)
        self._w.gt_frames.append(self._w.truth())
        return png, len(self._w.gt_frames) - 1

    def semantic_png(self) -> bytes:
        self._push()
        return self.scene.obs("semantic", self.SEM_PX)

    def semantic_frame(self) -> tuple[bytes, int]:
        png = self.semantic_png()
        self._w.gt_frames.append(self._w.truth())
        return png, len(self._w.gt_frames) - 1


# ------------------------------------------------------------------------------------ models
class OfflineStandInVLM(FakeVLM):
    """No network. Decodes the segmentation frame; says so in its model name."""

    def perceive(self, role: str, image_png: bytes, meta: dict[str, Any]) -> CallResult:
        r = super().perceive(role, meta.get("semantic_png") or image_png, meta)
        return CallResult(r.text, 0, 0, 0.0, "offline-standin-vlm (colour decoder, not a neural net)")


class TolerantTransport(ChatTransport):
    """Retries once without response_format when a model rejects it (HTTP 400)."""

    def __init__(self, *a: Any, **k: Any):
        super().__init__(*a, **k)
        self.no_json: set[str] = set()

    def chat(self, model: str, messages: list[dict[str, Any]], max_tokens: int, json_mode: bool = False) -> dict[str, Any]:
        if model in self.no_json:
            json_mode = False
        try:
            return super().chat(model, messages, max_tokens, json_mode)
        except ApiError as e:
            if json_mode and "HTTP 400" in str(e):
                self.no_json.add(model)
                return super().chat(model, messages, max_tokens, False)
            raise


def _tiny_png() -> bytes:
    import io

    from PIL import Image, ImageDraw
    im = Image.new("RGB", (96, 96), (30, 30, 40))
    ImageDraw.Draw(im).ellipse([30, 30, 66, 66], fill=(36, 112, 255))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def vendor_of(model_id: str | None) -> str:
    if not model_id:
        return ""
    if model_id.startswith("nvidia/"):
        return "NVIDIA"
    return model_id.split("/")[0]


def choose_models(tr: ChatTransport, cfg: dict[str, Any]) -> dict[str, Any]:
    """Pick models by real calls. Ids come from GET /models; availability from a tiny live call."""
    data = tr.list_models().get("data", [])
    ids = [m.get("id", "") for m in data]
    print(f"endpoint ok: {len(ids)} models listed; NVIDIA ones: {[i for i in ids if i.lower().startswith('nvidia/')]}")
    report: dict[str, Any] = {"listed": len(ids), "probes": []}
    png64 = base64.b64encode(_tiny_png()).decode()

    def probe_vlm(mid: str) -> bool:
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": 'What colour is the circle? Reply as JSON {"colour":"..."}'},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png64}"}}]}]
        return _probe(mid, msgs, "vlm")

    def probe_text(mid: str) -> bool:
        return _probe(mid, [{"role": "user", "content": 'Reply with JSON {"ok":true}'}], "planner")

    def _probe(mid: str, msgs: list[dict[str, Any]], role: str) -> bool:
        t0 = time.time()
        try:
            r = tr.chat(mid, msgs, 200, json_mode=False)
            txt = ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            ok = bool(txt.strip())
            err = "" if ok else "empty reply"
        except ApiError as e:
            ok, err = False, str(e)[:120]
        report["probes"].append({"role": role, "model": mid, "ok": ok, "latency_s": round(time.time() - t0, 2), "error": err})
        print(f"  probe {role:8s} {mid}: {'OK' if ok else 'FAIL ' + err}")
        return ok

    vlm_c = [i for i in ids if i.lower().startswith("nvidia/") and re.search(r"(vl|vision|vila|neva|cosmos)", i.lower())]
    cfg_v = cfg["models"]["vlm_primary"]["id"]
    if cfg_v and cfg_v in ids and cfg_v not in vlm_c:
        vlm_c.append(cfg_v)
    vlm = next((m for m in vlm_c if probe_vlm(m)), None)
    pl_c = [cfg["models"][r]["id"] for r in ("planner", "planner_fast") if cfg["models"][r]["id"] in ids]
    pl_c += [i for i in ids if i.lower().startswith("nvidia/") and "nemotron" in i.lower() and i not in pl_c
             and not re.search(r"(vl|vision|embed|rerank|guard|safety|parse)", i.lower())][:3]
    planner = next((m for m in dict.fromkeys(pl_c) if probe_text(m)), None)
    report.update(vlm=vlm, planner=planner)
    return report


def apply_models(cfg: dict[str, Any], vlm: str, planner: str) -> bool:
    """Write chosen ids into cfg; return True when every role has a price."""
    priced = {m["id"]: m for m in cfg["models"].values() if m.get("price_in") is not None}
    for role in ("vlm_primary", "vlm_cheap"):
        p = priced.get(vlm, {})
        cfg["models"][role] = {"id": vlm, "price_in": p.get("price_in"), "price_out": p.get("price_out")}
    for role in ("planner", "planner_fast", "planner_escalation"):
        p = priced.get(planner, {})
        cfg["models"][role] = {"id": planner, "price_in": p.get("price_in"), "price_out": p.get("price_out")}
    return all(cfg["models"][r].get("price_in") is not None for r in ROLES)


# ------------------------------------------------------------------------------------ run
def _prepare_site(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    dist = SCENE_DIR / "dist"
    for name in ("app.js", "driver.js"):
        shutil.copy(dist / name, run_dir / name)
    shutil.copy(SCENE_DIR / "driver.html", run_dir / "driver.html")
    tpl = (SCENE_DIR / "index.html").read_text(encoding="utf-8")
    (run_dir / "index.html").write_text('<!doctype html>\n<meta charset="utf-8">\n'
                                        '<meta name="viewport" content="width=device-width,initial-scale=1">\n' + tpl,
                                        encoding="utf-8")
    shutil.copytree(SCENE_DIR / "assets", run_dir / "assets", dirs_exist_ok=True)


def _trim(text: str, n: int = 2400) -> str:
    return text if len(text) <= n else text[:n] + " ...[trimmed]"


def build_mission(args: argparse.Namespace, room: Room, world: World, frames: list[list[float]],
                  ignite: dict[int, float], log: dict[str, Any], budget: Budget, cfg: dict[str, Any],
                  live: bool, report: dict[str, Any] | None, price_known: bool) -> dict[str, Any]:
    ev = log.get("events", [])
    pick_t = next((e["t"] for e in ev if e["type"] == "pickup" and e.get("ok")), None)
    rel_t = next((e["t"] for e in ev if e["type"] == "release"), None)
    # ---- replay timeline: scan hold, driving frames, grasp hold, release hold, outro hold
    out: list[list[float]] = []
    T = 0.0

    def hold(base: list[float], seconds: float, code: int, carried: int, fig: tuple[float, float] | None = None) -> None:
        nonlocal T
        n = int(round(seconds / DT))
        for k in range(n):
            f = list(base)
            f[0] = round(T, 3)
            f[7] = carried
            if fig:
                f[5], f[6] = fig
            f += [code, round((k + 1) / n, 4)]
            out.append(f)
            T += DT

    t2T: list[tuple[float, float]] = []
    f0 = frames[0]
    hold([0.0, f0[0], *f0[1:]], HOLD_SCAN, 3, 0)
    picked = released = False
    prev_car = 0
    for i, f in enumerate(frames):
        t, x, y, th, fx, fy, car = f
        if not picked and car and not prev_car and pick_t is not None:
            p = frames[i - 1] if i else f
            hold([0.0, p[0], p[1], p[2], p[3], p[4], p[5], 0], HOLD_PICK, 1, 0,
                 fig=(x + cfg["sim"]["carry_offset_m"] * math.cos(th), y + cfg["sim"]["carry_offset_m"] * math.sin(th)))
            picked = True
        if picked and not released and prev_car and not car:
            p = frames[i - 1]
            hold([0.0, p[0], p[1], p[2], p[3], p[4], p[5], 0], HOLD_REL, 2, 0)
            released = True
        out.append([round(T, 3), round(t, 3), round(x, 3), round(y, 3), round(th, 4), round(fx, 3), round(fy, 3), int(car), 0, 0])
        t2T.append((t, T))
        T += DT
        prev_car = car
    last = frames[-1]
    lf = [0.0, last[0], last[1], last[2], last[3], last[4], last[5], 0]
    if picked and not released and last[6]:
        hold(lf, HOLD_REL, 2, 0)
    if world.outcome == "rescued":
        hold(lf, HOLD_END, 4, 0)

    def to_T(t: float, k: int = 0) -> float:
        if t <= 0.05:
            return round(0.7 + 0.8 * k, 2)           # first perception / plan happen during the intro scan
        for ts, TT in t2T:
            if ts >= t - 1e-6:
                return round(TT, 2)
        return round(t2T[-1][1], 2)

    # ---- calls: log entries zipped with the budget ledger rows (same order)
    rows = list(budget.rows)
    vrows = [r for r in rows if r["kind"] == "vlm"]
    prows = [r for r in rows if r["kind"] in ("plan", "repair")]

    def src_for(model: str | None, role: str | None) -> str:
        if role == "safety_astar":
            return "safety_fallback"
        if not live:
            return "offline_standin"
        return "nvidia_model" if (model or "").startswith("nvidia/") else "nebius_model"

    def costs(r: dict[str, Any] | None) -> dict[str, Any]:
        if not r:
            return {"latency_s": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0}
        z = not live
        return {"latency_s": 0 if z else r["latency_s"], "tokens_in": 0 if z else r["tokens_in"],
                "tokens_out": 0 if z else r["tokens_out"], "cost_usd": 0 if z else r["cost_usd"]}

    calls: list[dict[str, Any]] = []
    k0 = 0
    for i, p in enumerate(log.get("perceptions", [])):
        r = vrows[i] if i < len(vrows) else None
        c = {"kind": "perception", "t": p["t"], "T": to_T(p["t"], k0 if p["t"] <= 0.05 else 0), "role": p["role"],
             "model": p["model"], "source": src_for(p["model"], p["role"]),
             "status": "accepted" if p.get("valid") else "invalid_json", "reasons": [p["error"]] if p.get("error") else [],
             "raw": _trim(p.get("raw", "")), "parsed": p.get("parsed"), "grounding": p.get("grounding"),
             "model_reported": p.get("model_reported"),
             "frame": f"frames/{p['frame_file']}" if p.get("frame_file") else None,
             "summary": f"figurine/exit located ({p.get('grounding')})" if p.get("valid") else "reply rejected", **costs(r)}
        k0 += 1 if p["t"] <= 0.05 else 0
        calls.append(c)
    k1, pi, plans_out = 1, 0, []
    for p in log.get("plans", []):
        safety = p.get("role") == "safety_astar"
        r = None if safety else (prows[pi] if pi < len(prows) else None)
        pi += 0 if safety else 1
        tt = p["t"]
        T_ = to_T(tt, k1 if tt <= 0.05 else 0)
        k1 += 1 if tt <= 0.05 else 0
        status = "accepted" if p.get("accepted") else f"rejected:{p.get('rejected_by', '?')}"
        calls.append({"kind": "safety_route" if safety else "plan", "t": tt, "T": T_, "role": p.get("role"),
                      "model": None if safety else p.get("model"), "source": src_for(p.get("model"), p.get("role")),
                      "status": status, "reasons": p.get("reasons", []), "raw": _trim(p.get("raw", "")) if not safety else "",
                      "parsed": p.get("plan"), "frame": None, "counted": not safety,
                      "summary": (p.get("plan") or {}).get("rationale", ""), **costs(r)})
        if p.get("accepted") and p.get("plan"):
            wps = p["plan"]["waypoints"]
            plans_out.append({"t": tt, "T": T_, "source": src_for(p.get("model"), p.get("role")),
                              "rationale": p["plan"].get("rationale", ""), "waypoints": wps})
    for e in ev:
        if e["type"] in ("vlm_error", "plan_error", "vlm_blocked", "plan_blocked"):
            calls.append({"kind": "perception" if e["type"].startswith("vlm") else "plan", "t": e["t"], "T": to_T(e["t"]),
                          "role": e.get("role"), "model": None, "source": "none", "status": e["type"],
                          "reasons": [e.get("error") or e.get("reason", "")], "raw": "", "parsed": None, "frame": None,
                          "summary": "", "latency_s": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0, "counted": False})
    seen_ev: set[tuple[str, str]] = set()
    kept = []
    for c in sorted(calls, key=lambda c: c["T"]):     # a cap that blocks every later call is listed once
        if c["source"] == "none":
            key = (c["kind"], c["status"])
            if key in seen_ev:
                continue
            seen_ev.add(key)
        kept.append(c)
    calls = kept
    # waypoint completion times (for route drawing)
    arr = np.array([[f[1], f[2], f[3], f[0]] for f in out if f[8] == 0])
    for pl in plans_out:
        start_i = int(np.searchsorted(arr[:, 3], pl["T"]))
        cur = start_i
        for wp in pl["waypoints"]:
            tol = 0.3 if wp.get("action") != "move" else 0.2
            hit = None
            for j in range(cur, len(arr)):
                if math.hypot(arr[j, 0] - wp["x"], arr[j, 1] - wp["y"]) <= tol:
                    hit = j
                    break
            if hit is None:
                break
            wp["done_by"] = round(float(arr[hit, 3]), 2)
            cur = hit
    pv = [c for c in calls if c["kind"] == "perception" and c.get("model")]
    pl = [c for c in calls if c["kind"] == "plan" and c.get("model")]
    models = {"perception": {"id": pv[0]["model"] if pv else cfg["models"]["vlm_primary"]["id"], "vendor": vendor_of(pv[0]["model"]) if (pv and live) else "offline stand-in"},
              "planner": {"id": pl[0]["model"] if pl else cfg["models"]["planner"]["id"], "vendor": vendor_of(pl[0]["model"]) if (pl and live) else "offline stand-in"}}
    n_calls = sum(1 for c in calls if c.get("counted", True) and c["source"] != "none")
    return {
        "schema": "crab-rescue-mission/1",
        "meta": {"mode": "nebius_live" if live else "offline_standin", "seed": args.seed, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "outcome": world.outcome, "rescued_t": round(world.rescued_t, 2) if world.rescued_t else None,
                 "models": models, "price_known": price_known if live else True,
                 "budget": {"calls": n_calls, "usd": round(sum(c["cost_usd"] for c in calls), 6), "cap_usd": cfg["caps"]["max_total_usd"],
                            "tokens_in": sum(c["tokens_in"] for c in calls), "tokens_out": sum(c["tokens_out"] for c in calls)},
                 "preflight": report,
                 "locomotion": "procedural joint-by-joint gait animation on the Jumper rig; the body is moved by a kinematic simulator (the locomotion.onnx policy is not connected)",
                 "grasp": "visual grasp: claw IK, the figurine is attached to the body; contact physics is not simulated",
                 "fire": "cellular automaton, not combustion physics", "dt": DT},
        "room": room_to_json(room), "ignite": [[k, v] for k, v in sorted(ignite.items())],
        "frames": out, "calls": calls, "plans": plans_out, "events": ev,
    }


def run_mission(args: argparse.Namespace) -> Path:
    cfg = load_config()
    cfg["caps"].update(max_vlm_calls_per_episode=args.max_vlm_calls, max_replans_per_episode=8, max_repairs=1,
                       max_escalations=0, max_total_usd=args.max_usd)
    cfg["agent"]["vlm_period_s"] = args.vlm_period
    live = args.live
    report: dict[str, Any] | None = None
    price_known = True
    transport = None
    if live:
        import os
        key = os.environ.get("NEBIUS_API_KEY", "")
        if not key:
            raise SystemExit("Set NEBIUS_API_KEY in the environment (it is never read from files or printed).")
        transport = TolerantTransport(os.environ.get("NEBIUS_BASE_URL", BASE_URL), key, timeout_s=90.0, max_retries=1)
        report = choose_models(transport, cfg)
        if not report.get("vlm") or not report.get("planner"):
            raise SystemExit(f"preflight failed (vlm={report.get('vlm')}, planner={report.get('planner')}); "
                             "nothing was run. See the probe lines above.")
        print(f"selected vision model : {report['vlm']}   ({vendor_of(report['vlm'])})")
        print(f"selected planner model: {report['planner']}   ({vendor_of(report['planner'])})")
        price_known = apply_models(cfg, report["vlm"], report["planner"])
        if not price_known:
            print("note: no price in config.yaml for a selected model; the dollar cap cannot be enforced "
                  "(call-count caps still apply) and cost is shown as unknown.")
    run_dir = Path(args.out) / (f"{time.strftime('%Y%m%d_%H%M%S')}_seed{args.seed}_{'live' if live else 'offline'}")
    _prepare_site(run_dir)
    (run_dir / "frames").mkdir(exist_ok=True)
    gw, gh = cfg["sim"]["grid"]
    room = generate_room(args.seed, gw, gh, cfg["sim"]["cell_m"])
    world = World(room, cfg, args.seed)
    tracker = IgniteTracker(world)
    frames: list[list[float]] = [[0.0, world.pose[0], world.pose[1], world.pose[2], world.fig[0], world.fig[1], int(world.carried)]]

    def rec(w: World) -> None:
        tracker(w)
        frames.append([w.t, w.pose[0], w.pose[1], w.pose[2], w.fig[0], w.fig[1], int(w.carried)])

    world.observers.append(rec)
    scene = Scene3D(run_dir)
    try:
        scene.load_room(room)
        sensor = SensorView3D(world, scene, tracker)
        budget = Budget(cfg, ledger_path=run_dir / "ledger.jsonl", fake=not live, allow_unpriced=live and not price_known)
        budget.new_episode(f"seed{args.seed}")
        router = Router(cfg["caps"]["max_escalations"], cfg["agent"]["low_confidence"])
        if live:
            ids = {r: cfg["models"][r]["id"] for r in ROLES}
            mt = cfg["caps"]["max_tokens_per_call"]
            vlm, planner = NebiusVLM(transport, ids, mt, system=VLM_SYSTEM_3D), NebiusPlanner(transport, ids, mt)
        else:
            vlm, planner = OfflineStandInVLM((gw, gh), seed=args.seed), FakePlanner(seed=args.seed)
        t0 = time.time()
        log = run_agent(sensor, vlm, planner, budget, router, cfg, frame_dir=run_dir / "frames", log={})
        print(f"episode: {world.outcome} at sim t={world.t:.1f}s, wall {time.time() - t0:.0f}s, "
              f"{scene.renders} frames rendered, page errors: {scene.errors[:2] or 'none'}")
    finally:
        scene.close()
    mission = build_mission(args, room, world, frames, tracker.t, log, budget, cfg, live, report, price_known)
    (run_dir / "mission.json").write_text(json.dumps(mission, separators=(",", ":")), encoding="utf-8")
    b = mission["meta"]["budget"]
    print(f"saved {run_dir / 'mission.json'}\n  calls {b['calls']}, tokens {b['tokens_in']}/{b['tokens_out']}, "
          f"cost ${b['usd']:.5f} (cap ${b['cap_usd']}), outcome {mission['meta']['outcome']}")
    return run_dir


def record_video(run_dir: Path, fps: int = 24, size: tuple[int, int] = (1280, 720)) -> Path:
    from playwright.sync_api import sync_playwright
    ff = shutil.which("ffmpeg")
    if not ff:
        raise SystemExit("ffmpeg is needed for --video")
    site = Site(run_dir)
    out = run_dir / "mission.mp4"
    with sync_playwright() as pw:
        br = _launch(pw)
        pg = br.new_page(viewport={"width": size[0], "height": size[1]})
        errs: list[str] = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(f"http://127.0.0.1:{site.port}/index.html?rec=1&mission=mission.json")
        pg.wait_for_function("window.__rec && window.__rec.ready === true", timeout=120000)
        dur = pg.evaluate("window.__rec.duration")
        n = int(dur * fps)
        proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-vcodec", "mjpeg",
                                 "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                                 "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        t0 = time.time()
        for i in range(n + 1):
            pg.evaluate("t => window.__rec.seek(t)", min(i / fps, dur))
            proc.stdin.write(pg.screenshot(type="jpeg", quality=92))
            if i % 120 == 0:
                print(f"  video {i}/{n} frames ({time.time() - t0:.0f}s)", flush=True)
        proc.stdin.close()
        proc.wait()
        br.close()
    site.close()
    print(f"video: {out} ({dur:.1f}s, {size[0]}x{size[1]}, {fps} fps); page errors: {errs[:2] or 'none'}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crab_rescue.mission3d")
    ap.add_argument("--live", action="store_true", help="call the real Nebius models (NEBIUS_API_KEY)")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="runs3d")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--max-usd", type=float, default=0.05)
    ap.add_argument("--max-vlm-calls", type=int, default=4)
    ap.add_argument("--vlm-period", type=float, default=25.0, help="seconds of sim time between vision calls")
    args = ap.parse_args(argv)
    run_dir = run_mission(args)
    if args.video:
        record_video(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
