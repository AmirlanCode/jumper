"""python -m crab_rescue.app  ->  open http://127.0.0.1:8765

A local web app: a live crab-rescue simulation in the browser, driven by the real agent loop.
Run it on a machine that can reach Nebius. Two modes:

* offline stand-in models ($0, no key);
* NVIDIA models served by Nebius Token Factory: paste the key in the page (kept in this
  process's memory only, never written to disk, never sent back to the browser), load the model
  list from /v1/models, pick models, run.

The server binds to 127.0.0.1 only. The simulation pauses while a model call is in flight.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .agent import run_agent
from .baselines import run_baselines_entry
from .budget import Budget
from .clients import DiskCache, FakePlanner, FakeVLM, ModelCallError
from .config import ROLES, load_config
from .discover import probe
from .evaluator import episode_metrics
from .executor import SensorView, World
from .nebius import ApiError, ChatTransport, NebiusPlanner, NebiusVLM
from .router import Router
from .scene import generate_room

PAGE = Path(__file__).with_name("app.html")

# Token Factory public catalog prices checked on 2026-10-29. Keep model ids in sync with
# the account's /v1/models response; unknown ids need an explicit UI override.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "openbmb/MiniCPM-V-4_5": (0.658, 1.11),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
}


class Stopped(Exception):
    pass


class Session:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.creds: dict[str, str | None] = {"key": None, "base": None}
        self.reset()

    def reset(self) -> None:
        self.status = "idle"            # idle | running | done | error
        self.error: str | None = None
        self.stop = False
        self.paused = False
        self.speed = 4.0
        self.snap: dict[str, Any] | None = None
        self.static: dict[str, Any] | None = None
        self.log: dict[str, Any] = {"perceptions": [], "plans": [], "events": []}
        self.pngs: list[str] = []
        self.vmeta: list[dict[str, Any]] = []
        self.pmeta: list[dict[str, Any]] = []
        self.busy: dict[str, Any] | None = None
        self.budget: Budget | None = None
        self.metrics: dict[str, Any] | None = None
        self.mode = "offline"
        self.method = "agent"
        self.priced = False


S = Session()


# ---------------------------------------------------------------------------------------
# tracing wrappers: expose "the model is thinking" and keep what each call saw
# ---------------------------------------------------------------------------------------
class TraceVLM:
    def __init__(self, inner: Any, ids: dict[str, str]):
        self.inner, self.ids = inner, ids

    def perceive(self, role: str, png: bytes, meta: dict[str, Any]):
        S.busy = {"kind": "vision", "role": role, "model": self.ids.get(role, "stand-in"), "since": time.time()}
        t0 = time.time()
        try:
            res = self.inner.perceive(role, png, meta)
        except ModelCallError as e:
            S.pngs.append(base64.b64encode(png).decode())
            S.vmeta.append({"latency_s": round(time.time() - t0, 2), "error": str(e)[:200]})
            raise
        finally:
            S.busy = None
        S.pngs.append(base64.b64encode(png).decode())
        S.vmeta.append({"latency_s": round(res.latency_s or time.time() - t0, 2),
                        "tokens_in": res.tokens_in, "tokens_out": res.tokens_out})
        return res


class TracePlanner:
    def __init__(self, inner: Any, ids: dict[str, str]):
        self.inner, self.ids = inner, ids

    def plan(self, role: str, context: dict[str, Any], feedback: list[str]):
        S.busy = {"kind": "planning", "role": role, "model": self.ids.get(role, "stand-in"), "since": time.time()}
        t0 = time.time()
        try:
            res = self.inner.plan(role, context, feedback)
        except ModelCallError as e:
            S.pmeta.append({"latency_s": round(time.time() - t0, 2), "error": str(e)[:200]})
            raise
        finally:
            S.busy = None
        S.pmeta.append({"latency_s": round(res.latency_s or time.time() - t0, 2),
                        "tokens_in": res.tokens_in, "tokens_out": res.tokens_out})
        return res


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------
def _snapshot(w: World) -> dict[str, Any]:
    ys, xs = w.fire.burning.nonzero()
    return {"t": round(w.t, 2), "pose": [round(v, 3) for v in w.pose], "fig": [round(v, 3) for v in w.fig],
            "carried": w.carried, "fire": [[int(x), int(y)] for x, y in zip(xs, ys)],
            "done": w.done, "outcome": w.outcome, "exposure": round(w.fire_exposure_s, 1),
            "collisions": w.collisions}


def start_run(p: dict[str, Any]) -> None:
    cfg = load_config()
    cfg["caps"]["max_tokens_per_call"] = int(p.get("max_tokens", 3000))
    for k in ("max_vlm_calls_per_episode", "max_replans_per_episode"):
        if p.get(k):
            cfg["caps"][k] = int(p[k])
    seed, method, mode = int(p.get("seed", 0)), p.get("method", "agent"), p.get("mode", "offline")
    gw, gh = cfg["sim"]["grid"]
    room = generate_room(seed, gw, gh, cfg["sim"]["cell_m"])
    world = World(room, cfg, seed)
    S.reset()
    S.mode, S.method, S.speed = mode, method, float(p.get("speed", 4.0))
    S.static = {"grid": room.grid.tolist(), "cell": room.cell, "w": room.w, "h": room.h,
                "exit_xy": room.exit_xy, "seed": seed}

    ids: dict[str, str] = {}
    if mode == "nvidia" and method == "agent":
        if not S.creds["key"]:
            raise ValueError("Connect first: paste the key and press 'Load models'.")
        v, pl = p.get("vlm"), p.get("planner")
        if not v or not pl:
            raise ValueError("Pick a vision model and a planner model.")
        ids = {"vlm_primary": v, "vlm_cheap": p.get("vlm_cheap") or v, "planner": pl,
               "planner_fast": p.get("planner_fast") or pl, "planner_escalation": p.get("planner_escalation") or pl}
        transport = ChatTransport(S.creds["base"] or "", S.creds["key"] or "", max_retries=2)
        cache = DiskCache(Path("runs") / ".app_cache")
        mt = cfg["caps"]["max_tokens_per_call"]
        vlm: Any = NebiusVLM(transport, ids, mt, cache)
        planner: Any = NebiusPlanner(transport, ids, mt, cache)
        pi, po = p.get("price_in"), p.get("price_out")
        override = (float(pi), float(po)) if pi not in (None, "") and po not in (None, "") else None
        for r in ROLES:
            price = MODEL_PRICES.get(ids[r], override)
            cfg["models"][r].update(id=ids[r],
                                    price_in=price[0] if price else None,
                                    price_out=price[1] if price else None)
        S.priced = all(cfg["models"][r]["price_in"] is not None and
                       cfg["models"][r]["price_out"] is not None for r in ROLES)
        budget = Budget(cfg, fake=False, allow_unpriced=True)
    else:
        vlm, planner = FakeVLM((gw, gh), seed=seed), FakePlanner(seed=seed)
        budget = Budget(cfg, fake=True)
        ids = {r: "stand-in (offline)" for r in ROLES}
    S.budget = budget
    router = Router(cfg["caps"]["max_escalations"], cfg["agent"]["low_confidence"])
    step = cfg["sim"]["dt"]

    def observer(w: World) -> None:
        while S.paused and not S.stop:
            time.sleep(0.05)
        if S.stop:
            raise Stopped()
        S.snap = _snapshot(w)
        time.sleep(step / max(S.speed, 0.25))

    world.observers.append(observer)
    S.snap = _snapshot(world)
    S.status = "running"

    def work() -> None:
        try:
            if method == "agent":
                run_agent(SensorView(world), TraceVLM(vlm, ids), TracePlanner(planner, ids), budget, router,
                          cfg, log=S.log)
                S.metrics = episode_metrics(world, "agent", seed, budget, S.log)
            else:
                run_baselines_entry(method, world, cfg)
                S.metrics = episode_metrics(world, method, seed, None, None, uses_ground_truth=True)
            S.snap = _snapshot(world)
            S.status = "done"
        except Stopped:
            S.status = "idle"
        except Exception as e:  # noqa: BLE001 - shown to the user, never contains the key
            S.status, S.error = "error", f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            S.busy = None

    threading.Thread(target=work, daemon=True).start()


def state_payload(q: dict[str, int]) -> dict[str, Any]:
    lg = S.log
    per = [dict(e, **(S.vmeta[i] if i < len(S.vmeta) else {}), png=S.pngs[i] if i < len(S.pngs) else None)
           for i, e in enumerate(lg["perceptions"]) if i >= q["p"]]
    pl = [dict(e, **(S.pmeta[i] if i < len(S.pmeta) else {}))
          for i, e in enumerate(lg["plans"]) if i >= q["l"]]
    b = S.budget
    bud = None
    if b is not None:
        bud = {"calls": b.total_calls, "vlm_calls": b.vlm_calls, "replans": b.replans,
               "escalations": b.escalations, "tokens_in": b.ep_tokens_in, "tokens_out": b.ep_tokens_out,
               "cost_usd": round(b.ep_cost, 6), "priced": S.priced or b.fake}
    busy = dict(S.busy, waited_s=round(time.time() - S.busy["since"], 1)) if S.busy else None
    return {"status": S.status, "error": S.error, "mode": S.mode, "method": S.method, "static": S.static,
            "snap": S.snap, "busy": busy, "paused": S.paused, "speed": S.speed,
            "perceptions": per, "plans": pl, "events": lg["events"][q["e"]:],
            "n": {"p": len(lg["perceptions"]), "l": len(lg["plans"]), "e": len(lg["events"])},
            "budget": bud, "metrics": S.metrics}


def models_hint(ids: list[str]) -> dict[str, list[str]]:
    low = [(i, name.lower()) for i, name in ((x, x) for x in ids)]
    vis = [i for i, nm in low if any(k in nm for k in (
        "-vl", "vl-", "vision", "cosmos", "omni", "vlm", "minicpm-v"))]
    txt = [i for i, nm in low if i not in vis and any(k in nm for k in ("nemotron", "llama", "qwen", "deepseek"))]
    return {"vision": vis, "text": txt}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a: Any) -> None:
        pass

    def _json(self, code: int, obj: Any) -> None:
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:  # noqa: N802
        path, _, qs = self.path.partition("?")
        if path == "/":
            b = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif path == "/api/state":
            q = {k: int(v) for k, v in (kv.split("=") for kv in qs.split("&") if "=" in kv)}
            self._json(200, state_payload({"p": q.get("p", 0), "l": q.get("l", 0), "e": q.get("e", 0)}))
        elif path == "/api/info":
            self._json(200, {"connected": bool(S.creds["key"]), "base": S.creds["base"]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        origin = self.headers.get("Origin")
        if origin and origin.split("://")[-1] != self.headers.get("Host"):
            return self._json(403, {"error": "cross-origin request refused"})
        n = int(self.headers.get("Content-Length") or 0)
        p = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/connect":
                key, base = (p.get("key") or S.creds["key"] or ""), (p.get("base") or "").strip()
                if not key or not base:
                    return self._json(400, {"error": "key and base URL are required"})
                data = ChatTransport(base, key, max_retries=1).list_models().get("data", [])
                S.creds.update(key=key, base=base)
                ids = sorted(m.get("id", "?") for m in data)
                return self._json(200, {"ok": True, "ids": ids, "hints": models_hint(ids)})
            if self.path == "/api/probe":
                if not S.creds["key"]:
                    return self._json(400, {"error": "connect first"})
                t = ChatTransport(S.creds["base"] or "", S.creds["key"] or "", max_retries=1)
                return self._json(200, probe(t, p["role"], p["model"]))
            if self.path == "/api/start":
                if S.status == "running":
                    return self._json(409, {"error": "a run is already in progress"})
                start_run(p)
                return self._json(200, {"ok": True})
            if self.path == "/api/pause":
                S.paused = not S.paused
                return self._json(200, {"paused": S.paused})
            if self.path == "/api/speed":
                S.speed = float(p.get("speed", 4))
                return self._json(200, {"speed": S.speed})
            if self.path == "/api/stop":
                S.stop = True
                return self._json(200, {"ok": True})
            return self._json(404, {"error": "not found"})
        except (ApiError, ValueError, KeyError) as e:
            return self._json(400, {"error": str(e)[:300]})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crab_rescue.app")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the page in your browser")
    a = ap.parse_args(argv)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}"
    print(f"Crab Rescue app running at {url}  (Ctrl-C to stop)")
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        S.stop = True
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
