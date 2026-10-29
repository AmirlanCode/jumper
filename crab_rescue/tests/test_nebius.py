"""Real-client path, exercised against a local mock server (no network, no key, no cost)."""
import base64
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml

from crab_rescue import discover
from crab_rescue import run as run_mod
from crab_rescue.clients import DiskCache, FakePlanner, FakeVLM
from crab_rescue.config import ROLES, load_config
from crab_rescue.nebius import ApiError, ChatTransport, NebiusPlanner, NebiusVLM

SECRET = "v1.TESTKEY-" + "x" * 60


class Mock:
    def __init__(self):
        self.requests, self.fail_next, self.require_key = [], 0, SECRET
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.headers.get("Authorization") != f"Bearer {mock.require_key}":
                    return self._send(401, {"error": "bad key"})
                self._send(200, {"data": [{"id": "org/Vision-A"}, {"id": "org/Text-B", "extra": 1}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                      "body": body})
                if self.headers.get("Authorization") != f"Bearer {mock.require_key}":
                    return self._send(401, {"error": "bad key"})
                if mock.fail_next > 0:
                    mock.fail_next -= 1
                    return self._send(500, {"error": "boom"})
                msgs = body["messages"]
                user = msgs[-1]["content"]
                if isinstance(user, list):                                   # image request
                    part = next(p for p in user if p["type"] == "image_url")
                    png = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                    text_q = next(p["text"] for p in user if p["type"] == "text")
                    if "colour is the square" in text_q:
                        text = "Red"
                    else:
                        text = FakeVLM(seed=1).perceive("vlm_primary", png, {}).text
                else:
                    try:
                        ctx = json.loads(user)
                        w, h = ctx["room_size_m"]
                        c = ctx["cell_m"]
                        ctx["grid"] = [int(w / c), int(h / c), c]
                        text = "thinking {not json} ... " + FakePlanner()._good_plan(ctx)
                    except (ValueError, KeyError):
                        text = "OK"
                self._send(200, {"choices": [{"message": {"content": text}}],
                                 "usage": {"prompt_tokens": 111, "completion_tokens": 22}})

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1/"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


@pytest.fixture()
def mock():
    m = Mock()
    yield m
    m.close()


def test_transport_format_retries_and_key_never_leaks(mock):
    t = ChatTransport(mock.url, SECRET, max_retries=2)
    mock.fail_next = 1                                   # one 500, then success
    r = t.chat("m", [{"role": "user", "content": "hi"}], max_tokens=7)
    assert r["usage"]["prompt_tokens"] == 111 and len(mock.requests) == 2
    last = mock.requests[-1]
    assert last["path"] == "/v1/chat/completions" and last["body"]["max_tokens"] == 7
    assert last["body"]["model"] == "m" and last["auth"] == f"Bearer {SECRET}"
    bad = ChatTransport(mock.url, "wrong-key", max_retries=0)
    with pytest.raises(ApiError) as e:
        bad.chat("m", [], 5)
    assert "401" in str(e.value) and "wrong-key" not in str(e.value)
    mock.fail_next = 99                                   # always failing -> bounded retries, no hang
    n = len(mock.requests)
    with pytest.raises(ApiError):
        ChatTransport(mock.url, SECRET, max_retries=1).chat("m", [], 5)
    assert len(mock.requests) - n == 2


def test_vlm_and_planner_clients_cache_and_parse(mock, tmp_path):
    from crab_rescue.render import png_bytes, render_world
    from crab_rescue.scene import generate_room
    import numpy as np
    room = generate_room(4)
    png = png_bytes(render_world(room, np.zeros((room.h, room.w), bool), room.start, room.figurine))
    ids = {r: f"org/{r}" for r in ROLES}
    cache = DiskCache(tmp_path)
    vlm = NebiusVLM(ChatTransport(mock.url, SECRET), ids, 500, cache)
    a = vlm.perceive("vlm_cheap", png, {})
    assert a.tokens_in == 111 and a.model == "org/vlm_cheap" and '"figurine"' in a.text
    body = mock.requests[-1]["body"]
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    n = len(mock.requests)
    b = vlm.perceive("vlm_cheap", png, {})
    assert len(mock.requests) == n and b.tokens_in == 0 and "(cached)" in b.model   # disk cache hit
    pl = NebiusPlanner(ChatTransport(mock.url, SECRET), ids, 500)
    from crab_rescue.schemas import parse_perception, parse_plan
    per = parse_perception(a.text)
    ctx = {"grid": [20, 16, 0.5], "pose": list(room.start), "t": 0.0, "carrying": False,
           "perception": per.model_dump(), "figurine": None, "fire_cells_per_s": 0.0}
    res = pl.plan("planner", ctx, [])
    assert parse_plan(res.text, (10, 8)).waypoints            # reasoning chatter before the JSON tolerated


def test_discover_dry_run_then_probe(mock, monkeypatch, tmp_path, capsys):
    assert discover.main(["--probe", "vlm_primary=org/Vision-A"]) == 0
    assert "DRY RUN" in capsys.readouterr().out and not mock.requests
    monkeypatch.setenv("NEBIUS_API_KEY", SECRET)
    monkeypatch.setenv("NEBIUS_BASE_URL", mock.url)
    out = tmp_path / "d.json"
    rc = discover.main(["--yes", "--probe", "vlm_primary=org/Vision-A", "planner=org/Text-B",
                        "planner_fast=org/not-listed", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and SECRET not in text and SECRET not in out.read_text()
    d = json.loads(out.read_text())
    assert d["model_ids"] == ["org/Text-B", "org/Vision-A"] and d["extra_fields_seen"] == ["extra"]
    by_role = {p["role"]: p for p in d["probes"]}
    assert by_role["vlm_primary"]["image"]["correct"] is True and "image" not in by_role["planner"]
    assert "planner_fast" not in by_role                    # id not in /models is skipped, not guessed


def test_real_episode_end_to_end_against_mock(mock, monkeypatch, tmp_path):
    cfg = load_config()
    for r in ROLES:
        cfg["models"][r].update(id=f"org/{r}", price_in=1.0, price_out=2.0)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("NEBIUS_API_KEY", SECRET)
    monkeypatch.setenv("NEBIUS_BASE_URL", mock.url)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / ".dryrun_done").touch()
    rc = run_mod.main(["--seeds", "1", "--methods", "agent", "--no-video", "--config", str(cfg_path),
                       "--out", str(runs)])
    assert rc == 0
    run = next(p for p in runs.iterdir() if p.name[0].isdigit())
    rows = [json.loads(x) for x in (run / "ledger.jsonl").read_text().splitlines()]
    assert rows and all(r["fake"] is False for r in rows)
    assert sum(r["cost_usd"] for r in rows) == pytest.approx(
        sum((r["tokens_in"] * 1.0 + r["tokens_out"] * 2.0) / 1e6 for r in rows))
    m = json.loads((run / "ep_0" / "agent" / "metrics.json").read_text())
    assert m["fake_models"] is False and m["cost_usd"] > 0
    for p in run.rglob("*"):                                   # the key is nowhere on disk
        if p.is_file() and p.suffix in {".json", ".jsonl", ".md"}:
            assert SECRET not in p.read_text(errors="ignore")


def test_agent_survives_api_errors(mock):
    from crab_rescue.agent import run_agent
    from crab_rescue.budget import Budget
    from crab_rescue.executor import SensorView, World
    from crab_rescue.router import Router
    from crab_rescue.scene import generate_room
    cfg = copy.deepcopy(load_config())
    cfg["sim"]["max_episode_s"] = 20.0
    room = generate_room(1, *cfg["sim"]["grid"], cfg["sim"]["cell_m"])
    w = World(room, cfg, 1)
    mock.fail_next = 10**6                                    # every call fails
    ids = {r: "m" for r in ROLES}
    t = ChatTransport(mock.url, SECRET, max_retries=0)
    log = run_agent(SensorView(w), NebiusVLM(t, ids, 100), NebiusPlanner(t, ids, 100),
                    Budget(cfg, fake=True), Router(2), cfg)
    assert w.done and any(e["type"] == "vlm_error" for e in log["events"])
