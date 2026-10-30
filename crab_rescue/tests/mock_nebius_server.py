"""Local OpenAI-compatible MOCK server for checking the --live code path without network, key or cost.

    python -m crab_rescue.tests.mock_nebius_server            # prints the base URL, serves until Ctrl+C
    NEBIUS_BASE_URL=http://127.0.0.1:PORT/v1/ NEBIUS_API_KEY=mock python -m crab_rescue.mission3d --live

The model ids it advertises are fake ("mock/..."), so a run against it is labelled non-NVIDIA by the runner.
It is NOT a model and says nothing about how real NVIDIA models behave.
"""
from __future__ import annotations

import base64
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from crab_rescue.clients import FakePlanner, FakeVLM

import os

# Default ids are "mock/...". For exercising the preflight filter (which only considers nvidia/* ids) a throw-away
# test can set MOCK_PREFIX=nvidia; such a run must be discarded, it is not an NVIDIA result.
_P = os.environ.get("MOCK_PREFIX", "mock")
MODELS = [f"{_P}/vision-vl", f"{_P}/planner-nemotron"]


def make_server(port: int = 0) -> HTTPServer:
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
            self._send(200, {"data": [{"id": m} for m in MODELS]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            user = body["messages"][-1]["content"]
            if isinstance(user, list):
                part = next(p for p in user if p["type"] == "image_url")
                png = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                q = next((p["text"] for p in user if p["type"] == "text"), "")
                text = '{"colour":"red"}' if "colour" in q and "circle" in q else FakeVLM(seed=1).perceive("vlm_primary", png, {}).text
            else:
                try:
                    ctx = json.loads(user)
                    w, h = ctx["room_size_m"]
                    c = ctx["cell_m"]
                    ctx["grid"] = [int(w / c), int(h / c), c]
                    text = FakePlanner()._good_plan(ctx)
                except (ValueError, KeyError):
                    text = '{"ok":true}'
            self._send(200, {"choices": [{"message": {"content": text}}],
                             "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    return HTTPServer(("127.0.0.1", port), H)


if __name__ == "__main__":
    srv = make_server(int(sys.argv[1]) if len(sys.argv) > 1 else 8799)
    print(f"MOCK server (not NVIDIA): http://127.0.0.1:{srv.server_port}/v1/", flush=True)
    srv.serve_forever()
