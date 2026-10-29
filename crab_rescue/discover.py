"""python -m crab_rescue.discover [--yes] [--probe role=MODEL_ID ...]

Phase-0 model discovery, to run where the network reaches Nebius (your Mac):

1. GET {base}/models and list the ids (the ONLY legitimate source of model ids).
2. For each `--probe role=id`, send one tiny text request and, for vision roles, one tiny image
   request (a 64x64 PNG with a red square), and report status, latency, token usage and whether
   the model answered correctly. Total cost is a few hundred tokens per probe.

Nothing is sent without --yes. The key comes from the environment and is never printed or saved.
Prices, context limits and "serverless" status are NOT in the API response this script relies
on: read them from the Token Factory console and put them in config.yaml.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .config import ROLES
from .nebius import ApiError, ChatTransport, env_credentials

VISION_ROLES = ("vlm_primary", "vlm_cheap")
HINTS = ("cosmos", "nemotron", "qwen")


def red_square_png() -> bytes:
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([16, 16, 48, 48], fill=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def probe(t: ChatTransport, role: str, model: str) -> dict:
    out: dict = {"role": role, "model": model}
    tests = [("text", [{"role": "user", "content": "Reply with exactly: OK"}], "ok")]
    if role in VISION_ROLES:
        b64 = base64.b64encode(red_square_png()).decode()
        tests.append(("image", [{"role": "user", "content": [
            {"type": "text", "text": "What colour is the square? Answer with one word."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}], "red"))
    for kind, messages, expect in tests:
        t0 = time.time()
        try:
            resp = t.chat(model, messages, max_tokens=200)
            text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            u = resp.get("usage") or {}
            out[kind] = {"ok": True, "correct": expect in text.lower(), "latency_s": round(time.time() - t0, 2),
                         "tokens_in": u.get("prompt_tokens"), "tokens_out": u.get("completion_tokens"),
                         "reply_head": text[:80]}
        except ApiError as e:
            out[kind] = {"ok": False, "error": str(e)[:200]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crab_rescue.discover")
    ap.add_argument("--yes", action="store_true", help="actually call the API (tiny, < $0.01)")
    ap.add_argument("--probe", nargs="*", default=[], metavar="ROLE=MODEL_ID",
                    help=f"roles: {', '.join(ROLES)}")
    ap.add_argument("--out", default="runs/discovery.json")
    a = ap.parse_args(argv)
    plan = ["GET /models"] + [f"probe {p}" for p in a.probe]
    if not a.yes:
        print("DRY RUN (nothing sent). Would do:", *plan, sep="\n  ")
        print("Re-run with --yes to execute. Total cost is a few hundred tokens per probe.")
        return 0
    key, base = env_credentials()
    t = ChatTransport(base, key, max_retries=1)
    result: dict = {"base_host": base.split("//")[-1].split("/")[0]}
    try:
        data = t.list_models().get("data", [])
    except ApiError as e:
        print("GET /models failed:", e)
        return 1
    ids = sorted(m.get("id", "?") for m in data)
    result["model_ids"] = ids
    print(f"{len(ids)} models. Candidates by name:")
    for i in ids:
        if any(h in i.lower() for h in HINTS):
            print("  ", i)
    result["extra_fields_seen"] = sorted({k for m in data for k in m if k != "id"})
    result["probes"] = []
    for spec in a.probe:
        role, _, mid = spec.partition("=")
        if role not in ROLES or mid not in ids:
            print(f"skip {spec!r}: unknown role or id not in /models")
            continue
        r = probe(t, role, mid)
        result["probes"].append(r)
        print(json.dumps(r))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"saved {a.out} (no secrets in it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
