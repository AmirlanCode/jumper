"""python -m crab_rescue.build_site --run runs/<ts> --out site/

Builds a STATIC site (HTML/CSS/JS, no backend, no keys, no network at view time) from one run
directory. Only a small allowlist of assets is copied: the hero episode's videos, the frames the
VLM saw in that episode, and three downloads.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

TEMPLATE = Path(__file__).with_name("site_template.html")
FRAME_LIMIT = 40


def pick_hero(summary: dict[str, Any], run: Path) -> int:
    """Best story: an agent rescue that also had a fire-spread event; else any rescue; else seed 0."""
    best, best_score = None, -1.0
    for e in summary["episodes"]:
        if e["method"] != "agent" or not e["rescued"]:
            continue
        trace = run / f"ep_{e['seed']}" / "agent" / "trace.json"
        spread = sum(1 for x in json.loads(trace.read_text()) if x["type"] == "fire_spread") if trace.exists() else 0
        score = spread * 10 + e.get("replans", 0) - e["time_to_rescue_s"] / 100
        if score > best_score:
            best, best_score = e["seed"], score
    if best is not None:
        return best
    return summary["episodes"][0]["seed"]


def build_timeline(run: Path, seed: int) -> list[dict[str, Any]]:
    d = run / f"ep_{seed}" / "agent"
    items: list[dict[str, Any]] = []
    if (d / "perception.json").exists():
        for p in json.loads((d / "perception.json").read_text()):
            items.append({"t": p["t"], "kind": "perception", "role": p["role"], "model": p["model"],
                          "frame": p.get("frame_file"), "valid": p["valid"],
                          "json": p.get("parsed") or p.get("raw"), "error": p.get("error")})
    if (d / "plans.json").exists():
        for p in json.loads((d / "plans.json").read_text()):
            items.append({"t": p["t"], "kind": "plan", "role": p["role"], "model": p["model"],
                          "reason": p["reason"], "accepted": p["accepted"],
                          "rejected_by": p.get("rejected_by"), "reasons": p.get("reasons"),
                          "json": p.get("plan") or p.get("raw")})
    if (d / "trace.json").exists():
        for e in json.loads((d / "trace.json").read_text()):
            if e["type"] in ("fire_spread", "path_blocked", "pickup", "release", "plan_blocked",
                             "vlm_blocked", "plan_failed", "stuck", "episode_end"):
                items.append({"t": e["t"], "kind": "event", "type": e["type"], "detail": e})
    order = {"event": 0, "perception": 1, "plan": 2}
    items.sort(key=lambda x: (x["t"], order[x["kind"]]))
    return items


def build(run: Path, out: Path) -> Path:
    summary = json.loads((run / "summary.json").read_text())
    seed = pick_hero(summary, run)
    if out.exists():
        shutil.rmtree(out)
    assets = out / "assets"
    (assets / "frames").mkdir(parents=True)
    (out / "downloads").mkdir()

    videos = {}
    for m in ("agent", "blind", "oracle"):
        v = run / f"ep_{seed}" / m / "video.mp4"
        if v.exists():
            shutil.copy(v, assets / f"{m}_seed{seed}.mp4")
            videos[m] = f"assets/{m}_seed{seed}.mp4"

    timeline = build_timeline(run, seed)
    used = 0
    for it in timeline:
        if it.get("frame") and used < FRAME_LIMIT:
            src = run / f"ep_{seed}" / "agent" / "frames" / it["frame"]
            if src.exists():
                shutil.copy(src, assets / "frames" / it["frame"])
                used += 1
            else:
                it["frame"] = None
        elif it.get("frame"):
            it["frame"] = None

    bench = None
    if (run / "bench_vlm.json").exists():
        bench = json.loads((run / "bench_vlm.json").read_text())

    for name, src in (("plans.json", run / f"ep_{seed}" / "agent" / "plans.json"),
                      ("report.md", run / "report.md"), ("summary.json", run / "summary.json")):
        if src.exists():
            shutil.copy(src, out / "downloads" / name)

    scene = json.loads((run / f"ep_{seed}" / "agent" / "scene.json").read_text())
    outcome = next((e for e in summary["episodes"] if e["seed"] == seed and e["method"] == "agent"), None)
    data = {"summary": {k: summary[k] for k in ("methods", "fake_models", "seeds")},
            "episodes": [{k: e[k] for k in ("seed", "method", "rescued", "outcome", "time_to_rescue_s",
                                           "fire_exposure_s", "collisions")} for e in summary["episodes"]],
            "hero": {"seed": seed, "videos": videos, "timeline": timeline, "outcome": outcome,
                     "fire_origin": scene["fire_origin"]},
            "bench": bench}
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data).replace("</", "<\\/"))
    (out / "index.html").write_text(html, encoding="utf-8")
    return out / "index.html"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crab_rescue.build_site")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="site")
    a = ap.parse_args(argv)
    p = build(Path(a.run), Path(a.out))
    print(f"site written to {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
