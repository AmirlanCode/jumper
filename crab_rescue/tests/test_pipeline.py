"""Budget caps, routing, agent isolation, repair/escalation/fallback, run CLI, site builder."""
import ast
import copy
import json
import re
from pathlib import Path

import pytest

from crab_rescue import run as run_mod
from crab_rescue.agent import run_agent
from crab_rescue.budget import Budget, BudgetExceeded, PriceUnknown
from crab_rescue.build_site import build
from crab_rescue.clients import CallResult, DiskCache, FakePlanner, FakeVLM
from crab_rescue.config import load_config
from crab_rescue.executor import SensorView, World
from crab_rescue.router import Router
from crab_rescue.scene import generate_room

CFG = load_config()
PKG = Path(__file__).resolve().parents[1]


def _world(seed=2, cfg=CFG):
    room = generate_room(seed, *cfg["sim"]["grid"], cfg["sim"]["cell_m"])
    return World(room, cfg, seed)


def _agent(cfg, planner, seed=2, vlm=None, fake=True):
    w = _world(seed, cfg)
    b = Budget(cfg, fake=fake)
    r = Router(cfg["caps"]["max_escalations"])
    log = run_agent(SensorView(w), vlm or FakeVLM(tuple(cfg["sim"]["grid"]), seed=seed), planner, b, r, cfg)
    return w, b, log


# ---- budget ---------------------------------------------------------------------------
def test_budget_caps_and_unknown_prices(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["caps"].update(max_vlm_calls_per_episode=2, max_replans_per_episode=1, max_escalations=1)
    b = Budget(cfg, ledger_path=tmp_path / "l.jsonl", fake=True)
    for _ in range(2):
        b.check("vlm", "vlm_cheap")
        b.record("vlm", "vlm_cheap", "m", 10, 5, 0.1)
    with pytest.raises(BudgetExceeded):
        b.check("vlm", "vlm_cheap")
    b.check("plan", "planner")
    b.record("plan", "planner", "m", 10, 5, 0.1)
    with pytest.raises(BudgetExceeded):
        b.check("plan", "planner")
    b.record("repair", "planner_escalation", "m", 1, 1, 0.0)
    with pytest.raises(BudgetExceeded):
        b.check("repair", "planner_escalation")
    b.new_episode("next")                       # caps are per episode
    b.check("vlm", "vlm_cheap")
    rows = [json.loads(x) for x in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and {"model", "tokens_in", "tokens_out", "latency_s", "cost_usd"} <= set(rows[0])
    # real (non-fake) call with no id/price must refuse instead of guessing
    with pytest.raises(PriceUnknown):
        Budget(CFG, fake=False).check("vlm", "vlm_primary")


def test_total_usd_cap_and_cost_math():
    cfg = copy.deepcopy(CFG)
    cfg["models"]["planner"].update(id="x", price_in=1.0, price_out=2.0)
    cfg["caps"]["max_total_usd"] = 0.001
    b = Budget(cfg, fake=False)
    row = b.record("plan", "planner", "x", 1000, 500, 0.0)
    assert row["cost_usd"] == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)
    b.record("repair", "planner", "x", 1000, 500, 0.0)
    with pytest.raises(BudgetExceeded):
        b.check("repair", "planner")


# ---- routing --------------------------------------------------------------------------
def test_routing_rules():
    r = Router(max_escalations=1)
    assert r.pick_vlm(first=True, fire_event=False, confidence=None) == "vlm_primary"
    assert r.pick_vlm(first=False, fire_event=False, confidence=0.9) == "vlm_cheap"
    assert r.pick_vlm(first=False, fire_event=True, confidence=0.9) == "vlm_primary"
    assert r.pick_vlm(first=False, fire_event=False, confidence=0.3) == "vlm_primary"
    assert r.pick_planner(time_pressure=False, failed_before=False) == "planner"
    assert r.pick_planner(time_pressure=True, failed_before=False) == "planner_fast"
    assert r.pick_planner(time_pressure=True, failed_before=True) == "planner_escalation"
    assert r.pick_planner(time_pressure=False, failed_before=True) == "planner"   # cap of 1 reached


# ---- agent isolation ----------------------------------------------------------------------
def test_agent_never_imports_ground_truth_modules():
    tree = ast.parse((PKG / "agent.py").read_text())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level:
            imported.add(n.module)
    assert imported.isdisjoint({"executor", "scene", "fire", "evaluator", "baselines", "run"}), imported
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"World", "Room", "FireModel", "gt_frames", "truth", "burning"}, names


def test_sensor_view_exposes_no_world():
    sv = SensorView(_world())
    public = {n for n in dir(sv) if not n.startswith("_")}
    assert public == {"actuate", "bounds", "carrying", "done", "frame", "grasp", "pose", "release", "time"}


# ---- repair / escalation / fallback ----------------------------------------------------------
def test_bad_first_plans_are_repaired_not_executed():
    w, b, log = _agent(CFG, FakePlanner(bad_first_rate=1.0, seed=1))
    rejected = [p for p in log["plans"] if not p["accepted"]]
    assert rejected and {p["rejected_by"] for p in rejected} <= {"schema", "validator"}
    assert any(p["accepted"] for p in log["plans"])
    # every plan the robot followed was an accepted one; the unsafe one never ran
    assert w.collisions <= 2


class AlwaysBad:
    def plan(self, role, context, feedback):
        return CallResult("not json at all", 10, 5, model=f"bad-{role}")


def test_unfixable_planner_escalates_then_holds_and_never_exceeds_caps():
    cfg = copy.deepcopy(CFG)
    cfg["caps"]["max_escalations"] = 2
    cfg["sim"]["max_episode_s"] = 40.0
    w, b, log = _agent(cfg, AlwaysBad())
    roles = {p["role"] for p in log["plans"]}
    assert "planner_escalation" in roles
    assert b.escalations <= 2
    assert not any(p["accepted"] for p in log["plans"])
    assert any(e["type"] == "plan_failed" for e in log["events"])
    assert w.outcome != "rescued"


def test_caps_stop_calls_and_agent_falls_back_to_last_valid_plan():
    cfg = copy.deepcopy(CFG)
    cfg["caps"].update(max_vlm_calls_per_episode=1, max_replans_per_episode=1)
    w, b, log = _agent(cfg, FakePlanner(seed=3))
    assert b.vlm_calls == 1 and b.replans <= 1
    assert any(e["type"] in ("vlm_blocked", "plan_blocked") for e in log["events"])
    assert w.done                                # it kept going on the last valid plan, no crash


def test_invalid_vlm_json_gets_one_repair_retry():
    class Flaky(FakeVLM):
        def perceive(self, role, png, meta):
            if self.n == 0:
                self.n += 1
                return CallResult("sorry, I cannot", 10, 5, model="flaky")
            return super().perceive(role, png, meta)
    w, b, log = _agent(CFG, FakePlanner(seed=1), vlm=Flaky(tuple(CFG["sim"]["grid"])))
    first = log["perceptions"][:2]
    assert first[0]["valid"] is False and first[1]["valid"] is True and first[1]["attempt"] == 1


def test_disk_cache_roundtrip(tmp_path):
    c = DiskCache(tmp_path)
    k = c.key("prompt", b"png")
    assert c.get(k) is None
    c.put(k, CallResult("x", 1, 2, 0.5, "m"))
    assert c.get(k).text == "x" and c.get(c.key("other")) is None


# ---- executor ground truth -----------------------------------------------------------------
def test_world_counts_walls_and_fire_and_judges_by_ground_truth():
    w = _world(0)
    for _ in range(60):                          # drive west into the wall ring
        w.step((0.0, -0.8, 0.0))
    assert w.collisions >= 1
    x, y, _ = w.pose
    assert 0 < x < w.room.width_m and 0 < y < w.room.height_m
    w2 = _world(0)
    w2.fig = list(w2.room.exit_xy)               # figurine placed in the exit cell -> truth says rescued
    w2.step((0, 0, 0))
    assert w2.done and w2.outcome == "rescued"


# ---- run CLI + site ------------------------------------------------------------------------
def test_dry_run_is_default_for_real_runs_and_spends_nothing(tmp_path, capsys):
    rc = run_mod.main(["--seeds", "2", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "DRY RUN" in out and "UNKNOWN" in out
    assert not [p for p in tmp_path.iterdir() if p.name[0].isdigit()]   # no run dir created
    assert run_mod.main(["--seeds", "1", "--out", str(tmp_path)]) == 2   # second time: refuses real run (M2)


def test_end_to_end_fake_run_and_site(tmp_path):
    runs = tmp_path / "runs"
    assert run_mod.main(["--seeds", "2", "--fake", "--out", str(runs)]) == 0
    run = next(p for p in runs.iterdir() if p.name[0].isdigit())
    for f in ("ledger.jsonl", "report.md", "summary.json"):
        assert (run / f).exists()
    ep = run / "ep_0" / "agent"
    for f in ("plans.json", "perception.json", "metrics.json", "trace.json", "video.mp4"):
        assert (ep / f).exists(), f
    m = json.loads((ep / "metrics.json").read_text())
    assert m["uses_ground_truth"] is False and "perception" in m and m["fake_models"] is True
    assert json.loads((run / "ep_0" / "blind" / "metrics.json").read_text())["uses_ground_truth"] is True
    index = build(run, tmp_path / "site")
    html = index.read_text()
    assert "stand-in models" in html and "plans.json" in html
    assert (tmp_path / "site" / "downloads" / "report.md").exists()
    # secret scan over everything the site ships
    for p in (tmp_path / "site").rglob("*"):
        if p.is_file() and p.suffix in {".html", ".json", ".md"}:
            t = p.read_text(errors="ignore")
            assert not re.search(r"Bearer\s+\S{20,}|v1\.[A-Za-z0-9_-]{40,}|NEBIUS_API_KEY\s*=\s*\S", t), p
            assert "http://" not in t and "fetch(" not in t and "XMLHttpRequest" not in t
