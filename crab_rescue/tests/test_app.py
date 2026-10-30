"""The live app's server logic, offline (no browser, no network)."""
import time

import pytest

from crab_rescue import app


def _wait_done(timeout=120):
    t0 = time.time()
    while app.S.status == "running" and time.time() - t0 < timeout:
        time.sleep(0.05)


def test_offline_run_streams_state_and_judges_by_ground_truth():
    app.start_run({"seed": 5, "method": "agent", "mode": "offline", "speed": 1000})
    _wait_done()
    j = app.state_payload({"p": 0, "l": 0, "e": 0})
    assert j["status"] == "done" and j["snap"]["outcome"] in ("rescued", "figurine_lost_to_fire", "timeout")
    assert j["perceptions"] and j["plans"] and j["perceptions"][0]["png"]
    assert j["budget"]["calls"] > 0 and j["budget"]["priced"] is True      # fake models cost $0, known
    later = app.state_payload({"p": j["n"]["p"], "l": j["n"]["l"], "e": j["n"]["e"]})
    assert later["perceptions"] == [] and later["plans"] == []              # incremental, no repeats


def test_real_mode_refuses_without_key_and_stop_works():
    app.S.creds.update(key=None, base=None)
    with pytest.raises(ValueError):
        app.start_run({"seed": 1, "method": "agent", "mode": "nvidia", "vlm": "a", "planner": "b"})
    app.start_run({"seed": 2, "method": "blind", "mode": "offline", "speed": 1})   # real-time: slow
    app.S.stop = True
    _wait_done(10)
    assert app.S.status == "idle"
