"""Fire model, scene, schemas, validator, follower. No models, no network."""
import json

import pytest
from pydantic import ValidationError

from crab_rescue.config import load_config
from crab_rescue.fire import FireModel
from crab_rescue.follower import WaypointFollower
from crab_rescue.scene import EXIT, FREE, generate_room
from crab_rescue.schemas import Perception, Plan, extract_json, parse_perception, parse_plan
from crab_rescue.validator import build_map, validate_plan

CFG = load_config()


def test_rooms_are_deterministic_and_solvable():
    a, b = generate_room(3), generate_room(3)
    assert (a.grid == b.grid).all() and a.figurine == b.figurine
    for s in range(30):
        r = generate_room(s)
        assert r.grid[r.fire_origin[1], r.fire_origin[0]] == FREE
        assert all(r.grid[y, x] == EXIT for x, y in r.exit_cells)


def test_fire_is_seeded_monotone_and_never_burns_walls():
    r = generate_room(1)
    f1, f2 = FireModel(r, 5, 0.3), FireModel(r, 5, 0.3)
    prev = f1.area
    for t in range(1, 40):
        f1.advance_to(t)
        f2.advance_to(t)
        assert f1.area >= prev
        prev = f1.area
    assert (f1.burning == f2.burning).all()
    assert f1.area > 1
    assert not (f1.burning & (r.grid != FREE)).any()
    # control group: a fire with p=0 must not spread (the assertion above can fail)
    f0 = FireModel(r, 5, 0.0)
    f0.advance_to(40)
    assert f0.area == 1


def test_schema_rejects_code_extra_fields_and_out_of_room():
    ok = {"waypoints": [{"x": 1, "y": 1, "speed": 0.4}], "rationale": "r"}
    assert parse_plan(json.dumps(ok), (10, 8)).waypoints[0].action == "move"
    with pytest.raises(ValidationError):
        parse_plan(json.dumps({**ok, "code": "import os"}), (10, 8))
    with pytest.raises(ValidationError):
        parse_plan(json.dumps({"waypoints": [{"x": 1, "y": 1, "action": "os.system"}], "rationale": ""}), (10, 8))
    with pytest.raises(ValidationError):
        parse_plan(json.dumps({"waypoints": [{"x": 1, "y": 1, "speed": 5.0}], "rationale": ""}), (10, 8))
    with pytest.raises(ValidationError):
        parse_plan(json.dumps({"waypoints": [{"x": 11, "y": 1}], "rationale": ""}), (10, 8))
    assert extract_json('noise ```json\n{"a": 1}\n``` tail') == '{"a": 1}'
    with pytest.raises(ValueError):
        extract_json("no object here")
    with pytest.raises(ValidationError):
        parse_perception('{"confidence": 2.0}')


def _pmap(fire_boxes=(), obstacles=()):
    per = Perception.model_validate({
        "figurine": {"x": 0.8, "y": 0.5}, "exit": {"x": 0.025, "y": 0.375},
        "fire": list(fire_boxes), "obstacles": list(obstacles), "confidence": 0.9})
    return build_map(per, 20, 16, 0.5)


def _plan(pts):
    return Plan(waypoints=[{"x": x, "y": y, "speed": 0.4} for x, y in pts], rationale="t")


def test_validator_accepts_clear_path_and_rejects_fire_wall_bounds():
    clear = _pmap()
    assert validate_plan(_plan([(5, 4)]), clear, (2, 4)) == []
    fire = _pmap(fire_boxes=[{"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6}])      # cells around (5,4)
    assert any("fire" in r for r in validate_plan(_plan([(7, 4)]), fire, (2, 4)))
    wall = _pmap(obstacles=[{"x0": 0.45, "y0": 0.1, "x1": 0.55, "y1": 0.9}])
    assert any("wall" in r for r in validate_plan(_plan([(7, 4)]), wall, (2, 4)))
    assert any("outside" in r for r in validate_plan(_plan([(11, 4)]), clear, (2, 4)))
    # the outer ring is a wall too, except at the exit
    assert any("wall" in r for r in validate_plan(_plan([(2, 0.2)]), clear, (2, 4)))
    assert validate_plan(_plan([(0.3, 3.0)]), clear, (2, 3.0)) == []


def test_validator_lets_a_crab_leave_fire_it_stands_next_to():
    fire = _pmap(fire_boxes=[{"x0": 0.15, "y0": 0.4, "x1": 0.25, "y1": 0.6}])    # cells x=3..4
    assert validate_plan(_plan([(2.0, 4.0)]), fire, (2.05, 4.0)) == []


def test_follower_commands_stay_in_limits_and_fire_actions():
    cmax = tuple(CFG["limits"]["cmd_max"])
    f = WaypointFollower(cmax)
    f.set_plan(Plan(waypoints=[{"x": 3, "y": 4, "speed": 0.5, "action": "pickup"}], rationale=""))
    pose = [1.0, 1.0, 3.0]
    fired = None
    for _ in range(4000):
        cmd, act = f.command(tuple(pose))
        assert abs(cmd[0]) <= cmax[0] and abs(cmd[1]) <= cmax[1] and abs(cmd[2]) <= cmax[2]
        if act:
            fired = act
            break
        import math
        pose[2] += cmd[2] * 0.1
        pose[0] += cmd[0] * math.cos(pose[2]) * 0.1
        pose[1] += cmd[0] * math.sin(pose[2]) * 0.1
    assert fired == "pickup" and f.finished
