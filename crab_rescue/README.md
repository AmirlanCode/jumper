# Crab Rescue

A room is on fire, a figurine is inside, and a crab robot must find it and bring it to the exit
while the fire spreads. No human in the loop.

```
camera frames -> VLM perception (JSON) -> LLM planner (JSON waypoints) -> safety validator
      ^                                                                        |
      |  monitor: fire spreads / path blocked -> look again -> replan          v
  simulator  <---------- velocity commands (vx, vy, yaw) <---------- waypoint follower
```

Everything lives in `crab_rescue/`; nothing upstream is modified.

## Status (milestone M1: everything works on fakes, $0)

| Part | State |
|---|---|
| Parametric room, spreading fire grid, metrics, baselines | done, tested |
| Strict JSON schemas, grid safety validator, repair -> escalate -> last-valid-plan | done, tested |
| Budget caps + `ledger.jsonl`, model routing, `--dry-run` | done, tested |
| Static showcase site | done (`build_site`) |
| Real OpenAI-compatible VLM/planner clients, `discover` probe script | built, tested against a local mock server; **not yet run against Nebius** (needs network from a machine that reaches it) |
| Real model ids and prices in `config.yaml` | **empty on purpose**: ids come from `/v1/models`, prices from the console |
| MuJoCo + ONNX locomotion executor | **not yet** (M2): the executor here is a kinematic stand-in |

## Running against the real models (from a machine that can reach Nebius)

```bash
pip install pydantic numpy pillow pyyaml
read -s NEBIUS_API_KEY && export NEBIUS_API_KEY        # typed, not echoed, not in shell history
export NEBIUS_BASE_URL=https://api.tokenfactory.us-central1.nebius.com/v1/
python -m crab_rescue.discover                         # dry run: shows what it would do
python -m crab_rescue.discover --yes                   # GET /models: the only source of model ids
# put the ids you choose in config.yaml (models.<role>.id) and prices from the console
python -m crab_rescue.discover --yes --probe vlm_primary=<id> vlm_cheap=<id> planner=<id> \
    planner_fast=<id> planner_escalation=<id>          # tiny text + image test per role, < $0.01
python -m crab_rescue.run --seeds 1 --methods agent,blind,oracle    # first run is a dry run
python -m crab_rescue.run --seeds 1 --methods agent,blind,oracle    # then the real smoke episode
```

`discover` never prints or saves the key; `config.yaml` is refused until every role has an id
and a price; on any cap the run stops new calls and keeps the last valid plan.

## Reproduce (fakes, $0)

```bash
pip install pydantic numpy pillow pyyaml pytest      # nothing GPU related
python -m pytest crab_rescue/tests -q                # 19 tests, no API, no GPU
python -m crab_rescue.run --seeds 20 --fake          # runs/<ts>/ with videos, metrics, report.md
python -m crab_rescue.bench_vlm --fake --out runs/<ts>
python -m crab_rescue.build_site --run runs/<ts> --out site/
python -m crab_rescue.run --seeds 20 --dry-run       # prints planned calls/tokens/cost, spends nothing
```

## Results (20 seeds, FAKE stand-in models, kinematic executor)

| method | rescued | mean time (s) | fire exposure (s) | collisions |
|---|---|---|---|---|
| blind (knows targets, sees nothing) | 1/20 | 36.5 | 12.26 | 3.15 |
| agent (stand-in VLM + planner) | 10/20 | 39.68 | 2.66 | 0.4 |
| oracle (true map, upper bound) | 14/20 | 38.21 | 2.18 | 1.0 |

These numbers say the pipeline and the evaluation work. They are **not** a measurement of any
Nebius model. Seeds where the agent loses are listed in each run's `report.md`.

## Honest simplifications

* Fire is a simulated hazard grid, not fire physics.
* "Carry" is weld-on-contact within 0.4 m; no grasping is simulated.
* The executor is a kinematic stand-in taking the locomotion policy's (vx, vy, yaw) commands;
  no balance model, so falls are not measured.
* The simulation pauses during model calls.
* Success is judged by simulator ground truth only; the agent module cannot reach it (tested).

## Secrets

Keys are read from `NEBIUS_API_KEY` / `NEBIUS_BASE_URL` only. `.env.local`, `api_key*.txt` and
`runs/` are git-ignored; `.env.example` has placeholders.
