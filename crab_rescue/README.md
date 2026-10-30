# Crab Rescue

A room is on fire, a figurine is inside, and a crab robot must find it and bring it to the exit.
No human in the loop: **NVIDIA Nemotron-3.5-Super-VL, served on Nebius Token Factory, drives the
robot one step at a time** from what it sees in the camera frame.

Track: **Physical AI** (simulation; no hardware in the loop yet).

## The recorded rescue

| | |
|---|---|
| Model | `nvidia/Nemotron-3.5-Super-VL` (NVIDIA open model) via Nebius Token Factory |
| Outcome | figurine found, picked up and carried to the exit in **41.0 s** of simulated time |
| Cost | 32 model calls, 35 341 tokens in / 2 396 out, **$0.043** |
| Path planner | none: every move is the model's own command; code only executes it |

Each step is two calls to the same model:

1. **Look.** The camera frame goes in; the model returns where the figurine and the exit are.
2. **Act.** The frame, its own sighting, a wall/fire map from the same camera, free distance in
   eight directions, its pose and its last steps go in; it returns one command:
   `move_to` (at most 1.5 m), `pickup` or `release`.

A safety check refuses a step that would cross a wall or fire, and the robot then asks the model
again. The figurine's position is never given to the model; it finds it in the image. In the
recorded run the fire is held static and the room layout is seed 191.

```bash
pip install -e .                                   # from the repository root
pip install pydantic numpy pillow pyyaml playwright imageio-ffmpeg
python -m playwright install chromium
read -s NEBIUS_API_KEY && export NEBIUS_API_KEY    # typed, not echoed, never written to a file
python -m crab_rescue.model_drive --seed 191       # writes runs3d/model_drive/<ts>_seed191_live/
```

The output directory is a self-contained site (`index.html` replays the run from `mission.json`,
serve it with `python -m http.server`), plus `ledger.jsonl` with every call, its raw reply,
tokens and cost, and `frames/` with the exact images the model saw. `./run_crab_rescue.sh` at the
repository root sets up an environment and runs the original two-model pipeline in one command.

### How NVIDIA and Nebius are used

* **NVIDIA Nemotron-3.5-Super-VL** is both the eyes and the driver: image understanding and the
  step-by-step decision come from the same open model.
* **Nebius Token Factory** serves it through an OpenAI-compatible API (`crab_rescue/nebius.py`);
  `crab_rescue/discover.py` lists the account's models and probes each candidate with one tiny
  text and image call before a run. Other NVIDIA models on Token Factory (Nemotron 3 Super / Ultra,
  Cosmos Reasoner) plug into the same roles through `config.yaml`.
* Nemotron reasons before answering by default; the client turns that off with
  `chat_template_kwargs={"enable_thinking": false}` so each step returns its JSON in a few
  dozen tokens. That is what makes 32 calls cost four cents.

### Built during the hackathon

The Jumper training framework in this repository existed before the submission period.
Everything under `crab_rescue/` and `run_crab_rescue.sh` was written during it (first commit
2026-10-29): the burning-room simulator, the 3D scene and replay site, the Token Factory client,
the safety check, and the model-driven controller.

## The original pipeline

```
camera frames -> VLM perception (JSON) -> LLM planner (JSON waypoints) -> safety validator
      ^                                                                        |
      |  monitor: fire spreads / path blocked -> look again -> replan          v
  simulator  <---------- velocity commands (vx, vy, yaw) <---------- waypoint follower
```

Everything lives in `crab_rescue/`; nothing upstream is modified.

## Status (real Nebius rescue demo recorded; see the limitations below)

| Part | State |
|---|---|
| Parametric room, spreading fire grid, metrics, baselines | done, tested |
| Strict JSON schemas, grid safety validator, repair -> escalate -> last-valid-plan | done, tested |
| Budget caps + `ledger.jsonl`, model routing, `--dry-run` | done, tested |
| Static showcase site | done (`build_site`) |
| Real OpenAI-compatible VLM/planner clients, `discover` probe script | Nebius `/v1/models`, text and image calls verified; one real episode recorded |
| Real model ids and prices in `config.yaml` | populated from the account model list and model catalog on 2026-10-29 |
| Real rescue outcome | seed 191 rescued in 41.0 s, every move chosen by Nemotron-3.5-Super-VL (`model_drive.py`, above) |
| MuJoCo + ONNX locomotion executor | **not yet**: the rescue app uses a kinematic stand-in; walking policy and physical claw policy are not integrated |

## The live app (watch the crab rescue the figurine)

```bash
pip install pydantic numpy pillow pyyaml
python -m crab_rescue.app --open          # http://127.0.0.1:8765, bound to this machine only
```

Pick "Offline stand-in models" for a $0 demo, or "NVIDIA models on Nebius": paste the key in the
page, press Load models, choose a vision and a planner model, press the tiny Test button, then
Start rescue. The page shows each frame the model saw, its JSON, the route it chose, whether the
safety check approved it, and tokens/cost. The key stays in the app's memory.

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

## 3D rescue mission (`mission3d.py`, `scene3d/`)

A presentation scene on top of the pipeline above: the Jumper crab (real URDF/STL meshes) notices a lying
figurine in a burning room, plans a route, walks to it, grasps it with both claws, carries it to the SAFE ZONE and
confirms. Observations are rendered from the 3D scene (lit frame plus a flat semantic frame); the VLM/planner JSON
goes through the same schema + safety validator + waypoint follower as the 2D mission.

```bash
export NEBIUS_API_KEY="..."          # env only. Never put the key in files, logs or commits.
python -m crab_rescue.mission3d --live --seed 5 --video --max-usd 0.05 --max-vlm-calls 4 --vlm-period 25
python -m crab_rescue.mission3d --seed 5 --video      # offline, labelled OFFLINE STAND-IN
```

Output goes to `runs3d/<timestamp>_seed<N>_<live|offline>/`, a self-contained site: `index.html`, `mission.json`
(schema `crab-rescue-mission/1`), `ledger.jsonl` (per-call input/raw reply/parsed/validation/time/errors, no secrets),
`frames/` (the exact images the models saw), `mission.mp4` (1280x720, 24 fps, frame-stepped replay of that run).
Serve it with `python -m http.server` inside the folder, or open the file input in the page and load your own log.

Live mode first probes `/models` and sends one tiny image call per NVIDIA vision candidate and one text call for the
planner. The IDs that answered are written to the log. Nothing is assumed available.

Source labels: `nvidia_model`, `nebius_model` (non-NVIDIA model), `offline_standin`, `safety_fallback` (local A*).
Fire and wall cells always come from the semantic channel, not from a model; when the model's figurine/exit point
disagrees with it by more than 0.06 the value is overridden and the log says so (`grounding`).

Limitations: gait is procedural (the real `locomotion.onnx` is not connected, body motion is kinematic); claw grasp is
a visual IK mock with no contact physics; fire is a cellular automaton; simulation only, not evidence of real-robot
rescue readiness.
