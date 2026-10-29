"""Metrics. Success is judged by simulator ground truth, never by a model's claim."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .executor import World
from .schemas import Perception
from .validator import rasterize


def perception_accuracy(world: World, perceptions: list[dict[str, Any]]) -> dict[str, Any]:
    fig_err, exit_err, ious, fire_recall = [], [], [], []
    by_role: dict[str, list[float]] = {}
    for p in perceptions:
        if not p.get("valid"):
            continue
        gt = world.gt_frames[p["frame_id"]]
        per = Perception.model_validate(p["parsed"])
        W, H = world.room.width_m, world.room.height_m
        if per.figurine is not None:
            e = math.hypot(per.figurine.x * W - gt["figurine"][0], per.figurine.y * H - gt["figurine"][1])
            fig_err.append(e)
        else:
            fig_err.append(float("nan"))
        if per.exit is not None:
            exit_err.append(math.hypot(per.exit.x * W - gt["exit_xy"][0], per.exit.y * H - gt["exit_xy"][1]))
        else:
            exit_err.append(float("nan"))
        pf = rasterize(per.fire, world.room.w, world.room.h)
        gf = gt["fire"]
        inter, union = (pf & gf).sum(), (pf | gf).sum()
        iou = float(inter / union) if union else 1.0
        ious.append(iou)
        fire_recall.append(float(inter / gf.sum()) if gf.sum() else 1.0)
        by_role.setdefault(p["role"], []).append(iou)

    def mean(v: list[float]) -> float | None:
        v = [x for x in v if not math.isnan(x)]
        return round(float(np.mean(v)), 4) if v else None

    return {"n_perceptions": len(ious), "figurine_err_m": mean(fig_err), "exit_err_m": mean(exit_err),
            "fire_iou": mean(ious), "fire_recall": mean(fire_recall),
            "fire_iou_by_role": {k: mean(v) for k, v in by_role.items()},
            "figurine_missed": int(sum(math.isnan(x) for x in fig_err))}


def episode_metrics(world: World, method: str, seed: int, budget=None, log: dict[str, Any] | None = None,
                    uses_ground_truth: bool = False) -> dict[str, Any]:
    m: dict[str, Any] = {
        "seed": seed, "method": method, "uses_ground_truth": uses_ground_truth,
        "rescued": world.outcome == "rescued", "outcome": world.outcome,
        "time_to_rescue_s": round(world.rescued_t, 2) if world.rescued_t is not None else None,
        "sim_time_s": round(world.t, 2),
        "fire_exposure_s": round(world.fire_exposure_s, 2), "fire_entries": world.fire_entries,
        "collisions": world.collisions, "falls": None,
        "falls_note": "not applicable: kinematic stand-in executor has no balance model",
        "fire_cells_final": world.fire.area,
    }
    if budget is not None and log is not None:
        plans = log["plans"]
        m.update({
            "replans": budget.replans, "vlm_calls": budget.vlm_calls, "escalations": budget.escalations,
            "invalid_plans": sum(1 for p in plans if not p.get("accepted")),
            "plan_attempts": len(plans),
            "tokens_in": budget.ep_tokens_in, "tokens_out": budget.ep_tokens_out,
            "cost_usd": round(budget.ep_cost, 6),
            "perception": perception_accuracy(world, log["perceptions"]),
        })
    return m
