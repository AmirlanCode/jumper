"""Cost control: every model call is checked against hard caps and logged to a ledger."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    """A cap was hit; callers stop new calls and fall back to the last valid plan."""


class PriceUnknown(RuntimeError):
    """A real model has no price in config.yaml; do not guess, ask the user."""


class Budget:
    def __init__(self, cfg: dict[str, Any], ledger_path: Path | None = None, fake: bool = False):
        self.caps = cfg["caps"]
        self.models = cfg["models"]
        self.fake = fake
        self.ledger_path = ledger_path
        self.total_usd = 0.0
        self.total_calls = 0
        self.new_episode("-")

    def new_episode(self, episode: str) -> None:
        self.episode = episode
        self.vlm_calls = 0
        self.replans = 0
        self.escalations = 0
        self.ep_tokens_in = 0
        self.ep_tokens_out = 0
        self.ep_cost = 0.0
        self.ep_invalid_calls = 0

    # ---- pre-call gate -------------------------------------------------------------
    def check(self, kind: str, role: str, est_in: int = 0, est_out: int = 0) -> None:
        if kind == "vlm" and self.vlm_calls >= self.caps["max_vlm_calls_per_episode"]:
            raise BudgetExceeded(f"max_vlm_calls_per_episode={self.caps['max_vlm_calls_per_episode']}")
        if kind == "plan" and self.replans >= self.caps["max_replans_per_episode"]:
            raise BudgetExceeded(f"max_replans_per_episode={self.caps['max_replans_per_episode']}")
        if role == "planner_escalation" and self.escalations >= self.caps["max_escalations"]:
            raise BudgetExceeded(f"max_escalations={self.caps['max_escalations']}")
        if est_in + est_out > self.caps["max_tokens_per_call"] * 4 and not self.fake:
            raise BudgetExceeded("estimated tokens for this call exceed the per-call cap x4")
        if not self.fake:
            m = self.models[role]
            if m.get("price_in") is None or m.get("price_out") is None or m.get("id") is None:
                raise PriceUnknown(f"models.{role}: id/price missing in config.yaml")
        if self.total_usd >= self.caps["max_total_usd"]:
            raise BudgetExceeded(f"max_total_usd={self.caps['max_total_usd']}")

    def cost_of(self, role: str, tin: int, tout: int) -> float:
        m = self.models[role]
        if self.fake or m.get("price_in") is None:
            return 0.0
        return (tin * m["price_in"] + tout * m["price_out"]) / 1e6

    # ---- post-call record -----------------------------------------------------------
    def record(self, kind: str, role: str, model: str, tin: int, tout: int,
               latency_s: float, **meta: Any) -> dict[str, Any]:
        cost = self.cost_of(role, tin, tout)
        self.total_usd += cost
        self.ep_cost += cost
        self.ep_tokens_in += tin
        self.ep_tokens_out += tout
        self.total_calls += 1
        if kind == "vlm":
            self.vlm_calls += 1
        if kind == "plan":
            self.replans += 1
        if role == "planner_escalation":
            self.escalations += 1
        row = {"ts": time.time(), "episode": self.episode, "kind": kind, "role": role,
               "model": model, "tokens_in": tin, "tokens_out": tout,
               "latency_s": round(latency_s, 4), "cost_usd": cost, "fake": self.fake, **meta}
        if self.ledger_path:
            with self.ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        return row

    def summary(self) -> dict[str, Any]:
        return {"total_usd": self.total_usd, "total_calls": self.total_calls,
                "cap_usd": self.caps["max_total_usd"]}
