"""Model routing. Roles map to model ids through config.yaml; this only picks the role."""

from __future__ import annotations


class Router:
    def __init__(self, max_escalations: int, low_confidence: float = 0.5):
        self.max_escalations = max_escalations
        self.low_confidence = low_confidence
        self.escalations = 0

    def new_episode(self) -> None:
        self.escalations = 0

    def pick_vlm(self, *, first: bool, fire_event: bool, confidence: float | None) -> str:
        """Cheap model for routine frames; the strong one for key moments."""
        if first or fire_event or (confidence is not None and confidence < self.low_confidence):
            return "vlm_primary"
        return "vlm_cheap"

    def pick_planner(self, *, time_pressure: bool, failed_before: bool) -> str:
        """Super by default; Lightning under time pressure; Ultra only after a failure."""
        if failed_before and self.escalations < self.max_escalations:
            self.escalations += 1
            return "planner_escalation"
        return "planner_fast" if time_pressure else "planner"
