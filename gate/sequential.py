"""Finite registered-look orchestration; union-bound spending, not normal boundaries."""

import math
from dataclasses import asdict

from gate.verdict import VerdictResult, verdict


class SequentialGate:
    def __init__(
        self,
        required_strata,
        margin,
        alpha=0.05,
        weights=(0.2, 0.3, 0.5),
        higher_is_worse=True,
        futility_at=None,
    ):
        self.required = tuple(required_strata)
        self.margin, self.alpha = margin, alpha
        self.weights = tuple(weights)
        if (
            not self.weights
            or any(not math.isfinite(w) or w <= 0 for w in self.weights)
            or not math.isclose(sum(self.weights), 1.0, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("positive look weights must sum to one")
        # Reuse structural validation even with no observations.
        verdict([], margin, alpha, required_strata=self.required, higher_is_worse=higher_is_worse)
        self.higher_is_worse = higher_is_worse
        self.futility_at = {int(k): tuple(v) for k, v in (futility_at or {}).items()}
        if any(
            k < 1 or k > len(self.weights) or not set(v) <= set(self.required)
            for k, v in self.futility_at.items()
        ):
            raise ValueError("invalid registered futility schedule")
        self.look = 0
        self.terminal = False
        self._batches = []
        self._rows = []
        self._seen_roads = set()
        self._closed = {}
        self.result = None

    @property
    def active_strata(self):
        return tuple(s for s in self.required if s not in self._closed)

    def update(self, batch):
        if self.terminal:
            raise ValueError("terminal decision cannot receive more observations")
        batch = [dict(row) for row in batch]
        active = self.active_strata
        if {r.get("stratum") for r in batch} != set(active):
            raise ValueError("each batch must cover every active stratum and no closed stratum")
        roads = {r.get("road_id") for r in batch}
        if roads & self._seen_roads:
            raise ValueError("a road cannot enter two looks")
        previous_scenes = {r["scene_id"] for r in self._rows}
        if any(r.get("scene_id") in previous_scenes for r in batch):
            raise ValueError("a scene cannot enter two roads or looks")
        current_rows = [r for r in self._rows if r["stratum"] in active] + batch
        current = verdict(
            current_rows,
            self.margin,
            self.alpha * self.weights[self.look],
            required_strata=active,
            higher_is_worse=self.higher_is_worse,
            regression_family=self.required,
        )
        # Validation above is complete before state mutation.
        self._batches.append(batch)
        self._rows.extend(batch)
        self._seen_roads.update(roads)
        self.look += 1
        for s, state in current.strata.items():
            if state == "GO":
                self._closed[s] = current.details[s]
        states = {s: "GO" for s in self._closed}
        states.update(current.strata)
        details = dict(self._closed)
        details.update(current.details)
        futility = any(states[s] != "GO" for s in self.futility_at.get(self.look, ()))
        if "NO-GO" in states.values():
            overall = "NO-GO"
        elif futility:
            overall = "INSUFFICIENT"
        elif all(s == "GO" for s in states.values()):
            overall = "GO"
        else:
            overall = "INSUFFICIENT"
        self.terminal = overall in ("GO", "NO-GO") or futility or self.look == len(self.weights)
        self.result = VerdictResult(
            overall,
            states,
            details,
            self.alpha,
            self.margin,
            method="finite_look_spent_alpha_road_KL",
        )
        return self.result

    def to_dict(self):
        return {
            "schema_version": 1,
            "required_strata": list(self.required),
            "margin": self.margin,
            "alpha": self.alpha,
            "weights": list(self.weights),
            "higher_is_worse": self.higher_is_worse,
            "futility_at": dict(self.futility_at),
            "batches": [[dict(r) for r in b] for b in self._batches],
            "result": None if self.result is None else asdict(self.result),
        }

    @classmethod
    def from_dict(cls, state):
        if state.get("schema_version") != 1:
            raise ValueError("unsupported state version")
        gate = cls(
            state["required_strata"],
            state["margin"],
            state["alpha"],
            state["weights"],
            state["higher_is_worse"],
            state["futility_at"],
        )
        for batch in state["batches"]:
            gate.update(batch)
        expected = None if gate.result is None else asdict(gate.result)
        if expected != state.get("result"):
            raise ValueError("state result does not match replay")
        return gate
