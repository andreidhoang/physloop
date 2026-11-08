"""Agent-authored S1 structural contracts and strict pairing admission.

These objects do not establish statistical validity. Pairing matches scene/replicate
identities, preserves road clusters, and never infers RNG coupling from a replicate ID.
Continuous values retain their original direction; consumers must consult MetricSpec.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Literal


class PairingError(ValueError):
    """Input cannot be admitted as a complete, protocol-compatible comparison."""


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _finite(value, name):
    if not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: Literal["binary", "continuous"]
    higher_is_worse: bool
    unit: str

    def __post_init__(self):
        _text(self.name, "metric name")
        _text(self.unit, "metric unit")
        if self.kind not in ("binary", "continuous"):
            raise ValueError("metric kind must be binary or continuous")
        if not isinstance(self.higher_is_worse, bool):
            raise TypeError("higher_is_worse must be boolean")


@dataclass(frozen=True)
class Protocol:
    identity: str
    seed_mode: Literal["synthetic_seeded", "unseeded", "seeded"]
    config_hash: str
    evaluator: str
    score_definition: str
    deviation_convention: str

    def __post_init__(self):
        for name in (
            "identity",
            "config_hash",
            "evaluator",
            "score_definition",
            "deviation_convention",
        ):
            _text(getattr(self, name), name)
        if self.seed_mode not in ("synthetic_seeded", "unseeded", "seeded"):
            raise ValueError("unsupported seed_mode")


@dataclass(frozen=True)
class RolloutRecord:
    arm: Literal["A", "B"]
    scene_id: str
    road_id: str
    stratum: str
    replicate_id: int | str
    value: float
    completed: bool = True
    deviation_m: float | None = None
    seed: int | None = None

    def __post_init__(self):
        if self.arm not in ("A", "B"):
            raise ValueError("arm must be A or B")
        for name in ("scene_id", "road_id", "stratum"):
            _text(getattr(self, name), name)
        if isinstance(self.replicate_id, str):
            _text(self.replicate_id, "replicate_id")
        else:
            _nonnegative_int(self.replicate_id, "replicate_id")
        _finite(self.value, "value")
        if not isinstance(self.completed, bool):
            raise TypeError("completed must be boolean")
        if self.deviation_m is not None:
            _finite(self.deviation_m, "deviation_m")
            if self.deviation_m < 0:
                raise ValueError("deviation_m must be nonnegative")
        if self.seed is not None:
            _nonnegative_int(self.seed, "seed")


@dataclass(frozen=True)
class ComparisonSpec:
    reference: str
    candidate: str
    protocol: Protocol
    required_strata: tuple[str, ...]
    margin: float
    alpha: float
    metric: MetricSpec

    def __post_init__(self):
        _text(self.reference, "reference")
        _text(self.candidate, "candidate")
        if not isinstance(self.protocol, Protocol) or not isinstance(self.metric, MetricSpec):
            raise TypeError("protocol and metric require their typed contracts")
        if not isinstance(self.required_strata, tuple) or not self.required_strata:
            raise ValueError("required_strata must be a nonempty tuple")
        for name in self.required_strata:
            _text(name, "required stratum")
        if len(set(self.required_strata)) != len(self.required_strata):
            raise ValueError("required_strata contains duplicates")
        _finite(self.margin, "margin")
        _finite(self.alpha, "alpha")
        if not 0 < self.margin < 1:
            raise ValueError("margin must be between zero and one for the bounded v0 gate")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one")


@dataclass(frozen=True)
class PairingReport:
    rows: tuple[dict, ...]
    n_records: int
    n_pairs: int
    n_scenes: int
    n_roads: int
    missing_strata: tuple[str, ...]


def is_pair(
    a: RolloutRecord,
    b: RolloutRecord,
    reference_protocol: Protocol,
    candidate_protocol: Protocol,
) -> bool:
    """Check identity compatibility; this predicate does not certify outcome completeness."""
    if reference_protocol != candidate_protocol or a.arm == b.arm:
        return False
    if (a.scene_id, a.road_id, a.stratum, a.replicate_id) != (
        b.scene_id,
        b.road_id,
        b.stratum,
        b.replicate_id,
    ):
        return False
    if reference_protocol.seed_mode != "unseeded":
        return a.seed is not None and a.seed == b.seed
    return True


def pair_records(
    records: Iterable[RolloutRecord],
    spec: ComparisonSpec,
    *,
    candidate_protocol: Protocol | None = None,
) -> PairingReport:
    """Admit all records or raise, preserving required-but-absent strata explicitly.

    A and B identify the reference/candidate roles in ``spec``. Both arms use its protocol
    unless an explicit candidate protocol is supplied; any difference rejects admission.
    The caller must supply the actual protocol, not label incompatible records alike.
    No inner join, filtering of failed runs, or pooling of replicates is performed.
    """
    candidate_protocol = spec.protocol if candidate_protocol is None else candidate_protocol
    if candidate_protocol != spec.protocol:
        raise PairingError("candidate protocol differs from reference protocol")
    records = tuple(records)
    indexed = {}
    scene_metadata = {}
    for record in records:
        if not isinstance(record, RolloutRecord):
            raise PairingError("each input must be a RolloutRecord")
        if not record.completed:
            raise PairingError(f"incomplete rollout: {record.scene_id}/{record.arm}")
        if record.stratum not in spec.required_strata:
            raise PairingError(f"unexpected stratum: {record.stratum}")
        if spec.metric.kind == "binary" and record.value not in (0, 1):
            raise PairingError("binary metric values must be exactly zero or one")
        metadata = (record.road_id, record.stratum)
        previous = scene_metadata.setdefault(record.scene_id, metadata)
        if previous != metadata:
            raise PairingError(f"scene metadata changed: {record.scene_id}")
        key = (record.scene_id, record.replicate_id, record.arm)
        if key in indexed:
            raise PairingError(f"duplicate rollout identity: {key}")
        indexed[key] = record

    rows = []
    # Canonical ordering makes results insensitive to input file order. String and integer
    # replicate IDs stay distinct, including 0 versus "0".
    keys = sorted(
        {(r.scene_id, r.replicate_id) for r in records},
        key=lambda key: (key[0], type(key[1]).__name__, str(key[1])),
    )
    for scene_id, replicate_id in keys:
        a = indexed.get((scene_id, replicate_id, "A"))
        b = indexed.get((scene_id, replicate_id, "B"))
        if a is None or b is None:
            raise PairingError(f"unmatched rollout identity: {scene_id}/{replicate_id}")
        if not is_pair(a, b, spec.protocol, candidate_protocol):
            raise PairingError(f"seed mismatch or missing seed: {scene_id}/{replicate_id}")
        row = {
            "scene_id": scene_id,
            "road_id": a.road_id,
            "stratum": a.stratum,
            "replicate_id": replicate_id,
            "a_value": a.value,
            "b_value": b.value,
            "a_deviation_m": a.deviation_m,
            "b_deviation_m": b.deviation_m,
            "a_seed": a.seed,
            "b_seed": b.seed,
        }
        if spec.metric.kind == "binary":
            failure_value = 1 if spec.metric.higher_is_worse else 0
            row.update(a_fail=a.value == failure_value, b_fail=b.value == failure_value)
        rows.append(row)

    present_strata = {row["stratum"] for row in rows}
    return PairingReport(
        rows=tuple(rows),
        n_records=len(records),
        n_pairs=len(rows),
        n_scenes=len({row["scene_id"] for row in rows}),
        n_roads=len({row["road_id"] for row in rows}),
        missing_strata=tuple(name for name in spec.required_strata if name not in present_strata),
    )
