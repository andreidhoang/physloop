"""Agent-authored paired synthetic truth for S1; no AlpaSim or dataset inputs.

The parameters named ``*_correlation`` are categorical copy probabilities, not
calibrated intraclass correlations. Replicate zero is the scene latent; subsequent
replicates copy it with the chosen probability or draw an independent pair. Thus a
single replicate preserves the road dependence introduced at the scene layer.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real

import numpy as np


def _integer(value, name, *, positive=False):
    lower = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite numeric probability")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")


def _pair_probabilities(p_a, q_plus, q_minus):
    for name, value in (("p_a", p_a), ("q_plus", q_plus), ("q_minus", q_minus)):
        _probability(value, name)
    probabilities = (1 - p_a - q_plus, q_minus, q_plus, p_a - q_minus)
    if any(value < 0 for value in probabilities):
        raise ValueError("joint probabilities require q_minus <= p_a and p_a + q_plus <= 1")
    return probabilities


def generate(
    seed: int,
    strata: dict[str, int],
    *,
    scenes_per_road: int = 1,
    replicates: int = 1,
    p_a: float = 0.07,
    q_plus: float = 0.04,
    q_minus: float = 0.02,
    road_correlation: float = 0,
    replicate_correlation: float = 0,
    effects: dict[str, tuple[float, float]] | None = None,
) -> list[dict]:
    """Generate paired binary rows, with ``strata`` specifying complete road counts.

    In each stratum, category probabilities are (neither, A-only, B-only, both) =
    (1-p_a-q_plus, q_minus, q_plus, p_a-q_minus). Consequently P(A)=p_a,
    P(B)=p_a+q_plus-q_minus, and paired mean difference is q_plus-q_minus.
    Copy mixtures preserve those marginals while inducing cluster dependence.
    ``effects`` overrides (q_plus, q_minus) for named strata, with unchanged p_a.
    """
    _integer(seed, "seed")
    _integer(scenes_per_road, "scenes_per_road", positive=True)
    _integer(replicates, "replicates", positive=True)
    _probability(road_correlation, "road_correlation")
    _probability(replicate_correlation, "replicate_correlation")
    base_probabilities = _pair_probabilities(p_a, q_plus, q_minus)
    if not isinstance(strata, Mapping) or not strata:
        raise ValueError("strata must be a nonempty mapping of names to road counts")
    for name, road_count in strata.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("stratum names must be nonempty strings")
        _integer(road_count, "road count", positive=True)
    effects = {} if effects is None else effects
    if not isinstance(effects, Mapping):
        raise TypeError("effects must be a mapping")
    if set(effects) - set(strata):
        raise ValueError("effects names an unknown stratum")
    probabilities = {name: base_probabilities for name in strata}
    for name, effect in effects.items():
        if not isinstance(effect, (tuple, list)) or len(effect) != 2:
            raise ValueError("each effect must contain (q_plus, q_minus)")
        probabilities[name] = _pair_probabilities(p_a, *effect)

    rows = []
    for stratum_index, name in enumerate(sorted(strata)):
        # Each stratum has its own stream; one stratum's draws cannot consume another's.
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), stratum_index]))
        road_count = int(strata[name])
        p = probabilities[name]
        road_latents = rng.choice(4, size=road_count, p=p)
        scene_latents = rng.choice(4, size=(road_count, scenes_per_road), p=p)
        copy_road = rng.random((road_count, scenes_per_road)) < road_correlation
        scene_latents = np.where(copy_road, road_latents[:, None], scene_latents)
        outcomes = np.empty((road_count, scenes_per_road, replicates), dtype=np.int8)
        outcomes[:, :, 0] = scene_latents
        if replicates > 1:
            shape = (road_count, scenes_per_road, replicates - 1)
            independent = rng.choice(4, size=shape, p=p)
            copy_scene = rng.random(shape) < replicate_correlation
            outcomes[:, :, 1:] = np.where(copy_scene, scene_latents[:, :, None], independent)
        for road in range(road_count):
            road_id = f"{name}/road-{road:06d}"
            for scene in range(scenes_per_road):
                scene_id = f"{road_id}/scene-{scene:04d}"
                for replicate in range(replicates):
                    category = int(outcomes[road, scene, replicate])
                    rows.append(
                        {
                            "scene_id": scene_id,
                            "road_id": road_id,
                            "stratum": name,
                            "replicate_id": replicate,
                            "a_fail": category in (1, 3),
                            "b_fail": category in (2, 3),
                        }
                    )
    return rows


def split_road_batches(rows: Sequence[dict], K: int = 3, seed: int = 0) -> tuple[list[dict], ...]:
    """Predeclare balanced random road batches covering every observed stratum.

    A road remains wholly in one batch, including all its scenes and replicates.
    Roads may not span strata. Each stratum must have at least K roads. This function
    does not inspect outcomes or filter closed strata; the sequential consumer does so.
    """
    _integer(K, "K", positive=True)
    _integer(seed, "seed")
    if not rows:
        raise ValueError("cannot split empty rows")
    road_rows = defaultdict(list)
    road_strata = {}
    seen_pairs = set()
    scene_metadata = {}
    for row in rows:
        for field in ("road_id", "scene_id", "stratum"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"each row requires a nonempty {field}")
        road, scene, stratum = row["road_id"], row["scene_id"], row["stratum"]
        if road_strata.setdefault(road, stratum) != stratum:
            raise ValueError("a road cannot cross strata")
        if scene_metadata.setdefault(scene, (road, stratum)) != (road, stratum):
            raise ValueError("scene metadata changed")
        replicate = row.get("replicate_id", 0)
        if isinstance(replicate, str):
            if not replicate.strip():
                raise ValueError("replicate_id must be nonempty")
        else:
            _integer(replicate, "replicate_id")
        identity = (scene, replicate)
        if identity in seen_pairs:
            raise ValueError("duplicate scene/replicate pair")
        seen_pairs.add(identity)
        road_rows[road].append(dict(row))
    by_stratum = defaultdict(list)
    for road, stratum in road_strata.items():
        by_stratum[stratum].append(road)
    if any(len(roads) < K for roads in by_stratum.values()):
        raise ValueError("a stratum has fewer roads than K; not every look can cover it")
    rng = np.random.default_rng(int(seed))
    batches = tuple([] for _ in range(K))
    for stratum in sorted(by_stratum):
        roads = rng.permutation(sorted(by_stratum[stratum]))
        for batch, road_group in zip(batches, np.array_split(roads, K), strict=True):
            for road in road_group:
                batch.extend(
                    sorted(
                        road_rows[str(road)],
                        key=lambda row: (
                            row["scene_id"],
                            type(row.get("replicate_id", 0)).__name__,
                            str(row.get("replicate_id", 0)),
                        ),
                    )
                )
    return batches
