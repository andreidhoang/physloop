"""S2 operational power and design sensitivity through the actual gate (CPU-06 remainder, W4).

Serves S2.6 [C] (G1, G3-R1). Agent-authored evaluator engineering under D-018/D-024
(build-spec A7). Every simulated trial is decided by the protected path: fixed-sample
designs by ``gate.api.evaluate`` (pairing admission, then ``gate.verdict.verdict``) and
registered-look designs by ``gate.contracts.pair_records`` then
``gate.sequential.SequentialGate``. Zero-difference thresholds use the protected
``gate.verdict.paired_road_bounds``. No bound, average or normal approximation is
re-implemented here.

Verification level per function (build-spec section 1), all L1:

- ``zero_difference_upper`` / ``zero_difference_minimum_roads`` - literal known answers
  (36/51/72/183 roads at alpha 0.05 for margins 0.10/0.07/0.05/0.02; previous integer fails);
  the search is capped at ``MINIMUM_ROADS_SEARCH_CAP`` roads and returns ``None`` beyond it.
- ``generate_rows`` - irregular-layout generator extension; equals the locked
  ``gate.synthetic.generate`` byte-for-byte on regular layouts (same draw order); marginal
  and cluster known answers on irregular layouts.
- ``run_sensitivity`` - deterministic 0/1 known answers, an exact binary enumeration oracle
  and fixed-seed Monte Carlo regressions; its superiority section reproduces the D-023
  441-road literals of ``evidence/superiority-rule-20261108-v2/power.json`` and equals
  ``gate.superiority_power.design_power`` on the same rows (module F4).

Superiority and falsifier power (completeness audit "S2.6 superiority/sequential"; D-023):
``design.superiority`` is a required key of design schema 2, ``null`` when unregistered (a
schema-1 design has no such key and reads the same way; review F4-07) (the report's
superiority section is then ``blocked`` with ``superiority_unregistered``; nothing is
defaulted) or ``{delta_min, primary, pooled_label, guards, gating_strata}``. Each trial's
gate result is classified by the protected-path consumer ``gate.superiority.decide`` (imported,
never copied): ``primary`` ``registered_strata`` decides on the stratified result already
computed for non-inferiority; ``pooled`` re-evaluates the same records relabelled to
``pooled_label`` through ``gate.api.evaluate``. ``guards`` are same-endpoint non-inferiority
guards, one ``gate.api.evaluate`` per guard at its own ``margin`` and the design alpha on the
registered strata, required GO on ``gating_strata`` (a nonempty registered subset; ``[]``
exactly when there are no guards). Registered looks are supported through
``gate.sequential.SequentialGate``: the decision reads the terminal look's result with that
``look`` and ``terminal`` (a stratum closes once non-inferior, so superiority is in effect
tested at the non-inferiority stopping look); pooled primaries and guards are not modelled
under looks and are rejected. Per scenario the report gives the full outcome distribution,
``improvement_power`` (trials whose improvement claim holds: IMPROVEMENT_SUPPORTED plus
IMPROVEMENT_SUPPORTED_BELOW_MIN) and ``falsifier_power`` (trials whose ruled-out claim
``claims.ruled_out.holds`` holds, i.e. lower > -delta_min on every registered stratum: the
D-023 per-stratum falsifier condition, counted as ``gate.superiority.aggregate_runs`` counts
claims; besides MEANINGFUL_IMPROVEMENT_RULED_OUT and IMPROVEMENT_SUPPORTED_BELOW_MIN it
includes REGRESSION_FLAGGED, GUARD_REGRESSION_FLAGGED and GUARD_NOT_CERTIFIED trials whose
primary strata rule the improvement out, so a certain regression has falsifier power 1; agent
technical default (D-018), review F4-01), each with the binomial standard error and
Clopper-Pearson interval. These are assumptions about a candidate rule, not
Huy's registered strata, guards, margins or delta_min.

Every scenario value is an assumption, not a measurement: a report never claims measured RL
variance, and real same-policy measurements may update only nuisance inputs (rates,
discordance, dependence), keeping an effect/discordance sensitivity range. A planner can
reveal poor power; it cannot authorize an inference-method change or tune margins/strata
from pilot outcomes. Design values (strata, margin, alpha, looks) are inputs with no
defaults; this module never chooses Huy's registered values.

Laws (``LAWS``):

- ``paired_binary_copy_v1``: the ``gate.synthetic`` joint law. Per stratum, categories
  (neither, A-only, B-only, both) = (1-p_a-q_plus, q_minus, q_plus, p_a-q_minus), so
  P(A) = p_a, P(B) = p_a+q_plus-q_minus, discordance = q_plus+q_minus. Infeasible laws
  are rejected. ``stratum_effects`` maps a stratum to ``[q_plus, q_minus]``.
- ``paired_continuous_uniform_shift_v1``: reference ``a = low + (high-low) u``; gate
  difference ``d = candidate_worse_by + noise (2v-1)``; candidate ``b = a + d`` when higher
  is worse, else ``b = a - d``. Hence E[a] = (low+high)/2 and E[d] = candidate_worse_by,
  with d bounded in [cwb-noise, cwb+noise]. Laws whose candidate range leaves [0,1] are
  rejected (never clipped or binarized); only float rounding within 1e-12 of the boundary is
  clipped. ``stratum_effects`` maps a stratum to its ``candidate_worse_by``.

Dependence (both laws): ``road_copy_probability`` and ``replicate_copy_probability`` are
categorical copy probabilities as in ``gate.synthetic`` (labelled
``copy_probability_not_icc``), not calibrated intraclass correlations. Replicate 0 is the
scene latent; later replicates copy it or draw independently. Extra scenes or replicates
never add independent roads.

Seeding: trial ``t`` uses ``SeedSequence([seed, t])`` for every scenario (common random
numbers across scenarios); each stratum then draws from ``SeedSequence([trial_seed, i])``
with ``i`` its index in sorted stratum order. Monte Carlo uncertainty is reported as the
binomial plug-in standard error and, separately, a two-sided 95% Clopper-Pearson interval;
a finite interval has no guarantee of covering the exact probability on every run.

Interface reconciliation (recorded, not silently chosen): build-spec A7 names
``run_sensitivity(design, scenarios, n_sim, seed)``; readiness CPU-06 names
``run_sensitivity(design, scenarios, rng_seed, trial_count)``. The build-spec order is used.

Agent technical defaults (D-018), inherited by Huy unless changed: the claim-based
``falsifier_power`` count above; binomial plug-in standard
error plus a separate two-sided 95% Clopper-Pearson interval; common random numbers across
scenarios (``SeedSequence([seed, t])``); clipping of candidate values only within 1e-12 float
rounding of [0, 1]; ``warning`` severity for ``zero_difference_control_not_certifiable``,
``zero_difference_minimum_not_found_within_cap`` and ``stratum_without_roads`` (diagnostics,
never blockers); the all-zero minimum search cap ``MINIMUM_ROADS_SEARCH_CAP`` = 10**6 roads
(``minimum_roads_at_alpha`` null beyond it); roads assigned to looks in layout order;
synthetic protocol fields (``synthetic_seeded`` with per-pair seed = row index).

This module never reads files other than its own source (producer identity), never performs
network or subprocess calls, never edits or copies ``gate/verdict.py``/``gate/stats/`` and
never emits a scientific prediction.
"""

import hashlib
import math
from dataclasses import replace
from numbers import Integral
from pathlib import Path

import numpy as np
from scipy.stats import beta

from gate import api
from gate import superiority as sup
from gate.contracts import ComparisonSpec, MetricSpec, Protocol, RolloutRecord, pair_records
from gate.s2_identity import identity_hash
from gate.sequential import SequentialGate
from gate.verdict import paired_road_bounds

ARTIFACT_KIND = "s2_power_sensitivity"
SCHEMA_VERSION = 1
# Review F4-07: schema 2 added the required ``superiority`` slot; schema 1 (no slot) stays
# readable and its superiority is recorded as unregistered (blocked), never defaulted.
DESIGN_SCHEMA_VERSION = 2
DESIGN_SCHEMA_VERSIONS = (1, 2)
BINARY_LAW = "paired_binary_copy_v1"
CONTINUOUS_LAW = "paired_continuous_uniform_shift_v1"
LAWS = (BINARY_LAW, CONTINUOUS_LAW)
LAW_KIND = {BINARY_LAW: "binary", CONTINUOUS_LAW: "continuous"}
LAW_PARAMETERS = {
    BINARY_LAW: ("p_a", "q_plus", "q_minus"),
    CONTINUOUS_LAW: ("reference_low", "reference_high", "candidate_worse_by", "noise_half_width"),
}
OUTCOMES = ("GO", "NO-GO", "INSUFFICIENT")
DESIGN_FIELDS = (
    "schema_version",
    "design_id",
    "required_strata",
    "margin",
    "alpha",
    "metric",
    "layout",
    "looks",
    "superiority",
)
DESIGN_FIELDS_V1 = tuple(field for field in DESIGN_FIELDS if field != "superiority")
SUPERIORITY_FIELDS = ("delta_min", "primary", "pooled_label", "guards", "gating_strata")
SUPERIORITY_PRIMARIES = ("registered_strata", "pooled")
GUARD_FIELDS = ("name", "margin")
IMPROVEMENT_DEFINITION = (
    "trials whose improvement claim holds: IMPROVEMENT_SUPPORTED + "
    "IMPROVEMENT_SUPPORTED_BELOW_MIN")
# Agent technical default (D-018), review F4-01: the falsifier counts the ruled-out *claim*
# (the D-023 per-stratum condition, as gate.superiority.aggregate_runs counts claims), not
# the two outcomes named after it; regression-flagged and guard-limited trials can count.
FALSIFIER_DEFINITION = (
    "trials whose ruled-out claim holds (claims.ruled_out.holds: lower > -delta_min on every "
    "registered stratum), whatever the outcome label: MEANINGFUL_IMPROVEMENT_RULED_OUT, "
    "IMPROVEMENT_SUPPORTED_BELOW_MIN, and also REGRESSION_FLAGGED, GUARD_REGRESSION_FLAGGED "
    "or GUARD_NOT_CERTIFIED trials whose primary strata rule it out; agent technical default "
    "(D-018)")
SCENARIO_FIELDS = (
    "scenario_id",
    "information_role",
    "law",
    "parameters",
    "dependence",
    "stratum_effects",
)
DEPENDENCE_FIELDS = ("road_copy_probability", "replicate_copy_probability")
LOOK_FIELDS = ("weights", "futility_at", "roads_per_look")
METRIC_FIELDS = ("name", "kind", "higher_is_worse", "unit")
INTERVAL_METHOD = "clopper_pearson_two_sided_0.95"
ROUNDING_TOLERANCE = 1e-12
# Agent technical default (D-018): the all-zero minimum is searched only up to this many roads.
MINIMUM_ROADS_SEARCH_CAP = 10**6


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _is_int(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_real(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _probability(value, name, *, open_interval=False):
    _require(_is_real(value), f"{name} must be a finite number")
    if open_interval:
        _require(0 < value < 1, f"{name} must lie strictly between 0 and 1")
    else:
        _require(0 <= value <= 1, f"{name} must lie in [0, 1]")


def _exact_keys(value, keys, name):
    _require(type(value) is dict, f"{name} must be a JSON object")
    for key in keys:
        _require(key in value, f"{name}.{key} is required")
    unknown = sorted(set(value) - set(keys), key=str)
    _require(not unknown, f"{name} has unknown fields: {unknown}")


# --- zero-difference control (W4) -------------------------------------------------------


def zero_difference_upper(n_roads, alpha):
    """Protected upper bound when every one of ``n_roads`` road differences is zero (L1)."""
    _require(_is_int(n_roads) and n_roads >= 1, "n_roads must be a positive integer")
    _probability(alpha, "alpha", open_interval=True)
    return paired_road_bounds([0.0] * int(n_roads), alpha)[1]


def zero_difference_minimum_roads(alpha, margin, *, max_roads=MINIMUM_ROADS_SEARCH_CAP):
    """Smallest road count whose all-zero upper bound is strictly below ``margin`` (L1).

    The closed form ``R > log(alpha/2) / log(1-margin)`` only seeds the search; the
    protected bound (including its outward padding) decides, and the previous integer is
    verified to fail. This is the identical-output control threshold, not a universal
    minimum: strong improvements can certify with fewer roads.

    Returns ``None`` when no road count up to ``max_roads`` certifies (review finding
    A7-05). The protected bound pads every upper endpoint outward by 1e-12, so a margin at
    or below that padding never certifies at any road count, and tiny margins would
    otherwise need millions to trillions of roads. The seed is checked against the cap
    before any bound is evaluated, so nothing beyond ``max_roads`` is ever allocated.
    ``max_roads`` defaults to ``MINIMUM_ROADS_SEARCH_CAP`` (agent technical default, D-018).
    """
    _probability(alpha, "alpha", open_interval=True)
    _probability(margin, "margin", open_interval=True)
    _require(_is_int(max_roads) and max_roads >= 1, "max_roads must be a positive integer")
    max_roads = int(max_roads)
    seed = math.log(alpha / 2) / math.log1p(-margin)
    if not math.isfinite(seed) or seed >= max_roads:
        return None
    n = max(1, math.floor(seed) + 1)
    while zero_difference_upper(n, alpha) >= margin:
        if n >= max_roads:
            return None
        n += 1
    while n > 1 and zero_difference_upper(n - 1, alpha) < margin:
        n -= 1
    return n


# --- input validation ------------------------------------------------------------------


def _check_layout(layout, required):
    _require(type(layout) is dict, "design.layout must map each stratum to a list of roads")
    _require(
        set(layout) == set(required),
        "design.layout must list exactly the registered strata (an empty list declares a "
        "stratum without roads)",
    )
    for stratum, roads in layout.items():
        _require(type(roads) is list, f"design.layout.{stratum} must be a list of roads")
        for index, road in enumerate(roads):
            _require(
                type(road) is list and road,
                f"design.layout.{stratum}[{index}] must be a nonempty list of scenes",
            )
            for replicates in road:
                _require(
                    _is_int(replicates) and replicates >= 1,
                    f"design.layout.{stratum}[{index}] replicate counts must be positive "
                    "integers",
                )


def _check_looks(looks, design, required):
    """Return (weights, futility, roads_per_look) or None for a fixed-sample design."""
    if looks is None:
        return None
    _exact_keys(looks, LOOK_FIELDS, "design.looks")
    weights = looks["weights"]
    _require(
        type(weights) is list and weights and all(_is_real(w) for w in weights),
        "design.looks.weights must be a nonempty list of numbers",
    )
    futility = looks["futility_at"]
    _require(type(futility) is dict, "design.looks.futility_at must be an object")
    converted = {}
    for key, strata in futility.items():
        _require(
            type(key) is str and key.isascii() and key.isdigit() and not key.startswith("0"),
            "design.looks.futility_at keys must be positive look numbers",
        )
        _require(
            type(strata) is list and all(type(s) is str for s in strata),
            "design.looks.futility_at values must be lists of strata",
        )
        converted[int(key)] = tuple(strata)
    # SequentialGate validates weights and the futility schedule against the strata.
    SequentialGate(
        required,
        design["margin"],
        design["alpha"],
        tuple(weights),
        design["metric"]["higher_is_worse"],
        converted,
    )
    counts = looks["roads_per_look"]
    _require(
        type(counts) is dict and set(counts) == set(required),
        "design.looks.roads_per_look must list every registered stratum",
    )
    for stratum, per_look in counts.items():
        _require(
            type(per_look) is list
            and len(per_look) == len(weights)
            and all(_is_int(c) and c >= 1 for c in per_look),
            f"design.looks.roads_per_look.{stratum} needs one positive road count per look "
            "(every look must cover every active stratum)",
        )
        _require(
            sum(per_look) == len(design["layout"][stratum]),
            f"design.looks.roads_per_look.{stratum} must assign every road in the layout",
        )
    return tuple(weights), converted, counts


def _design_superiority(design):
    """The registered superiority design; ``None`` (unregistered) for a schema-1 design."""
    return design["superiority"] if design["schema_version"] == 2 else None


def _check_design(design):
    _require(type(design) is dict, "design must be a JSON object")
    version = design.get("schema_version")
    _require(
        _is_int(version) and version in DESIGN_SCHEMA_VERSIONS,
        f"design.schema_version must be integer 2 (or 1, read without a superiority slot); "
        f"got {version!r}",
    )
    _exact_keys(design, DESIGN_FIELDS if version == 2 else DESIGN_FIELDS_V1, "design")
    _require(
        type(design["design_id"]) is str and design["design_id"].strip(),
        "design.design_id must be a nonempty string",
    )
    required = design["required_strata"]
    _require(
        type(required) is list
        and required
        and all(type(s) is str and s.strip() for s in required)
        and len(set(required)) == len(required),
        "design.required_strata must be a nonempty list of distinct strings",
    )
    _probability(design["margin"], "design.margin", open_interval=True)
    _probability(design["alpha"], "design.alpha", open_interval=True)
    _exact_keys(design["metric"], METRIC_FIELDS, "design.metric")
    try:
        MetricSpec(**design["metric"])
    except TypeError as exc:
        raise ValueError(f"design.metric: {exc}") from exc
    _check_layout(design["layout"], required)
    looks = _check_looks(design["looks"], design, tuple(required))
    _require(any(design["layout"][s] for s in required), "design.layout has no roads at all")
    _check_superiority(_design_superiority(design), tuple(required), looks)
    return tuple(required), looks


def _check_superiority(value, required, looks):
    """Validate ``design.superiority`` (null = unregistered); every field required."""
    if value is None:
        return
    _exact_keys(value, SUPERIORITY_FIELDS, "design.superiority")
    path = "design.superiority"
    delta = value["delta_min"]
    _require(_is_real(delta) and 0 < delta <= 1, f"{path}.delta_min must lie in (0, 1]")
    primary = value["primary"]
    _require(type(primary) is str and primary in SUPERIORITY_PRIMARIES,
             f"{path}.primary must be one of {list(SUPERIORITY_PRIMARIES)}")
    label = value["pooled_label"]
    if primary == "pooled":
        _require(type(label) is str and label.strip(),
                 f"{path}.pooled_label must name the pooled stratum for a pooled primary")
        _require(len(required) == 1 or label not in required,
                 f"{path}.pooled_label must not name one of several registered strata")
    else:
        _require(label is None, f"{path}.pooled_label applies only to a pooled primary (null)")
    guards = value["guards"]
    _require(type(guards) is list, f"{path}.guards must be a list (empty for no guards)")
    names = []
    for index, guard in enumerate(guards):
        _exact_keys(guard, GUARD_FIELDS, f"{path}.guards[{index}]")
        _require(type(guard["name"]) is str and guard["name"].strip(),
                 f"{path}.guards[{index}].name must be a nonempty string")
        _probability(guard["margin"], f"{path}.guards[{index}].margin", open_interval=True)
        names.append(guard["name"])
    _require(len(set(names)) == len(names), f"{path}.guards names must be unique")
    gating = value["gating_strata"]
    _require(type(gating) is list and all(type(s) is str for s in gating)
             and len(set(gating)) == len(gating),
             f"{path}.gating_strata must be a list of distinct strata")
    if guards:
        _require(gating and set(gating) <= set(required),
                 f"{path}.gating_strata must be a nonempty subset of the registered strata "
                 "when guards are registered")
    else:
        _require(not gating, f"{path}.gating_strata must be [] when no guard is registered")
    if looks is not None:
        _require(primary == "registered_strata" and not guards,
                 f"{path}: under registered sequential looks only a registered_strata primary "
                 "without guards is modelled (guards would close at other looks)")


def _binary_probabilities(p_a, q_plus, q_minus, name):
    for label, value in (("p_a", p_a), ("q_plus", q_plus), ("q_minus", q_minus)):
        _probability(value, f"{name}.{label}")
    # Same expression and float order as gate.synthetic._pair_probabilities.
    probabilities = (1 - p_a - q_plus, q_minus, q_plus, p_a - q_minus)
    _require(
        all(value >= 0 for value in probabilities),
        f"{name}: infeasible joint law (requires q_minus <= p_a and p_a + q_plus <= 1)",
    )
    return probabilities


def _check_continuous(low, high, worse_by, noise, higher_is_worse, name):
    for label, value in (
        ("reference_low", low),
        ("reference_high", high),
        ("candidate_worse_by", worse_by),
        ("noise_half_width", noise),
    ):
        _require(_is_real(value), f"{name}.{label} must be a finite number")
    _require(0 <= low <= high <= 1, f"{name}: reference range must satisfy 0 <= low <= high <= 1")
    _require(noise >= 0, f"{name}.noise_half_width must be nonnegative")
    _require(-1 <= worse_by <= 1, f"{name}.candidate_worse_by must lie in [-1, 1]")
    if higher_is_worse:
        b_low, b_high = low + worse_by - noise, high + worse_by + noise
    else:
        b_low, b_high = low - worse_by - noise, high - worse_by + noise
    _require(
        b_low >= -ROUNDING_TOLERANCE and b_high <= 1 + ROUNDING_TOLERANCE,
        f"{name}: unsupported law, candidate values leave [0, 1] "
        f"([{b_low:.6g}, {b_high:.6g}]); bounded laws are rejected, never clipped",
    )


def _check_scenario(scenario, index, design, required):
    name = f"scenarios[{index}]"
    _exact_keys(scenario, SCENARIO_FIELDS, name)
    for key in ("scenario_id", "information_role"):
        _require(
            type(scenario[key]) is str and scenario[key].strip(),
            f"{name}.{key} must be a nonempty string",
        )
    law = scenario["law"]
    _require(law in LAWS, f"{name}.law is unsupported: {law!r}; supported laws are {LAWS}")
    _require(
        LAW_KIND[law] == design["metric"]["kind"],
        f"{name}.law {law} is unsupported for a {design['metric']['kind']} metric "
        "(scores are never binarized)",
    )
    _exact_keys(scenario["parameters"], LAW_PARAMETERS[law], f"{name}.parameters")
    _exact_keys(scenario["dependence"], DEPENDENCE_FIELDS, f"{name}.dependence")
    for key in DEPENDENCE_FIELDS:
        _probability(scenario["dependence"][key], f"{name}.dependence.{key}")
    effects = scenario["stratum_effects"]
    _require(type(effects) is dict, f"{name}.stratum_effects must be an object")
    _require(
        set(effects) <= set(required),
        f"{name}.stratum_effects names an unregistered stratum",
    )
    parameters = scenario["parameters"]
    if law == BINARY_LAW:
        _binary_probabilities(parameters["p_a"], parameters["q_plus"], parameters["q_minus"],
                              f"{name}.parameters")
        for stratum, effect in effects.items():
            _require(
                type(effect) is list and len(effect) == 2,
                f"{name}.stratum_effects.{stratum} must be [q_plus, q_minus]",
            )
            _binary_probabilities(parameters["p_a"], *effect, f"{name}.stratum_effects.{stratum}")
    else:
        for stratum in required:
            _check_continuous(
                parameters["reference_low"],
                parameters["reference_high"],
                effects.get(stratum, parameters["candidate_worse_by"]),
                parameters["noise_half_width"],
                design["metric"]["higher_is_worse"],
                f"{name}.stratum_effects.{stratum}" if stratum in effects else name,
            )


# --- generator extension ---------------------------------------------------------------


def _expand(roads):
    """Per-scene road index and per-rollout (road, scene, replicate) in canonical order."""
    road_of_scene = []
    replicates = []
    for road_index, road in enumerate(roads):
        for count in road:
            road_of_scene.append(road_index)
            replicates.append(int(count))
    return np.asarray(road_of_scene, dtype=np.int64), replicates


def _draw(rng, size, p):
    """Binary category draws with probabilities ``p``, else (u, v) uniform pairs."""
    return rng.choice(4, size=size, p=p) if p is not None else rng.random((size, 2))


def _latents(rng, roads, road_copy, replicate_copy, p):
    """Return per-rollout latents in road/scene/replicate order (``p`` as in ``_draw``).

    Draw order matches ``gate.synthetic.generate`` with every array flattened in C order:
    road latents, scene latents, road-copy flags, then (if any replicate beyond the first)
    independent replicate latents and scene-copy flags.
    """
    road_of_scene, replicates = _expand(roads)
    road_latents = _draw(rng, len(roads), p)
    scene_latents = _draw(rng, len(road_of_scene), p)
    copy_road = rng.random(len(road_of_scene)) < road_copy
    mask = copy_road if scene_latents.ndim == 1 else copy_road[:, None]
    scene_latents = np.where(mask, road_latents[road_of_scene], scene_latents)
    extra = sum(count - 1 for count in replicates)
    if extra:
        independent = _draw(rng, extra, p)
        copy_scene = rng.random(extra) < replicate_copy
    latents, cursor = [], 0
    for scene_index, count in enumerate(replicates):
        latents.append(scene_latents[scene_index])
        for _ in range(count - 1):
            latents.append(
                scene_latents[scene_index] if copy_scene[cursor] else independent[cursor]
            )
            cursor += 1
    return latents


def _identities(stratum, roads):
    for road_index, road in enumerate(roads):
        road_id = f"{stratum}/road-{road_index:06d}"
        for scene_index, count in enumerate(road):
            scene_id = f"{road_id}/scene-{scene_index:04d}"
            for replicate in range(count):
                yield road_id, scene_id, replicate


def _generate(design, scenario, seed):
    parameters = scenario["parameters"]
    dependence = scenario["dependence"]
    effects = scenario["stratum_effects"]
    higher_is_worse = design["metric"]["higher_is_worse"]
    rows = []
    for stratum_index, stratum in enumerate(sorted(design["layout"])):
        roads = design["layout"][stratum]
        if not roads:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), stratum_index]))
        road_copy = dependence["road_copy_probability"]
        replicate_copy = dependence["replicate_copy_probability"]
        identities = _identities(stratum, roads)
        if scenario["law"] == BINARY_LAW:
            q_plus, q_minus = effects.get(stratum, (parameters["q_plus"], parameters["q_minus"]))
            p = _binary_probabilities(parameters["p_a"], q_plus, q_minus, "law")
            latents = _latents(rng, roads, road_copy, replicate_copy, p)
            for (road_id, scene_id, replicate), category in zip(identities, latents, strict=True):
                category = int(category)
                rows.append(
                    {
                        "scene_id": scene_id,
                        "road_id": road_id,
                        "stratum": stratum,
                        "replicate_id": replicate,
                        "a_fail": category in (1, 3),
                        "b_fail": category in (2, 3),
                    }
                )
            continue
        worse_by = effects.get(stratum, parameters["candidate_worse_by"])
        low, high = parameters["reference_low"], parameters["reference_high"]
        noise = parameters["noise_half_width"]
        latents = _latents(rng, roads, road_copy, replicate_copy, None)
        for (road_id, scene_id, replicate), (u, v) in zip(identities, latents, strict=True):
            a = low + (high - low) * float(u)
            d = worse_by + noise * (2 * float(v) - 1)
            b = a + d if higher_is_worse else a - d
            if not -ROUNDING_TOLERANCE <= b <= 1 + ROUNDING_TOLERANCE:
                raise AssertionError("validated law produced an out-of-range value")
            rows.append(
                {
                    "scene_id": scene_id,
                    "road_id": road_id,
                    "stratum": stratum,
                    "replicate_id": replicate,
                    "a_value": a,
                    "b_value": min(1.0, max(0.0, b)),
                }
            )
    return rows


def generate_rows(design, scenario, seed):
    """Return paired rows for one validated scenario under ``design.layout`` (L1).

    Binary rows carry ``a_fail``/``b_fail`` (the ``gate.synthetic`` shape); continuous rows
    carry ``a_value``/``b_value`` in [0, 1]. Strata with no roads contribute no rows.
    """
    required, _ = _check_design(design)
    _check_scenario(scenario, 0, design, required)
    _require(_is_int(seed) and seed >= 0, "seed must be a nonnegative integer")
    return _generate(design, scenario, seed)


# --- simulation through the gate --------------------------------------------------------


def _spec(design, design_hash, law):
    return ComparisonSpec(
        reference="synthetic_reference",
        candidate="synthetic_candidate",
        protocol=Protocol(
            identity="s2_power_synthetic_protocol",
            seed_mode="synthetic_seeded",
            config_hash=design_hash,
            evaluator="gate.s2_power",
            score_definition=f"synthetic:{law}",
            deviation_convention="not_applicable_synthetic_no_deviation",
        ),
        required_strata=tuple(design["required_strata"]),
        margin=design["margin"],
        alpha=design["alpha"],
        metric=MetricSpec(**design["metric"]),
    )


def _records(rows, metric):
    failure = 1 if metric["higher_is_worse"] else 0
    records = []
    for index, row in enumerate(rows):
        for arm, key in (("A", "a"), ("B", "b")):
            if metric["kind"] == "binary":
                value = failure if row[f"{key}_fail"] else 1 - failure
            else:
                value = row[f"{key}_value"]
            records.append(
                RolloutRecord(
                    arm=arm,
                    scene_id=row["scene_id"],
                    road_id=row["road_id"],
                    stratum=row["stratum"],
                    replicate_id=row["replicate_id"],
                    value=value,
                    completed=True,
                    deviation_m=None,
                    seed=index,
                )
            )
    return records


def _sequential_trial(records, spec, design, looks):
    weights, futility, counts = looks
    look_of_road = {}
    for stratum, per_look in counts.items():
        cursor = 0
        for look, count in enumerate(per_look):
            for road in range(cursor, cursor + count):
                look_of_road[f"{stratum}/road-{road:06d}"] = look
            cursor += count
    rows = []
    for paired in pair_records(records, spec).rows:
        row = dict(paired)
        if spec.metric.kind == "binary":
            # Same normalization as gate.api.evaluate: binary rows carry only failures.
            del row["a_value"], row["b_value"]
        rows.append(row)
    gate = SequentialGate(
        spec.required_strata,
        spec.margin,
        spec.alpha,
        weights,
        spec.metric.higher_is_worse,
        futility,
    )
    used_roads, used_pairs, result = set(), 0, None
    for look in range(len(weights)):
        active = set(gate.active_strata)
        batch = [
            row
            for row in rows
            if look_of_road[row["road_id"]] == look and row["stratum"] in active
        ]
        result = gate.update(batch)
        used_roads.update(row["road_id"] for row in batch)
        used_pairs += len(batch)
        if gate.terminal:
            break
    return result, gate.look, gate.terminal, len(used_roads), used_pairs


def _fixed_decision(records, spec, response, claim):
    """``gate.superiority.decide`` on one fixed-sample trial (primary, pooled, guards)."""
    required = spec.required_strata
    primary, strata = response, required
    if claim["primary"] == "pooled":
        label = claim["pooled_label"]
        pooled = [replace(record, stratum=label) for record in records]
        strata = (label,)
        primary = api.evaluate(pooled, replace(spec, required_strata=strata))
    guards = {}
    for guard in claim["guards"]:
        guard_spec = replace(spec, margin=guard["margin"])
        guards[guard["name"]] = api.evaluate(records, guard_spec)
    return sup.decide(
        primary,
        required_strata=strata,
        guards=guards,
        delta_min=claim["delta_min"],
        guard_strata=required if claim["primary"] == "pooled" and guards else None,
        gating_strata=tuple(claim["gating_strata"]) if guards else None,
    )


def _rate_entry(count, n_sim):
    return {
        "count": count,
        "rate": count / n_sim,
        "standard_error": math.sqrt((count / n_sim) * (1 - count / n_sim) / n_sim),
        "interval": _clopper_pearson(count, n_sim),
    }


def _clopper_pearson(count, n):
    low = 0.0 if count == 0 else float(beta.ppf(0.025, count, n - count + 1))
    high = 1.0 if count == n else float(beta.ppf(0.975, count + 1, n - count))
    return [low, high]


def _derived(scenario, required):
    parameters = scenario["parameters"]
    effects = scenario["stratum_effects"]
    if scenario["law"] == BINARY_LAW:

        def describe(q_plus, q_minus):
            discordance = q_plus + q_minus
            return {
                "reference_event_rate": parameters["p_a"],
                "candidate_event_rate": parameters["p_a"] + q_plus - q_minus,
                "discordance": discordance,
                "mean_difference": q_plus - q_minus,
                "candidate_worse_share_of_discordant": (
                    "not_estimable" if discordance == 0 else q_plus / discordance
                ),
            }

        base = describe(parameters["q_plus"], parameters["q_minus"])
        by_stratum = {
            s: describe(*effects.get(s, (parameters["q_plus"], parameters["q_minus"])))
            for s in required
        }
        return {**base, "by_stratum": by_stratum,
                "difference_convention": "positive means candidate worse"}
    return {
        "mean_gate_difference": parameters["candidate_worse_by"],
        "reference_mean": (parameters["reference_low"] + parameters["reference_high"]) / 2,
        "by_stratum": {
            s: {"mean_gate_difference": effects.get(s, parameters["candidate_worse_by"])}
            for s in required
        },
        "difference_convention": "positive means candidate worse",
    }


def _zero_difference_section(design, required, looks):
    alpha, margin = design["alpha"], design["margin"]
    entries, diagnostics = [], []
    schedule = (
        [(1, alpha, {s: len(design["layout"][s]) for s in required})]
        if looks is None
        else [
            (
                k + 1,
                alpha * weight,
                {s: sum(looks[2][s][: k + 1]) for s in required},
            )
            for k, weight in enumerate(looks[0])
        ]
    )
    for stratum in required:
        stratum_entries = []
        for look, look_alpha, cumulative in schedule:
            n = cumulative[stratum]
            upper = zero_difference_upper(n, look_alpha) if n else None
            stratum_entries.append(
                {
                    "stratum": stratum,
                    "look": look,
                    "alpha": look_alpha,
                    "margin": margin,
                    "roads": n,
                    "upper": upper,
                    "certifiable": upper is not None and upper < margin,
                    "minimum_roads_at_alpha": zero_difference_minimum_roads(look_alpha, margin),
                    "minimum_roads_search_cap": MINIMUM_ROADS_SEARCH_CAP,
                }
            )
        entries.extend(stratum_entries)
        if not design["layout"][stratum]:
            diagnostics.append(
                {
                    "code": "stratum_without_roads",
                    "path": f"design.layout.{stratum}",
                    "severity": "warning",
                    "evidence_refs": [],
                    "message": f"Registered stratum {stratum} has no roads; GO is impossible.",
                }
            )
        if not any(entry["certifiable"] for entry in stratum_entries):
            detail = "; ".join(
                f"look {e['look']}: alpha={e['alpha']:.6g}, R={e['roads']}, "
                f"U0={'n/a' if e['upper'] is None else format(e['upper'], '.6g')}"
                for e in stratum_entries
            )
            diagnostics.append(
                {
                    "code": "zero_difference_control_not_certifiable",
                    "path": f"design.layout.{stratum}",
                    "severity": "warning",
                    "evidence_refs": [],
                    "message": (
                        f"Identical outputs cannot certify margin {margin:.6g} in {stratum} "
                        f"({detail}). Design diagnostic only; not a universal road floor."
                    ),
                }
            )
        uncapped = [e for e in stratum_entries if e["minimum_roads_at_alpha"] is None]
        if uncapped:
            detail = ", ".join(f"look {e['look']} alpha={e['alpha']:.6g}" for e in uncapped)
            diagnostics.append(
                {
                    "code": "zero_difference_minimum_not_found_within_cap",
                    "path": f"design.layout.{stratum}",
                    "severity": "warning",
                    "evidence_refs": [],
                    "message": (
                        f"No road count up to {MINIMUM_ROADS_SEARCH_CAP} certifies margin "
                        f"{margin:.6g} with identical outputs in {stratum} ({detail}); the "
                        "protected bound pads outward by 1e-12. Search capped (agent "
                        "technical default, D-018); minimum_roads_at_alpha is null."
                    ),
                }
            )
    return entries, diagnostics


def _superiority_section(claim, looks, schema_version):
    if claim is None:
        legacy = schema_version == 1
        return {
            "status": "blocked",
            "issues": [{
                "code": "superiority_unregistered",
                "path": "design.schema_version" if legacy else "design.superiority",
                "severity": "blocking",
                "evidence_refs": [],
                "message": (
                    ("A design of schema_version 1 has no superiority slot; " if legacy else "")
                    + "No superiority design registered (delta_min, primary, guards, gating "
                    "strata are owner inputs); superiority and falsifier power not computed."),
            }],
        }
    return {
        "status": "passed",
        "issues": [],
        "rule_id": sup.RULE_ID,
        "inference_path": (
            "gate.superiority.decide" if looks is None
            else "gate.sequential.SequentialGate -> gate.superiority.decide"),
        "delta_min": claim["delta_min"],
        "primary": claim["primary"],
        "pooled_label": claim["pooled_label"],
        "guards": [dict(guard) for guard in claim["guards"]],
        "guard_kind": "same_endpoint_noninferiority_on_registered_strata_at_design_alpha",
        "gating_strata": list(claim["gating_strata"]),
        "classification": "assumption_not_measurement",
        "note": (
            "Each claim keeps its own level; the joint 'improved, by less than delta_min' "
            "statement holds only at the summed level. Sequential designs test superiority at "
            "the non-inferiority stopping look."),
    }


def run_sensitivity(design, scenarios, n_sim, seed):
    """Simulate GO/NO-GO/INSUFFICIENT frequencies of a design under assumed scenarios (L1).

    ``design`` (all fields required): ``schema_version`` 2 (a schema-1 design has no
    ``superiority`` key and is read as superiority unregistered; review F4-07), ``design_id``,
    ``required_strata``, ``margin``, ``alpha``, ``metric`` (``gate.contracts.MetricSpec``
    fields), ``layout`` (stratum -> list of roads, each road a list of per-scene replicate
    counts; an empty list declares a stratum without roads) and ``looks`` (``null`` for a
    fixed-sample analysis, or ``weights``, ``futility_at`` and ``roads_per_look``, which
    assigns complete roads to looks in layout order) and ``superiority`` (``null`` when
    unregistered, else the superiority design of the module docstring). ``scenarios`` is a
    nonempty list of scenario objects (see the module docstring). Invalid inputs raise
    ``ValueError``.
    """
    _require(_is_int(n_sim) and n_sim >= 1, "n_sim must be a positive integer")
    _require(_is_int(seed) and seed >= 0, "seed must be a nonnegative integer")
    required, looks = _check_design(design)
    _require(type(scenarios) is list and scenarios, "scenarios must be a nonempty list")
    for index, scenario in enumerate(scenarios):
        _check_scenario(scenario, index, design, required)
    ids = [scenario["scenario_id"] for scenario in scenarios]
    _require(len(set(ids)) == len(ids), "scenario_id values must be unique")
    design_hash = identity_hash(design)
    scenarios_hash = identity_hash(scenarios)

    trial_seeds = [
        int(np.random.SeedSequence([int(seed), trial]).generate_state(1)[0])
        for trial in range(n_sim)
    ]
    n_looks = 1 if looks is None else len(looks[0])
    claim = _design_superiority(design)
    results = []
    for scenario in scenarios:
        spec = _spec(design, design_hash, scenario["law"])
        counts = dict.fromkeys(OUTCOMES, 0)
        stopped = {str(k): 0 for k in range(1, n_looks + 1)}
        sup_counts = dict.fromkeys(sup.OUTCOMES, 0)
        sup_looks = {str(k): 0 for k in range(1, n_looks + 1)}
        improved = ruled_out = 0
        roads_total = pairs_total = looks_total = 0
        for trial_seed in trial_seeds:
            rows = _generate(design, scenario, trial_seed)
            records = _records(rows, design["metric"])
            if looks is None:
                response = api.evaluate(records, spec)
                outcome = response["verdict"]["overall"]
                look, used_roads, used_pairs = 1, len({r["road_id"] for r in rows}), len(rows)
                decision = (None if claim is None
                            else _fixed_decision(records, spec, response, claim))
            else:
                result, look, terminal, used_roads, used_pairs = _sequential_trial(
                    records, spec, design, looks
                )
                outcome = result.overall
                decision = None if claim is None else sup.decide(
                    result, required_strata=spec.required_strata, guards={},
                    delta_min=claim["delta_min"], look=look, terminal=terminal)
            counts[outcome] += 1
            stopped[str(look)] += 1
            looks_total += look
            roads_total += used_roads
            pairs_total += used_pairs
            if decision is not None:
                sup_counts[decision["outcome"]] += 1
                sup_looks[str(look)] += 1
                improved += decision["claims"]["improvement"]["holds"] is True
                ruled_out += decision["claims"]["ruled_out"]["holds"] is True
        superiority_result = None if claim is None else {
            "outcomes": {name: _rate_entry(count, n_sim) for name, count in sup_counts.items()},
            "improvement_power": {**_rate_entry(improved, n_sim),
                                  "definition": IMPROVEMENT_DEFINITION},
            "falsifier_power": {**_rate_entry(ruled_out, n_sim),
                                "definition": FALSIFIER_DEFINITION},
            "decided_at_look": sup_looks,
        }
        results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "information_role": scenario["information_role"],
                "classification": "assumption_not_measurement",
                "law": scenario["law"],
                "parameters": dict(scenario["parameters"]),
                "dependence": {**scenario["dependence"], "label": "copy_probability_not_icc"},
                "stratum_effects": dict(scenario["stratum_effects"]),
                "derived": _derived(scenario, required),
                "outcomes": {
                    name: {
                        "count": count,
                        "rate": count / n_sim,
                        "standard_error": math.sqrt((count / n_sim) * (1 - count / n_sim)
                                                    / n_sim),
                        "interval": _clopper_pearson(count, n_sim),
                    }
                    for name, count in counts.items()
                },
                "stopped_at_look": stopped,
                "expected_looks": looks_total / n_sim,
                "expected_roads": roads_total / n_sim,
                "expected_rollout_pairs": pairs_total / n_sim,
                "superiority": superiority_result,
            }
        )
    zero_entries, diagnostics = _zero_difference_section(design, required, looks)
    layout_summary = {
        s: {
            "roads": len(design["layout"][s]),
            "scenes": sum(len(road) for road in design["layout"][s]),
            "rollout_pairs": sum(sum(road) for road in design["layout"][s]),
        }
        for s in required
    }
    return {
        "artifact_kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": "synthetic",
        "producer": {"module": "gate.s2_power", "source_sha256": _source_sha256()},
        "input_hashes": {"design": design_hash, "scenarios": scenarios_hash},
        "classification": "assumption_not_measurement",
        "units": {
            "rates": "fraction of simulated trials",
            "expected_roads": "independent roads evaluated per trial",
            "expected_rollout_pairs": "paired rollouts evaluated per trial",
            "margin": "metric difference units (positive means candidate worse)",
        },
        "design_id": design["design_id"],
        "analysis": "fixed_sample" if looks is None else "sequential",
        "inference_path": (
            "gate.api.evaluate" if looks is None else "gate.sequential.SequentialGate"
        ),
        "look_alphas": [design["alpha"]] if looks is None else [
            design["alpha"] * weight for weight in looks[0]
        ],
        "n_sim": n_sim,
        "seed": seed,
        "seeding": "trial t: SeedSequence([seed, t]) shared by every scenario; stratum i: "
                   "SeedSequence([trial_seed, i]) over sorted strata",
        "monte_carlo": {
            "standard_error_method": "binomial_plugin sqrt(rate * (1 - rate) / n_sim)",
            "interval_method": INTERVAL_METHOD,
            "interval_note": (
                "reported separately; a finite Monte Carlo interval has no guarantee of "
                "covering the exact probability on every run"
            ),
        },
        "layout_summary": layout_summary,
        "superiority": _superiority_section(claim, looks, design["schema_version"]),
        "zero_difference_control": zero_entries,
        "diagnostics": diagnostics,
        "scenarios": results,
        "status": "passed",
        "status_meaning": (
            "simulation executed through the protected gate; frequencies describe assumed "
            "scenarios, not measured RL variance or a scientific prediction"
        ),
        "issues": [],
    }
