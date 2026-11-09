"""Candidate superiority rule over locked-gate outputs. Agent-authored under D-018/D-023/D-024.

Tickets: S2.6/S4.6 rule; R4 C1 (a guard regression blocks the improvement claim, not only the
outcome), R4 C4 (same-strata guards must cover the primary's per-stratum observations), R4 C10
(regression precedence and ``joint_level`` in run aggregation). Verification level: L1 for every
function (pure logic over gate outputs, literal known answers and negative controls on CPU).
Never: recomputes a bound, edits or copies ``gate/verdict.py`` / ``gate/stats/``, chooses a
registered value (strata, guards, ``delta_min``, run ids or run rule), or authorizes anything.

Serves S2.6 [C] (superiority sensitivity) and S4.6 [C] (an improvement claim needs a registered
superiority rule). This module **reads** the per-stratum endpoints that the protected
``gate.verdict.verdict`` and ``gate.sequential.SequentialGate`` already return; it never recomputes
the bound or the replicate/scene/road averaging, and it changes no protected file.

Status: a candidate implementation for Huy's registration, not a registered scientific rule. The
registered strata, the guard endpoints, the smallest improvement of interest ``delta_min``, the run
identities and the run-aggregation rule are required inputs with no defaults.

Conventions (from the gate): a stratum's difference is positive when the candidate is worse. Its
``upper`` endpoint covers the true difference from above with probability >= 1 - alpha and its
``lower`` endpoint from below, *separately*; the pair is not a joint 1 - alpha interval
(``docs/S1_CORRECTNESS.md``). Under the S1 assumptions (independent bounded road differences):

- **improvement claim**: ``upper < 0`` on every registered stratum, every gating guard stratum GO,
  no primary NO-GO and no guard NO-GO in any guard stratum (gating or not). An intersection-union
  test at level max(component alphas); no adjustment across strata. ``claims.improvement.holds``
  is never True when the outcome is a regression (R4 C1);
- **ruled-out claim**: ``lower > -delta_min`` on every registered stratum. An intersection-union test
  at the primary alpha that no improvement of delta_min or more exists on any stratum. It is not the
  complement of the improvement claim; mixed strata give INSUFFICIENT overall;
- each claim has its own level. When both hold ("improved, by less than delta_min"), the joint
  statement uses both endpoints of one dataset and is guaranteed only at the sum of the levels;
- for a ``SequentialGate`` result each stratum's endpoints come from one look at that look's spent
  alpha; by the union bound over looks each claim keeps the registered total alpha, including under
  data-dependent stopping. The gate closes a stratum once it is non-inferior, so superiority is in
  effect tested at the non-inferiority stopping look; register that, or use a fixed-sample analysis.

Per-stratum classes and the ``worse_bound`` flags are descriptive and unadjusted. A per-stratum
claim, or a claim that "some stratum is worse", needs its own multiplicity control. The protected
gate's Hoeffding regression flag can stay silent when the KL bound already shows the candidate
worse (lower > 0); the flag reports that case without changing the gate.
"""

import json
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass

import numpy as np

RULE_ID = "intersection_union_upper_below_zero_v4"

# Per-stratum classes
AT_LEAST_MIN_NOT_EXCLUDED = "IMPROVED_AT_LEAST_MIN_NOT_EXCLUDED"
REAL_BELOW_MIN = "IMPROVED_BELOW_MIN"
RULED_OUT = "MEANINGFUL_IMPROVEMENT_RULED_OUT"
INSUFFICIENT = "INSUFFICIENT"
MISSING = "MISSING_STRATUM"

# Overall outcomes (RULED_OUT and INSUFFICIENT are shared names), in precedence order
REGRESSION_FLAGGED = "REGRESSION_FLAGGED"
GUARD_REGRESSION_FLAGGED = "GUARD_REGRESSION_FLAGGED"
IMPROVED_BELOW_MIN = "IMPROVEMENT_SUPPORTED_BELOW_MIN"
SUPPORTED = "IMPROVEMENT_SUPPORTED"
GUARD_NOT_CERTIFIED = "GUARD_NOT_CERTIFIED"
CONFLICTING = "CONFLICTING_RUNS"
OUTCOMES = (
    REGRESSION_FLAGGED,
    GUARD_REGRESSION_FLAGGED,
    IMPROVED_BELOW_MIN,
    SUPPORTED,
    GUARD_NOT_CERTIFIED,
    RULED_OUT,
    INSUFFICIENT,
)

FIXED_METHOD = "road_positive_negative_KL; Hoeffding_Holm_regression"
SEQUENTIAL_METHOD = "finite_look_spent_alpha_road_KL"
KNOWN_METHODS = (FIXED_METHOD, SEQUENTIAL_METHOD)
STATES = ("GO", "NO-GO", "INSUFFICIENT")
RUN_RULES = ("all", "at_least_k")


def _real(value):
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def _check_delta_min(delta_min):
    if not _real(delta_min) or not math.isfinite(delta_min) or not 0 < delta_min <= 1:
        raise ValueError("delta_min must be a finite number in (0, 1]")


def _check_alpha(alpha, name="alpha"):
    if not _real(alpha) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError(f"{name} must be a finite probability in (0, 1)")


def stratum_outcome(lower, upper, *, delta_min):
    """Classify one stratum from its gate endpoints. Strict inequalities on both sides."""
    _check_delta_min(delta_min)
    if lower is None and upper is None:
        return MISSING
    if lower is None or upper is None:
        raise ValueError("a stratum needs both endpoints or neither")
    for value in (lower, upper):
        if not _real(value) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("endpoints must be finite real numbers in [-1, 1]")
    if lower > upper:
        raise ValueError("lower endpoint exceeds upper endpoint")
    improved = upper < 0
    excludes_min = lower > -delta_min
    if improved:
        return REAL_BELOW_MIN if excludes_min else AT_LEAST_MIN_NOT_EXCLUDED
    return RULED_OUT if excludes_min else INSUFFICIENT


def _normalize(result):
    """Accept a VerdictResult, its asdict() form, or a gate.api.evaluate response.

    Returns (overall, {stratum: (lower, upper, state)}, alpha, margin, method, metric_or_None,
    {stratum: (n_roads, n_scenes, n_replicates) or None}). Rejects inconsistent shapes.
    """
    if is_dataclass(result) and not isinstance(result, type):
        result = asdict(result)
    if not isinstance(result, Mapping):
        raise TypeError("expected a gate verdict result")
    metric = None
    if "verdict" in result:
        if "overall" in result or result.get("schema_version") != 1:
            raise ValueError("ambiguous result shape: expected a gate.api.evaluate response")
        if result.get("analysis_mode") != "fixed_sample" or not isinstance(
            result["verdict"], Mapping
        ):
            raise ValueError("unsupported gate.api response")
        metric = (result.get("comparison") or {}).get("metric")
        result = result["verdict"]
    for key in ("overall", "strata", "details", "alpha", "method", "margin"):
        if key not in result:
            raise ValueError(f"gate result lacks {key!r}")
    if result["method"] not in KNOWN_METHODS:
        raise ValueError(f"unsupported gate method {result['method']!r}")
    _check_alpha(result["alpha"])
    strata, details = result["strata"], result["details"]
    if not isinstance(strata, Mapping) or not isinstance(details, Mapping):
        raise TypeError("strata and details must be mappings")
    if not strata or set(strata) != set(details):
        raise ValueError("strata and details must name the same nonempty set")
    states = list(strata.values())
    if any(s not in STATES for s in states) or result["overall"] not in STATES:
        raise ValueError("unknown gate state")
    if (result["overall"] == "NO-GO") != ("NO-GO" in states):
        raise ValueError("overall NO-GO must match a stratum NO-GO")
    if (result["overall"] == "GO") != all(s == "GO" for s in states):
        raise ValueError("overall GO must match every stratum GO")
    out, counts = {}, {}
    for name in sorted(strata):
        d = details[name]
        if is_dataclass(d):
            d = asdict(d)
        if not isinstance(d, Mapping) or "lower" not in d or "upper" not in d:
            raise ValueError(f"stratum {name!r} lacks endpoints")
        absent = d.get("reason") == "missing_stratum" or d.get("n_roads", 1) == 0
        if absent != (d["lower"] is None and d["upper"] is None):
            raise ValueError(f"stratum {name!r} endpoints disagree with its road count")
        out[name] = (d["lower"], d["upper"], strata[name])
        keys = ("n_roads", "n_scenes", "n_replicates")
        counts[name] = tuple(d[k] for k in keys) if all(k in d for k in keys) else None
    return (
        result["overall"],
        out,
        float(result["alpha"]),
        result["margin"],
        result["method"],
        metric,
        counts,
    )


def _total(counts):
    """Summed (roads, scenes, replicates) over strata; None if any stratum lacks counts."""
    if any(c is None for c in counts.values()):
        raise ValueError("cannot verify paired observations: a stratum lacks its counts")
    return tuple(sum(c[i] for c in counts.values()) for i in range(3))


def _check_same_strata_coverage(name, guard_counts, primary_counts):
    """R4 C4: a guard on the primary's strata must cover the same observations per stratum.

    Each stratum's ``(n_roads, n_scenes, n_replicates)`` must equal the primary's. For a
    ``SequentialGate`` result a stratum keeps the counts of the look at which it closed, so a guard
    that closed at a different look than the primary is rejected here as well (fail closed);
    register guards that close with the primary, or use a fixed-sample analysis.
    """
    for stratum in sorted(primary_counts):
        guard, primary = guard_counts.get(stratum), primary_counts[stratum]
        if guard is None or primary is None:
            raise ValueError(
                f"guard {name!r} stratum {stratum!r}: cannot verify paired observations "
                "(road, scene and replicate counts are required)"
            )
        if tuple(guard) != tuple(primary):
            raise ValueError(
                f"guard {name!r} stratum {stratum!r} covers (n_roads, n_scenes, n_replicates) = "
                f"{tuple(guard)}, the primary {tuple(primary)}: not the same paired observations"
            )


def _strata_tuple(value, name):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of stratum names")
    out = tuple(value)
    if not out or len(set(out)) != len(out) or any(not isinstance(s, str) or not s for s in out):
        raise ValueError(f"register unique nonempty {name}")
    return out


def decide(
    primary,
    *,
    required_strata,
    guards,
    delta_min,
    guard_strata=None,
    gating_strata=None,
    look=None,
    terminal=None,
):
    """Apply the candidate rule to one checkpoint comparison.

    ``primary`` is the gate result for the registered primary endpoint. ``guards`` maps each
    registered guard name to its gate result (an explicit, possibly empty mapping). Guards use the
    primary's strata unless ``guard_strata`` registers another set, for example a pooled primary
    with per-stratum non-inferiority guards on the same endpoint; that is still an
    intersection-union claim. With a registered ``guard_strata`` different from the primary's,
    every guard must cover the same paired observations (equal summed road, scene and replicate
    counts), and each guard stratum's endpoints and worse-bound flag are recorded. When the guard
    strata equal the primary's, every stratum's ``(n_roads, n_scenes, n_replicates)`` must equal
    the primary's (R4 C4); a guard computed on a subset raises ``ValueError`` naming the stratum.
    ``gating_strata`` (a registered nonempty subset of the guard strata) makes only those strata
    required to be GO; the others are reported descriptively. A NO-GO regression flag in any guard
    stratum still blocks the claim, which only makes it more conservative. For a
    ``SequentialGate`` result, ``look`` (1-based) and ``terminal`` must be supplied and are
    recorded.
    """
    _check_delta_min(delta_min)
    required = _strata_tuple(required_strata, "required strata")
    expected_guard_strata = (
        required if guard_strata is None else _strata_tuple(guard_strata, "guard strata")
    )
    gating = (
        expected_guard_strata
        if gating_strata is None
        else _strata_tuple(gating_strata, "gating strata")
    )
    if not set(gating) <= set(expected_guard_strata):
        raise ValueError("gating strata must be a subset of the guard strata")
    if not isinstance(guards, Mapping):
        raise TypeError("guards must be an explicit mapping of guard name to gate result")
    overall, strata, alpha, _, method, metric, primary_counts = _normalize(primary)
    refined = set(expected_guard_strata) != set(required)
    if set(strata) != set(required):
        raise ValueError("primary strata differ from the registered strata")
    if method == SEQUENTIAL_METHOD:
        if isinstance(look, bool) or not isinstance(look, int) or look < 1:
            raise ValueError("a sequential result needs its 1-based look")
        if not isinstance(terminal, bool):
            raise ValueError("a sequential result needs terminal=True/False")
    elif look is not None or terminal is not None:
        raise ValueError("look/terminal apply only to sequential results")

    guard_report, levels = {}, [alpha]
    for name in sorted(guards):
        if not isinstance(name, str) or not name:
            raise ValueError("guard names must be nonempty strings")
        if guards[name] is primary:
            raise ValueError("the primary result cannot be its own guard")
        g_overall, g_strata, g_alpha, g_margin, g_method, g_metric, g_counts = _normalize(
            guards[name]
        )
        if set(g_strata) != set(expected_guard_strata):
            raise ValueError(f"guard {name!r} strata differ from the registered guard strata")
        if refined and _total(g_counts) != _total(primary_counts):
            raise ValueError(
                f"guard {name!r} and the primary do not cover the same paired observations"
            )
        if not refined:
            _check_same_strata_coverage(name, g_counts, primary_counts)
        guard_report[name] = {
            "state": g_overall,
            "gating_go": all(g_strata[s][2] == "GO" for s in gating),
            "alpha": g_alpha,
            "margin": g_margin,
            "method": g_method,
            "metric": g_metric,
            "strata": {
                s: {
                    "state": g_state,
                    "lower": g_lower,
                    "upper": g_upper,
                    "worse_bound_unadjusted": g_lower is not None and g_lower > 0,
                }
                for s, (g_lower, g_upper, g_state) in g_strata.items()
            },
        }
        levels.append(g_alpha)

    per = {}
    for name, (lower, upper, state) in strata.items():
        per[name] = {
            "lower": lower,
            "upper": upper,
            "gate_state": state,
            "outcome": stratum_outcome(lower, upper, delta_min=delta_min),
            "worse_bound_unadjusted": lower is not None and lower > 0,
        }
    outcomes = [p["outcome"] for p in per.values()]
    improvement = all(o in (AT_LEAST_MIN_NOT_EXCLUDED, REAL_BELOW_MIN) for o in outcomes)
    ruled_out = all(o in (RULED_OUT, REAL_BELOW_MIN) for o in outcomes)
    guards_go = all(g["gating_go"] for g in guard_report.values())
    guard_regression = any(g["state"] == "NO-GO" for g in guard_report.values())
    improvement_level = max(levels)

    if overall == "NO-GO":
        decision = REGRESSION_FLAGGED
    elif guard_regression:
        decision = GUARD_REGRESSION_FLAGGED
    elif improvement and guards_go:
        decision = IMPROVED_BELOW_MIN if ruled_out else SUPPORTED
    elif improvement:
        decision = GUARD_NOT_CERTIFIED
    elif ruled_out:
        decision = RULED_OUT
    else:
        decision = INSUFFICIENT
    return {
        "rule_id": RULE_ID,
        "outcome": decision,
        "claims": {
            # R4 C1: a regression flag (primary or any guard stratum) blocks the claim itself.
            "improvement": {
                "holds": improvement
                and guards_go
                and not guard_regression
                and overall != "NO-GO",
                "level": improvement_level,
            },
            "ruled_out": {"holds": ruled_out, "level": alpha},
        },
        "joint_level_when_both_hold": min(1.0, improvement_level + alpha),
        "delta_min": delta_min,
        "required_strata": sorted(required),
        "guard_strata": sorted(expected_guard_strata),
        "gating_strata": sorted(gating),
        "gate_method": method,
        "metric": metric,
        "look": look,
        "terminal": terminal,
        "strata": per,
        "guards": guard_report,
        "any_stratum_worse_bound_unadjusted": any(
            p["worse_bound_unadjusted"] for p in per.values()
        ),
        "any_guard_stratum_worse_bound_unadjusted": any(
            g_s["worse_bound_unadjusted"]
            for g in guard_report.values()
            for g_s in g["strata"].values()
        ),
        "per_stratum_outcomes_unadjusted": True,
        "difference_convention": "positive means candidate worse",
    }


def aggregate_runs(decisions, *, registered_run_ids, rule, family_alpha, k=None):
    """Combine per-checkpoint decisions from the registered independent training runs.

    ``decisions`` maps run id to ``decide`` output and must cover exactly ``registered_run_ids``.

    - ``rule="all"``: the claim holds for every run, each tested at <= family_alpha (an
      intersection-union statement about all named checkpoints).
    - ``rule="at_least_k"``: the claim holds for at least ``k`` of m runs, each tested at
      <= family_alpha / m (Bonferroni); Holm would need p-values the gate does not return. Use
      ``"all"`` rather than k = m, which is valid but needlessly conservative. An unadjusted "any run
      passes" rule is deliberately not offered.

    Improvement and ruled-out are counted from each run's separate claims. If enough runs carry
    both, the outcome is IMPROVEMENT_SUPPORTED_BELOW_MIN (a joint statement at the summed level); if
    different runs reach both counts, CONFLICTING_RUNS. Run identity comes from the registered ids;
    reusing one decision object for two runs is rejected. The result concerns named checkpoints,
    not a training method.

    Regression precedence (R4 C10): if any registered run's outcome is REGRESSION_FLAGGED the
    aggregate outcome is REGRESSION_FLAGGED; otherwise, if any run is GUARD_REGRESSION_FLAGGED,
    it is GUARD_REGRESSION_FLAGGED. Only then are the claim counts read, so a regressed run can
    never aggregate to MEANINGFUL_IMPROVEMENT_RULED_OUT or an improvement outcome, under either run
    rule. The claim counts are still reported.

    ``joint_level`` is the level of the joint "improved, by less than delta_min" statement: the
    family level of the improvement claim plus that of the ruled-out claim, capped at 1. A claim's
    family level is the largest run level under ``"all"`` and the summed run levels (the union
    bound, <= family_alpha) under ``"at_least_k"``. It is reported only when the outcome is
    IMPROVEMENT_SUPPORTED_BELOW_MIN, and is None otherwise.
    """
    _check_alpha(family_alpha, "family_alpha")
    if rule not in RUN_RULES:
        raise ValueError(f"rule must be one of {RUN_RULES}")
    ids = list(registered_run_ids)
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("registered_run_ids must be unique nonempty strings")
    if not isinstance(decisions, Mapping) or set(decisions) != set(ids):
        raise ValueError("decisions must cover exactly the registered runs")
    m = len(ids)
    if rule == "all":
        if k is not None:
            raise ValueError("k applies only to at_least_k")
        need, cap = m, family_alpha
    else:
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= m:
            raise ValueError("at_least_k needs an integer k in [1, m]")
        need, cap = k, family_alpha / m
    seen, signature = set(), None
    for run in ids:
        d = decisions[run]
        if not isinstance(d, Mapping) or d.get("rule_id") != RULE_ID:
            raise ValueError("every decision must come from this rule_id")
        if id(d) in seen:
            raise ValueError("the same decision object is used for two runs")
        seen.add(id(d))
        if d.get("outcome") not in OUTCOMES:
            raise ValueError(f"unknown outcome for run {run!r}")
        claims = d.get("claims")
        if not isinstance(claims, Mapping):
            raise TypeError(f"run {run!r} lacks its claims")
        for name in ("improvement", "ruled_out"):
            claim = claims.get(name)
            if not isinstance(claim, Mapping) or not isinstance(claim.get("holds"), bool):
                raise TypeError(f"run {run!r} lacks its {name} claim")
            _check_alpha(claim.get("level"), "run level")
            if claim["level"] > cap + 1e-12:
                what = "family_alpha" if rule == "all" else "family_alpha / m (Bonferroni)"
                raise ValueError(f"run {run!r} was tested above {what}")
        guards = d.get("guards", {})
        current = (
            d.get("delta_min"),
            d.get("gate_method"),
            json.dumps(d.get("metric"), sort_keys=True),
            tuple(d.get("required_strata", ())),
            tuple(d.get("guard_strata", ())),
            tuple(d.get("gating_strata", ())),
            tuple(
                (
                    g,
                    guards[g].get("margin"),
                    guards[g].get("alpha"),
                    guards[g].get("method"),
                    json.dumps(guards[g].get("metric"), sort_keys=True),
                )
                for g in sorted(guards)
            ),
        )
        if signature is not None and current != signature:
            raise ValueError(
                "runs differ in delta_min, gate method, metric, strata or guard definitions"
            )
        signature = current
    improved = {r for r in ids if decisions[r]["claims"]["improvement"]["holds"]}
    ruled = {r for r in ids if decisions[r]["claims"]["ruled_out"]["holds"]}
    n_improved, n_ruled_out = len(improved), len(ruled)
    run_outcomes = [decisions[r]["outcome"] for r in ids]
    any_regression = any(o in (REGRESSION_FLAGGED, GUARD_REGRESSION_FLAGGED) for o in run_outcomes)
    joint_level = None
    if REGRESSION_FLAGGED in run_outcomes:
        outcome = REGRESSION_FLAGGED
    elif GUARD_REGRESSION_FLAGGED in run_outcomes:
        outcome = GUARD_REGRESSION_FLAGGED
    elif len(improved & ruled) >= need:
        outcome = IMPROVED_BELOW_MIN  # joint statement: level is the sum of both claim levels
    elif n_improved >= need and n_ruled_out >= need:
        outcome = CONFLICTING
    elif n_improved >= need:
        outcome = SUPPORTED
    elif n_ruled_out >= need:
        outcome = RULED_OUT
    else:
        outcome = INSUFFICIENT
    if outcome == IMPROVED_BELOW_MIN:
        combine = max if rule == "all" else sum
        family_levels = [
            combine(float(decisions[r]["claims"][name]["level"]) for r in ids)
            for name in ("improvement", "ruled_out")
        ]
        joint_level = min(1.0, sum(family_levels))
    return {
        "rule_id": RULE_ID,
        "run_rule": rule,
        "k": need,
        "m": m,
        "family_alpha": family_alpha,
        "joint_level": joint_level,
        "outcome": outcome,
        "n_improvement_claims": n_improved,
        "n_ruled_out_claims": n_ruled_out,
        "any_run_regression_flagged": any_regression,
        "conflicting_runs": bool(improved - ruled) and bool(ruled - improved),
        "run_outcomes": {r: decisions[r]["outcome"] for r in ids},
        "scope": "named checkpoints; not a training-method interval",
    }
