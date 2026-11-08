"""Superiority and falsifier power through the real gate. Agent-authored under D-018/D-023 (S2.6).

Each trial draws paired binary rows with ``gate.synthetic.generate``, evaluates them with the
protected ``gate.verdict.verdict`` and classifies the result with ``gate.superiority.decide``. No
bound or averaging is re-implemented. Every scenario value is an assumption, not a measurement;
measured S2 road counts, discordance and dependence replace them.

Generator conventions (``gate.synthetic``): ``p_a`` is the reference failure rate, ``q_plus`` the
probability that only the candidate fails (candidate worse) and ``q_minus`` that only the reference
fails (candidate better). ``*_correlation`` values are copy probabilities, not calibrated ICCs.
"""

import numpy as np
from scipy.stats import beta

from gate import superiority as sup
from gate.synthetic import generate
from gate.verdict import verdict

OUTCOMES = sup.OUTCOMES
INTERVAL = "clopper_pearson_two_sided_0.95"


def _clopper_pearson(count, n, level=0.95):
    tail = (1 - level) / 2
    low = 0.0 if count == 0 else float(beta.ppf(tail, count, n - count + 1))
    high = 1.0 if count == n else float(beta.ppf(1 - tail, count + 1, n - count))
    return low, high


def superiority_power(
    *,
    strata,
    p_a,
    q_plus,
    q_minus,
    margin,
    alpha,
    delta_min,
    trials,
    seed,
    scenes_per_road=1,
    replicates=1,
    road_correlation=0.0,
    replicate_correlation=0.0,
    effects=None,
):
    """Outcome frequencies of the candidate rule for one primary endpoint with no guards.

    ``margin`` is only passed to ``verdict`` (it must be positive); superiority endpoints and the
    regression flag do not depend on it. Guards are omitted, so the guard outcomes cannot occur.

    This models a fixed-sample analysis. Under ``SequentialGate`` a stratum closes once it is
    non-inferior, so sequential superiority power differs and is not computed here. Trial ``t`` uses
    ``SeedSequence([seed, t])``; reusing one seed across scenarios therefore gives common random
    numbers between them. ``effects`` passes per-stratum (q_plus, q_minus) overrides to the
    generator, for heterogeneous or mixed-truth strata.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    counts = dict.fromkeys(OUTCOMES, 0)
    required = tuple(sorted(strata))
    for trial in range(trials):
        rows = generate(
            int(np.random.SeedSequence([seed, trial]).generate_state(1)[0]),
            dict(strata),
            scenes_per_road=scenes_per_road,
            replicates=replicates,
            p_a=p_a,
            q_plus=q_plus,
            q_minus=q_minus,
            road_correlation=road_correlation,
            replicate_correlation=replicate_correlation,
            effects=effects,
        )
        result = verdict(rows, margin, alpha, required_strata=required)
        decision = sup.decide(result, required_strata=required, guards={}, delta_min=delta_min)
        counts[decision["outcome"]] += 1
    return {
        "classification": "assumption_not_measurement",
        "rule_id": sup.RULE_ID,
        "interval_method": INTERVAL,
        "trials": trials,
        "seed": seed,
        "seeding": "SeedSequence([seed, trial])",
        "analysis": "fixed_sample",
        "scenario": {
            "strata": dict(strata),
            "scenes_per_road": scenes_per_road,
            "replicates": replicates,
            "p_a": p_a,
            "q_plus": q_plus,
            "q_minus": q_minus,
            "road_correlation": road_correlation,
            "replicate_correlation": replicate_correlation,
            "effects": None if effects is None else {k: list(v) for k, v in effects.items()},
            "margin": margin,
            "alpha": alpha,
            "delta_min": delta_min,
            "guards": [],
        },
        "outcomes": {
            name: {"count": c, "rate": c / trials, "interval": _clopper_pearson(c, trials)}
            for name, c in counts.items()
        },
    }


DESIGNS_DOC = {
    "all_strata": "improvement required on every stratum (no guards)",
    "pooled": "improvement on the pooled population (no guards)",
    "pooled_guard": "pooled improvement plus per-stratum non-inferiority of the same endpoint",
}


def design_power(
    *,
    strata,
    p_a,
    q_plus,
    q_minus,
    margin,
    guard_margins,
    alpha,
    delta_min,
    trials,
    seed,
    effects=None,
    scenes_per_road=1,
    replicates=1,
    road_correlation=0.0,
    replicate_correlation=0.0,
    pooled_label="pooled",
    gating_strata=None,
):
    """Compare registered-claim designs on the same simulated rows in every trial.

    Designs (no other guard endpoints are modelled):
    - ``all_strata``: ``decide`` on the stratified result, every stratum must improve;
    - ``pooled``: ``decide`` on the result with every row relabelled to one stratum;
    - ``pooled_guard@m``: the pooled claim plus the stratified result at margin ``m`` as a
      non-inferiority guard on the same endpoint, for each ``m`` in ``guard_margins``;
    - ``pooled_gating@m`` (only with ``gating_strata``): the same, but only the registered gating
      strata must be GO; the rest are descriptive, though a regression flag anywhere still blocks.

    ``margin`` is passed to the stratified and pooled ``verdict`` calls whose endpoints feed the
    superiority decisions; those endpoints and the regression flag do not depend on it. For each
    guard margin, per-stratum GO frequencies and the joint frequency (every stratum GO, which is
    what the guard needs) are returned. Fixed-sample analysis only. Trial ``t`` uses
    ``SeedSequence([seed, t])``: the designs within one call share rows exactly; calls with the
    same strata share most underlying draws; calls with different strata do not share rows.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    margins = tuple(guard_margins)
    labels = [f"{m:g}" for m in margins]
    if len(set(margins)) != len(margins) or len(set(labels)) != len(labels):
        raise ValueError("guard margins must be unique and have distinct labels")
    required = tuple(sorted(strata))
    if pooled_label in required and len(required) > 1:
        raise ValueError("pooled_label must not name one of several strata")
    if gating_strata is not None and not set(gating_strata) <= set(required):
        raise ValueError("gating strata must be among the strata")
    names = ("all_strata", "pooled") + tuple(f"pooled_guard@{label}" for label in labels)
    if gating_strata is not None:
        names += tuple(f"pooled_gating@{label}" for label in labels)
    counts = {name: dict.fromkeys(OUTCOMES, 0) for name in names}
    guard_go = {label: dict.fromkeys(required, 0) for label in labels}
    guard_joint = dict.fromkeys(labels, 0)
    for trial in range(trials):
        rows = generate(
            int(np.random.SeedSequence([seed, trial]).generate_state(1)[0]),
            dict(strata),
            scenes_per_road=scenes_per_road,
            replicates=replicates,
            p_a=p_a,
            q_plus=q_plus,
            q_minus=q_minus,
            road_correlation=road_correlation,
            replicate_correlation=replicate_correlation,
            effects=effects,
        )
        stratified = {
            m: verdict(rows, m, alpha, required_strata=required) for m in {margin, *margins}
        }
        pooled_rows = [{**row, "stratum": pooled_label} for row in rows]
        pooled = verdict(pooled_rows, margin, alpha, required_strata=(pooled_label,))
        decisions = {
            "all_strata": sup.decide(
                stratified[margin], required_strata=required, guards={}, delta_min=delta_min
            ),
            "pooled": sup.decide(
                pooled, required_strata=(pooled_label,), guards={}, delta_min=delta_min
            ),
        }
        for m, label in zip(margins, labels, strict=True):
            decisions[f"pooled_guard@{label}"] = sup.decide(
                pooled,
                required_strata=(pooled_label,),
                guards={"same_endpoint_noninferiority": stratified[m]},
                guard_strata=required,
                delta_min=delta_min,
            )
            if gating_strata is not None:
                decisions[f"pooled_gating@{label}"] = sup.decide(
                    pooled,
                    required_strata=(pooled_label,),
                    guards={"same_endpoint_noninferiority": stratified[m]},
                    guard_strata=required,
                    gating_strata=tuple(gating_strata),
                    delta_min=delta_min,
                )
            for s in required:
                guard_go[label][s] += stratified[m].strata[s] == "GO"
            guard_joint[label] += stratified[m].overall == "GO"
        for name, decision in decisions.items():
            counts[name][decision["outcome"]] += 1

    def summary(c):
        improved = c[sup.SUPPORTED] + c[sup.IMPROVED_BELOW_MIN]
        return {
            "improvement_rate": improved / trials,
            "improvement_interval": _clopper_pearson(improved, trials),
            "outcomes": {k: v for k, v in c.items() if v},
        }

    return {
        "classification": "assumption_not_measurement",
        "rule_id": sup.RULE_ID,
        "interval_method": INTERVAL,
        "analysis": "fixed_sample",
        "seeding": "SeedSequence([seed, trial]); designs share rows",
        "trials": trials,
        "seed": seed,
        "designs": DESIGNS_DOC,
        "scenario": {
            "strata": dict(strata),
            "scenes_per_road": scenes_per_road,
            "replicates": replicates,
            "p_a": p_a,
            "q_plus": q_plus,
            "q_minus": q_minus,
            "road_correlation": road_correlation,
            "replicate_correlation": replicate_correlation,
            "effects": None if effects is None else {k: list(v) for k, v in effects.items()},
            "margin": margin,
            "guard_margins": list(margins),
            "gating_strata": None if gating_strata is None else sorted(gating_strata),
            "alpha": alpha,
            "delta_min": delta_min,
        },
        "results": {name: summary(c) for name, c in counts.items()},
        "guard_go_rate": {
            m: {s: g / trials for s, g in per.items()} for m, per in guard_go.items()
        },
        "guard_joint_go_rate": {m: c / trials for m, c in guard_joint.items()},
    }
