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
