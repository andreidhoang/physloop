"""Known answers for the candidate superiority rule (agent-authored, D-023, S2.6/S4.6).

The rule only reads endpoints the locked gate already returns. Expected values below come from the
closed form of the locked bound, not from the module under test:
- every road difference zero over R roads: upper = 1 - (alpha/2)**(1/R), lower = -upper;
- every road difference -1 over R roads: upper = (1 - t) - t with t = (alpha/2)**(1/R).
Margins and delta_min here are illustrative test inputs, not Huy's registered values.
"""

import math
from dataclasses import asdict

import numpy as np
import pytest
from scipy.stats import binom, binomtest

from gate import superiority as sup
from gate import superiority_power as spw
from gate.api import evaluate
from gate.contracts import ComparisonSpec, MetricSpec, Protocol, RolloutRecord
from gate.sequential import SequentialGate
from gate.synthetic import generate
from gate.verdict import paired_road_bounds, verdict

ALPHA = 0.05
MARGIN = 0.05  # verdict() needs a positive margin; superiority endpoints do not depend on it


def rows(stratum, n_roads, better=0, worse=0):
    """One scene per road. `better` roads: A fails, B passes; `worse`: B fails, A passes."""
    pattern = [(True, False)] * better + [(False, True)] * worse
    pattern += [(False, False)] * (n_roads - len(pattern))
    return [
        {
            "scene_id": f"{stratum}-s{i}",
            "road_id": f"{stratum}-r{i}",
            "stratum": stratum,
            "a_fail": a,
            "b_fail": b,
        }
        for i, (a, b) in enumerate(pattern)
    ]


def zero_upper(R, alpha=ALPHA):
    return 1 - (alpha / 2) ** (1 / R)


def decide(result, strata=("a",), guards=None, delta_min=0.02, **kw):
    return sup.decide(
        result,
        required_strata=strata,
        guards={} if guards is None else guards,
        delta_min=delta_min,
        **kw,
    )


# --- per-stratum classification ---------------------------------------------------------------


@pytest.mark.parametrize(
    "lower, upper, outcome",
    [
        (-0.05, -0.01, sup.AT_LEAST_MIN_NOT_EXCLUDED),
        (-0.015, -0.001, sup.REAL_BELOW_MIN),
        (-0.01, 0.03, sup.RULED_OUT),
        (-0.05, 0.03, sup.INSUFFICIENT),
        (-0.05, 0.0, sup.INSUFFICIENT),  # upper exactly zero is not improvement
        (-0.02, 0.03, sup.INSUFFICIENT),  # lower exactly -delta_min does not rule out
        (None, None, sup.MISSING),
    ],
)
def test_stratum_outcome_boundaries(lower, upper, outcome):
    assert sup.stratum_outcome(lower, upper, delta_min=0.02) == outcome


@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.1, -0.1),
        (math.nan, 0.0),
        (-1.5, 0.0),
        (0.0, 1.5),
        (None, 0.1),
        (-0.1, None),
        (False, 0.1),
        (np.bool_(False), 0.1),
        ("-0.1", 0.1),
    ],
)
def test_stratum_outcome_rejects_invalid_endpoints(lower, upper):
    with pytest.raises(ValueError):
        sup.stratum_outcome(lower, upper, delta_min=0.02)


@pytest.mark.parametrize("delta_min", [0.0, -0.01, 1.01, math.nan, math.inf, True, "0.02"])
def test_delta_min_must_be_in_open_closed_unit_interval(delta_min):
    with pytest.raises(ValueError):
        sup.stratum_outcome(-0.1, -0.05, delta_min=delta_min)


def test_numpy_reals_are_accepted():
    assert sup.stratum_outcome(
        np.float32(-0.05), np.float64(-0.01), delta_min=np.float32(0.02)
    ) == (sup.AT_LEAST_MIN_NOT_EXCLUDED)


# --- decisions on real locked-gate results -----------------------------------------------------


def test_all_roads_improved_is_supported_with_closed_form_upper():
    result = verdict(rows("a", 50, better=50), MARGIN, ALPHA)
    t = (ALPHA / 2) ** (1 / 50)
    assert result.details["a"].upper == pytest.approx((1 - t) - t, abs=1e-9)
    d = decide(result)
    assert d["outcome"] == sup.SUPPORTED
    assert d["claims"]["improvement"] == {"holds": True, "level": ALPHA}
    assert d["claims"]["ruled_out"]["holds"] is False
    assert d["strata"]["a"]["outcome"] == sup.AT_LEAST_MIN_NOT_EXCLUDED
    assert d["rule_id"] == sup.RULE_ID


def test_identical_arms_rule_out_large_minimum_but_not_small_one():
    result = verdict(rows("a", 441), MARGIN, ALPHA)
    assert result.details["a"].upper == pytest.approx(zero_upper(441), abs=1e-9)
    assert result.details["a"].lower == pytest.approx(-zero_upper(441), abs=1e-9)
    assert zero_upper(441) == pytest.approx(0.0083298, abs=1e-6)
    assert decide(result)["outcome"] == sup.RULED_OUT
    assert decide(result, delta_min=0.005)["outcome"] == sup.INSUFFICIENT


def test_falsifier_needs_229_zero_roads_at_minimum_0_016():
    # 1 - 0.025**(1/R) < 0.016  <=>  R > ln(0.025)/ln(0.984) = 228.7
    assert decide(verdict(rows("a", 228), MARGIN, ALPHA), delta_min=0.016)["outcome"] == (
        sup.INSUFFICIENT
    )
    assert decide(verdict(rows("a", 229), MARGIN, ALPHA), delta_min=0.016)["outcome"] == (
        sup.RULED_OUT
    )


def test_mixed_strata_are_insufficient_overall_and_reported_per_stratum():
    data = rows("a", 50, better=50) + rows("b", 441)
    d = decide(verdict(data, MARGIN, ALPHA), strata=("a", "b"))
    assert d["outcome"] == sup.INSUFFICIENT
    assert d["strata"]["a"]["outcome"] == sup.AT_LEAST_MIN_NOT_EXCLUDED
    assert d["strata"]["b"]["outcome"] == sup.RULED_OUT
    assert d["per_stratum_outcomes_unadjusted"] is True


def test_below_min_on_every_stratum_is_a_joint_claim_at_summed_level():
    # 441 roads, 10 improved: upper < 0 (exact threshold for b = 0) and lower > -0.05
    result = verdict(rows("a", 441, better=10), MARGIN, ALPHA)
    d = decide(result, delta_min=0.05)
    assert d["outcome"] == sup.IMPROVED_BELOW_MIN
    assert d["claims"]["improvement"]["holds"] and d["claims"]["ruled_out"]["holds"]
    assert d["joint_level_when_both_hold"] == pytest.approx(2 * ALPHA)
    assert decide(verdict(rows("a", 441, better=9), MARGIN, ALPHA), delta_min=0.05)["outcome"] == (
        sup.RULED_OUT
    )


def test_one_stratum_below_min_and_one_large_is_supported_not_below_min():
    data = rows("a", 441, better=10) + rows("b", 50, better=50)
    d = decide(verdict(data, MARGIN, ALPHA), strata=("a", "b"), delta_min=0.05)
    assert d["outcome"] == sup.SUPPORTED


def test_ruled_out_includes_strata_improved_below_min():
    data = rows("a", 441, better=10) + rows("b", 441)
    d = decide(verdict(data, MARGIN, ALPHA), strata=("a", "b"), delta_min=0.05)
    assert d["strata"]["a"]["outcome"] == sup.REAL_BELOW_MIN
    assert d["strata"]["b"]["outcome"] == sup.RULED_OUT
    assert d["outcome"] == sup.RULED_OUT


def test_guard_must_be_go_and_guard_regression_takes_precedence():
    primary = verdict(rows("a", 50, better=50), MARGIN, ALPHA)
    good = verdict(rows("a", 441), MARGIN, ALPHA)
    weak = verdict(rows("a", 50), MARGIN, ALPHA)
    bad = verdict(rows("a", 200, worse=60), MARGIN, ALPHA)
    assert good.overall == "GO" and weak.overall == "INSUFFICIENT" and bad.overall == "NO-GO"
    assert decide(primary, guards={"progress": good})["outcome"] == sup.SUPPORTED
    d = decide(primary, guards={"progress": good, "offroad": weak})
    assert d["outcome"] == sup.GUARD_NOT_CERTIFIED
    assert d["guards"]["offroad"]["state"] == "INSUFFICIENT"
    assert d["guards"]["progress"]["margin"] == MARGIN
    assert decide(primary, guards={"offroad": bad})["outcome"] == sup.GUARD_REGRESSION_FLAGGED
    # a guard regression is visible even when the primary is only ruled out
    ruled = verdict(rows("a", 441), MARGIN, ALPHA)
    other_bad = verdict(rows("a", 441, worse=120), MARGIN, ALPHA)
    assert decide(ruled, guards={"offroad": other_bad})["outcome"] == sup.GUARD_REGRESSION_FLAGGED


def test_guard_strata_must_match_and_primary_cannot_guard_itself():
    primary = verdict(rows("a", 50, better=50), MARGIN, ALPHA)
    with pytest.raises(ValueError, match="guard"):
        decide(primary, guards={"g": verdict(rows("zzz", 441), MARGIN, ALPHA)})
    with pytest.raises(ValueError, match="own guard"):
        decide(primary, guards={"g": primary})
    with pytest.raises(ValueError, match="method"):
        decide(primary, guards={"g": {**asdict(primary), "method": "foreign"}})
    with pytest.raises(TypeError):
        decide(primary, guards=[primary])


def test_level_is_the_largest_component_alpha():
    primary = verdict(rows("a", 50, better=50), MARGIN, 0.025)
    guard = verdict(rows("a", 441), MARGIN, ALPHA)
    d = decide(primary, guards={"g": guard})
    assert d["claims"]["improvement"]["level"] == ALPHA
    assert d["claims"]["ruled_out"]["level"] == 0.025


def test_regression_flag_takes_precedence_in_one_of_two_strata():
    data = rows("a", 50, better=50) + rows("b", 200, worse=60)
    result = verdict(data, MARGIN, ALPHA)
    assert result.strata["b"] == "NO-GO"
    assert decide(result, strata=("a", "b"))["outcome"] == sup.REGRESSION_FLAGGED


def test_worse_bound_is_flagged_when_the_regression_test_is_silent():
    # Reviewer's case: KL lower bound > 0 on both strata, Hoeffding regression p-values large.
    data = generate(5, {"a": 300, "b": 200}, p_a=0.1, q_plus=0.06, q_minus=0.0)
    result = verdict(data, MARGIN, ALPHA)
    assert result.overall != "NO-GO"
    assert all(result.details[s].lower > 0 for s in ("a", "b"))
    d = decide(result, strata=("a", "b"))
    assert d["outcome"] == sup.RULED_OUT
    assert d["any_stratum_worse_bound_unadjusted"] is True
    assert all(d["strata"][s]["worse_bound_unadjusted"] for s in ("a", "b"))


def test_registered_strata_must_match_the_result():
    result = verdict(rows("a", 50, better=50), MARGIN, ALPHA)  # stratum b silently absent
    with pytest.raises(ValueError, match="registered"):
        decide(result, strata=("a", "b"))
    with pytest.raises(TypeError):
        decide(result, strata="a")


def test_missing_required_stratum_is_insufficient():
    result = verdict(rows("a", 50, better=50), MARGIN, ALPHA, required_strata=("a", "b"))
    d = decide(result, strata=("a", "b"))
    assert d["strata"]["b"]["outcome"] == sup.MISSING
    assert d["outcome"] == sup.INSUFFICIENT


def test_continuous_higher_is_better_direction():
    data = [
        {"scene_id": f"s{i}", "road_id": f"r{i}", "stratum": "a", "a_value": 0.4, "b_value": 0.9}
        for i in range(50)
    ]
    assert decide(verdict(data, MARGIN, ALPHA, higher_is_worse=False))["outcome"] == sup.SUPPORTED
    assert decide(verdict(data, MARGIN, ALPHA, higher_is_worse=True))["outcome"] != sup.SUPPORTED


def test_dataclass_and_dict_shapes_give_the_same_decision():
    result = verdict(rows("a", 50, better=50) + rows("b", 441), MARGIN, ALPHA)
    expected = decide(result, strata=("a", "b"))
    assert decide(asdict(result), strata=("a", "b")) == expected


def test_api_evaluate_response_is_accepted_and_metric_recorded():
    spec = ComparisonSpec(
        "teacher",
        "candidate",
        Protocol("test-v1", "unseeded", "config", "eval", "test", "max-lateral-m"),
        ("urban",),
        0.05,
        0.05,
        MetricSpec("test", "binary", True, "fraction"),
    )
    records = [
        RolloutRecord(arm, f"scene-{i}", f"road-{i}", "urban", 0, value)
        for i in range(220)
        for arm, value in (("A", 1), ("B", 0))
    ]
    response = evaluate(records, spec)
    d = decide(response, strata=("urban",))
    assert d["outcome"] == sup.SUPPORTED
    assert d["metric"]["higher_is_worse"] is True
    assert d == {**decide(response["verdict"], strata=("urban",)), "metric": d["metric"]}


def test_ambiguous_or_inconsistent_shapes_are_rejected():
    good = asdict(verdict(rows("a", 50, better=50), MARGIN, ALPHA))
    bad_cases = [
        ({**good, "method": "something_else"}, "method"),
        ({**good, "strata": {"z": "GO"}}, "strata"),
        ({**good, "alpha": 1.5}, "alpha"),
        ({**good, "strata": {"a": "BANANA"}}, "state"),
        ({**good, "overall": "INSUFFICIENT", "strata": {"a": "NO-GO"}}, "NO-GO"),
        ({**good, "overall": "GO", "strata": {"a": "INSUFFICIENT"}}, "GO"),
        ({**good, "verdict": good}, "ambiguous"),
        ({"schema_version": 2, "verdict": good}, "ambiguous"),
    ]
    for case, match in bad_cases:
        with pytest.raises(ValueError, match=match):
            decide(case)
    missing = verdict(rows("a", 50, better=50), MARGIN, ALPHA, required_strata=("a", "b"))
    broken = asdict(missing)
    broken["details"]["b"] = {**broken["details"]["b"], "lower": -0.1, "upper": -0.05}
    with pytest.raises(ValueError, match="road count"):
        decide(broken, strata=("a", "b"))


# --- sequential results -------------------------------------------------------------------


def test_sequential_result_needs_look_and_uses_spent_alpha_endpoints():
    gate = SequentialGate(("a",), MARGIN, ALPHA, weights=(0.5, 0.5))
    first = gate.update(rows("a", 30, better=30))
    fixed = verdict(rows("a", 30, better=30), MARGIN, ALPHA * 0.5)
    assert first.details["a"].upper == fixed.details["a"].upper
    with pytest.raises(ValueError, match="look"):
        decide(first)
    d = decide(first, look=1, terminal=gate.terminal)
    assert d["outcome"] == sup.SUPPORTED
    assert d["claims"]["improvement"]["level"] == ALPHA  # registered total, spent across looks
    assert (d["look"], d["terminal"]) == (1, True)
    with pytest.raises(ValueError, match="sequential"):
        decide(fixed, look=1, terminal=True)


def test_sequential_closed_stratum_keeps_its_earlier_endpoints():
    gate = SequentialGate(("a", "b"), MARGIN, ALPHA, weights=(0.5, 0.5))
    first = gate.update(rows("a", 441) + rows("b", 20, better=5))
    assert first.strata["a"] == "GO" and first.strata["b"] != "GO"
    second = gate.update(
        [
            {**r, "scene_id": "x" + r["scene_id"], "road_id": "x" + r["road_id"]}
            for r in rows("b", 200, better=200)
        ]
    )
    assert second.details["a"] == first.details["a"]  # closed at look 1 at spent alpha
    d = decide(second, strata=("a", "b"), look=2, terminal=gate.terminal)
    assert d["strata"]["a"]["outcome"] in (sup.RULED_OUT, sup.INSUFFICIENT)
    assert d["outcome"] != sup.SUPPORTED  # superiority on a cannot be reached after closure


# --- aggregation across independent training runs ------------------------------------------


def _decision(improved, ruled_out=False, level=0.05, outcome=None, delta_min=0.02):
    if outcome is None:
        outcome = (
            sup.IMPROVED_BELOW_MIN
            if improved and ruled_out
            else sup.SUPPORTED
            if improved
            else sup.RULED_OUT
            if ruled_out
            else sup.INSUFFICIENT
        )
    return {
        "rule_id": sup.RULE_ID,
        "outcome": outcome,
        "claims": {
            "improvement": {"holds": improved, "level": level},
            "ruled_out": {"holds": ruled_out, "level": level},
        },
        "delta_min": delta_min,
        "gate_method": sup.FIXED_METHOD,
        "required_strata": ["a"],
        "guards": {},
    }


IDS = ("run1", "run2", "run3")


def agg(decisions, **kw):
    return sup.aggregate_runs(dict(zip(IDS, decisions)), registered_run_ids=IDS, **kw)


def test_all_runs_rule_is_intersection_union_at_family_alpha():
    assert agg([_decision(True) for _ in IDS], rule="all", family_alpha=0.05)["outcome"] == (
        sup.SUPPORTED
    )
    out = agg([_decision(True), _decision(False), _decision(True)], rule="all", family_alpha=0.05)
    assert out["outcome"] == sup.INSUFFICIENT and out["n_improvement_claims"] == 2


def test_all_runs_ruled_out_and_all_below_min():
    ruled = [_decision(False, True) for _ in IDS]
    assert agg(ruled, rule="all", family_alpha=0.05)["outcome"] == sup.RULED_OUT
    below = [_decision(True, True) for _ in IDS]
    assert agg(below, rule="all", family_alpha=0.05)["outcome"] == sup.IMPROVED_BELOW_MIN


def test_conflicting_runs_are_reported_not_collapsed():
    ids = ("r1", "r2", "r3", "r4")
    decisions = [
        _decision(True, level=0.0125),
        _decision(True, level=0.0125),
        _decision(False, True, level=0.0125),
        _decision(False, True, level=0.0125),
    ]
    out = sup.aggregate_runs(
        dict(zip(ids, decisions)), registered_run_ids=ids, rule="at_least_k", k=2, family_alpha=0.05
    )
    assert out["outcome"] == sup.CONFLICTING and out["conflicting_runs"] is True


def test_regression_in_one_run_is_flagged():
    runs = [_decision(True, level=0.05 / 3), _decision(True, level=0.05 / 3)]
    runs.append(_decision(False, level=0.05 / 3, outcome=sup.REGRESSION_FLAGGED))
    out = agg(runs, rule="at_least_k", k=2, family_alpha=0.05)
    assert out["outcome"] == sup.SUPPORTED and out["any_run_regression_flagged"] is True


def test_at_least_k_requires_bonferroni_levels():
    runs = [_decision(True), _decision(False), _decision(False)]
    with pytest.raises(ValueError, match="Bonferroni"):
        agg(runs, rule="at_least_k", k=1, family_alpha=0.05)
    runs = [_decision(True, level=0.05 / 3), _decision(False, level=0.05 / 3)]
    runs.append(_decision(False, level=0.05 / 3))
    assert agg(runs, rule="at_least_k", k=1, family_alpha=0.05)["outcome"] == sup.SUPPORTED
    assert agg(runs, rule="at_least_k", k=2, family_alpha=0.05)["outcome"] == sup.INSUFFICIENT


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rule": "any", "family_alpha": 0.05},
        {"rule": "at_least_k", "family_alpha": 0.05},
        {"rule": "at_least_k", "k": 4, "family_alpha": 0.05},
        {"rule": "at_least_k", "k": 0, "family_alpha": 0.05},
        {"rule": "all", "k": 3, "family_alpha": 0.05},
        {"rule": "all", "family_alpha": 0.01},  # run level 0.05 exceeds the family alpha
    ],
)
def test_aggregation_rejects_invalid_rules(kwargs):
    with pytest.raises(ValueError):
        agg([_decision(True), _decision(True), _decision(True)], **kwargs)


def test_aggregation_is_bound_to_the_registered_runs():
    d = [_decision(True), _decision(True), _decision(False)]
    with pytest.raises(ValueError, match="registered"):  # a failing run dropped
        sup.aggregate_runs(
            {"run1": d[0], "run2": d[1]}, registered_run_ids=IDS, rule="all", family_alpha=0.05
        )
    same = _decision(True)
    with pytest.raises(ValueError, match="same decision"):
        agg([same, same, same], rule="all", family_alpha=0.05)
    with pytest.raises(ValueError, match="unique"):
        sup.aggregate_runs(
            {"x": d[0]}, registered_run_ids=("x", "x"), rule="all", family_alpha=0.05
        )


def test_aggregation_rejects_foreign_unknown_or_incomparable_decisions():
    with pytest.raises(ValueError, match="rule_id"):
        agg(
            [{**_decision(True), "rule_id": "other"}, _decision(True), _decision(True)],
            rule="all",
            family_alpha=0.05,
        )
    with pytest.raises(ValueError, match="unknown outcome"):
        agg(
            [_decision(True, outcome="WHATEVER"), _decision(True), _decision(True)],
            rule="all",
            family_alpha=0.05,
        )
    with pytest.raises(ValueError, match="differ"):
        agg(
            [_decision(True, delta_min=0.01), _decision(True), _decision(True)],
            rule="all",
            family_alpha=0.05,
        )


def test_real_decisions_aggregate():
    better = verdict(rows("a", 50, better=50), MARGIN, ALPHA)
    flat = verdict(rows("a", 441), MARGIN, ALPHA)
    decisions = {"run1": decide(better), "run2": decide(better), "run3": decide(flat)}
    out = sup.aggregate_runs(decisions, registered_run_ids=IDS, rule="all", family_alpha=ALPHA)
    assert out["outcome"] == sup.INSUFFICIENT
    assert out["run_outcomes"]["run3"] == sup.RULED_OUT


# --- power through the real gate ----------------------------------------------------------


def _exact_binary_power(R, p_better, p_worse, alpha=ALPHA):
    """Independent oracle: the bound depends only on (improved, worsened) road counts."""
    total = 0.0
    for nb in range(R + 1):
        w = binom.pmf(nb, R, p_worse)
        if w < 1e-12:
            continue
        for na in range(R - nb + 1):
            d = [-1.0] * na + [1.0] * nb + [0.0] * (R - na - nb)
            if paired_road_bounds(d, alpha)[1] < 0:
                total += w * binom.sf(na - 1, R - nb, p_better / (1 - p_worse))
                break
    return total


def scenario(**overrides):
    base = {
        "strata": {"a": 120},
        "p_a": 0.10,
        "q_plus": 0.0,
        "q_minus": 0.05,
        "margin": MARGIN,
        "alpha": ALPHA,
        "delta_min": 0.02,
        "trials": 400,
        "seed": 20261108,
    }
    base.update(overrides)
    return base


def supported_rate(report):
    o = report["outcomes"]
    return (o[sup.SUPPORTED]["count"] + o[sup.IMPROVED_BELOW_MIN]["count"]) / report["trials"]


@pytest.mark.parametrize("q_plus, q_minus", [(0.0, 0.05), (0.02, 0.08)])
def test_power_matches_exact_binary_oracle_within_monte_carlo_interval(q_plus, q_minus):
    report = spw.superiority_power(**scenario(q_plus=q_plus, q_minus=q_minus))
    exact = _exact_binary_power(120, p_better=q_minus, p_worse=q_plus)
    count = round(supported_rate(report) * report["trials"])
    low, high = spw._clopper_pearson(count, report["trials"])
    assert low <= exact <= high


def test_clopper_pearson_matches_scipy_exact():
    for n in (1, 7, 100):
        for c in range(n + 1):
            ref = binomtest(c, n).proportion_ci(method="exact")
            got = spw._clopper_pearson(c, n)
            assert got == pytest.approx((ref.low, ref.high), abs=1e-12)


def test_power_is_deterministic_for_a_seed_and_counts_sum_to_trials():
    a = spw.superiority_power(**scenario(trials=100))
    assert a == spw.superiority_power(**scenario(trials=100))
    assert sum(v["count"] for v in a["outcomes"].values()) == 100
    assert a["seeding"] == "SeedSequence([seed, trial])" and a["analysis"] == "fixed_sample"


def test_seeds_do_not_alias_across_seed_and_trial():
    s = [int(np.random.SeedSequence([0, 1_000_003]).generate_state(1)[0])]
    s.append(int(np.random.SeedSequence([1, 0]).generate_state(1)[0]))
    assert s[0] != s[1]


def test_certain_improvement_is_always_supported():
    report = spw.superiority_power(**scenario(strata={"a": 10}, p_a=1.0, q_minus=1.0, trials=20))
    assert supported_rate(report) == 1.0


def test_null_effect_rarely_supported():
    report = spw.superiority_power(**scenario(q_plus=0.05, q_minus=0.05, trials=400))
    assert supported_rate(report) < 0.02


def test_effects_model_mixed_truth_strata():
    report = spw.superiority_power(
        **scenario(strata={"a": 120, "b": 120}, effects={"b": (0.05, 0.05)}, trials=100)
    )
    assert supported_rate(report) < 0.05  # one null stratum blocks the intersection-union claim
    assert report["scenario"]["effects"] == {"b": [0.05, 0.05]}


def test_power_report_labels_assumptions_and_rejects_bad_trials():
    report = spw.superiority_power(**scenario(trials=10))
    assert report["classification"] == "assumption_not_measurement"
    with pytest.raises(ValueError):
        spw.superiority_power(**scenario(trials=0))
