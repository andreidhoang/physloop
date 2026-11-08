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


# --- registered guard strata and design comparison ---------------------------------------


def test_pooled_primary_with_registered_stratified_guard():
    data = rows("a", 220, better=20) + rows("b", 221, better=20)
    pooled = verdict([{**r, "stratum": "pooled"} for r in data], MARGIN, ALPHA)
    guard = verdict(data, 0.10, ALPHA)
    d = sup.decide(
        pooled,
        required_strata=("pooled",),
        guards={"ni": guard},
        guard_strata=("a", "b"),
        delta_min=0.02,
    )
    assert d["outcome"] in (sup.SUPPORTED, sup.IMPROVED_BELOW_MIN)
    assert d["guard_strata"] == ["a", "b"] and d["required_strata"] == ["pooled"]
    with pytest.raises(ValueError, match="registered guard strata"):
        sup.decide(
            pooled, required_strata=("pooled",), guards={"ni": guard}, delta_min=0.02
        )  # default: guards must use the primary's strata
    with pytest.raises(ValueError, match="registered guard strata"):
        sup.decide(
            pooled,
            required_strata=("pooled",),
            guards={"ni": guard},
            guard_strata=("a", "c"),
            delta_min=0.02,
        )
    with pytest.raises(TypeError):
        sup.decide(
            pooled, required_strata=("pooled",), guards={}, guard_strata="ab", delta_min=0.02
        )


def _exact_prob_upper_below(R, p_better, p_worse, t, alpha=ALPHA):
    """Exact P(upper < t) for one binary outcome per road; upper depends only on the counts and
    falls as improved roads rise, so each worsened count has a threshold improved count."""
    total = 0.0
    for nb in range(R + 1):
        w = binom.pmf(nb, R, p_worse)
        if w < 1e-13:
            continue
        for na in range(R - nb + 1):
            d = [-1.0] * na + [1.0] * nb + [0.0] * (R - na - nb)
            if paired_road_bounds(d, alpha)[1] < t:
                total += w * binom.sf(na - 1, R - nb, p_better / (1 - p_worse))
                break
    return total


def design(**overrides):
    base = {
        "strata": {"a": 60, "b": 60},
        "p_a": 0.20,
        "q_plus": 0.0,
        "q_minus": 0.15,
        "margin": MARGIN,
        "guard_margins": (0.05, 0.10),
        "alpha": ALPHA,
        "delta_min": 0.02,
        "trials": 400,
        "seed": 20261108,
    }
    base.update(overrides)
    return spw.design_power(**base)


def _within(report, name, exact):
    low, high = report["results"][name]["improvement_interval"]
    return low <= exact <= high


def test_design_power_matches_exact_oracles():
    report = design()
    per_stratum = _exact_prob_upper_below(60, 0.15, 0.0, 0.0)
    assert _within(report, "all_strata", per_stratum**2)  # strata are drawn independently
    assert _within(report, "pooled", _exact_prob_upper_below(120, 0.15, 0.0, 0.0))
    assert (
        report["results"]["pooled"]["improvement_rate"]
        > report["results"]["all_strata"]["improvement_rate"]
    )


def test_guard_go_rate_matches_exact_oracle_under_no_harm():
    report = design(q_plus=0.02, q_minus=0.02, p_a=0.08, strata={"a": 110, "b": 110})
    for m in (0.05, 0.10):
        exact = _exact_prob_upper_below(110, 0.02, 0.02, m)
        for s in ("a", "b"):
            count = round(report["guard_go_rate"][f"{m:g}"][s] * report["trials"])
            low, high = spw._clopper_pearson(count, report["trials"])
            assert low <= exact <= high


def test_one_stratum_all_strata_equals_pooled_exactly():
    report = design(strata={"a": 120})
    assert report["results"]["all_strata"] == report["results"]["pooled"]


def test_loose_guard_on_large_strata_changes_nothing():
    report = design(strata={"a": 200, "b": 200}, guard_margins=(0.99,), q_minus=0.05)
    assert report["results"]["pooled_guard@0.99"] == report["results"]["pooled"]


def test_harmed_rare_stratum_is_blocked_by_the_guard_but_not_by_pooling():
    report = design(
        strata={"common": 400, "rare": 41},
        p_a=0.08,
        q_plus=0.0,
        q_minus=0.08,
        effects={"rare": (0.25, 0.0)},
        guard_margins=(0.10,),
        trials=300,
    )
    assert report["results"]["pooled"]["improvement_rate"] > 0.05
    assert report["results"]["pooled_guard@0.1"]["improvement_rate"] == 0.0
    assert report["results"]["all_strata"]["improvement_rate"] == 0.0


def test_null_effect_is_rarely_supported_by_any_design():
    report = design(q_plus=0.05, q_minus=0.05, p_a=0.10, strata={"a": 220, "b": 221})
    for name, result in report["results"].items():
        assert result["improvement_rate"] < 0.02, name


def test_design_power_is_deterministic_and_validates_inputs():
    assert design(trials=50) == design(trials=50)
    with pytest.raises(ValueError):
        design(guard_margins=(0.05, 0.05))
    with pytest.raises(ValueError):
        design(strata={"pooled": 60, "b": 60})
    with pytest.raises(ValueError):
        design(trials=0)


def test_improvement_rate_does_not_depend_on_delta_min():
    small, large = design(delta_min=0.02, trials=150), design(delta_min=0.9, trials=150)
    for name in small["results"]:
        assert (
            small["results"][name]["improvement_rate"]
            == (large["results"][name]["improvement_rate"])
        )
    assert any(sup.IMPROVED_BELOW_MIN in r["outcomes"] for r in large["results"].values())


def test_stricter_guard_margin_never_helps_and_binds_here():
    report = design(
        strata={"a": 110, "b": 110}, p_a=0.08, q_plus=0.02, q_minus=0.06, guard_margins=(0.05, 0.10)
    )
    r = report["results"]
    tight, loose = (
        r["pooled_guard@0.05"]["improvement_rate"],
        r["pooled_guard@0.1"]["improvement_rate"],
    )
    assert tight < loose <= r["pooled"]["improvement_rate"]


# --- review fixes: guard strata, paired observations, aggregation comparability, rates ------


DATA = rows("a", 220, better=20) + rows("b", 221, better=20)


def pooled_of(data, margin=MARGIN):
    return verdict([{**r, "stratum": "pooled"} for r in data], margin, ALPHA)


def pooled_decide(pooled, guard, **kw):
    return sup.decide(
        pooled,
        required_strata=("pooled",),
        guards={"ni": guard},
        guard_strata=("a", "b"),
        delta_min=0.02,
        **kw,
    )


def test_guard_missing_or_adding_a_registered_stratum_is_rejected():
    with pytest.raises(ValueError, match="registered guard strata"):
        pooled_decide(pooled_of(DATA), verdict(rows("a", 220, better=20), 0.10, ALPHA))
    with pytest.raises(ValueError, match="registered guard strata"):
        pooled_decide(pooled_of(DATA), verdict(DATA + rows("c", 50), 0.10, ALPHA))


def test_duplicate_or_empty_strata_are_rejected():
    result = verdict(rows("a", 50, better=50), MARGIN, ALPHA)
    for kwargs in (
        {"required_strata": ("a", "a")},
        {"required_strata": ("a",), "guard_strata": ("a", "a")},
        {"required_strata": ("a",), "guard_strata": ()},
        {"required_strata": ()},
    ):
        with pytest.raises(ValueError, match="unique nonempty"):
            sup.decide(result, guards={}, delta_min=0.02, **kwargs)


def test_pooled_primary_and_guard_must_cover_the_same_observations():
    other = rows("a", 220, better=20) + rows("b", 200, better=20)  # 21 roads fewer
    with pytest.raises(ValueError, match="same paired observations"):
        pooled_decide(pooled_of(DATA), verdict(other, 0.10, ALPHA))
    assert pooled_decide(pooled_of(DATA), verdict(DATA, 0.10, ALPHA))["outcome"] in (
        sup.SUPPORTED,
        sup.IMPROVED_BELOW_MIN,
    )


def test_guard_stratum_worse_bound_is_recorded():
    # Reviewer's case: a improves strongly, b is demonstrably worse yet within the 0.10 margin.
    data = rows("a", 300, better=60) + rows("b", 300, worse=12)
    guard = verdict(data, 0.10, ALPHA)
    assert guard.strata["b"] == "GO" and guard.details["b"].lower > 0
    d = pooled_decide(pooled_of(data), guard)
    assert d["outcome"] == sup.SUPPORTED
    assert d["guards"]["ni"]["strata"]["b"]["worse_bound_unadjusted"] is True
    assert d["any_guard_stratum_worse_bound_unadjusted"] is True
    assert d["any_stratum_worse_bound_unadjusted"] is False  # the pooled primary hides it


def test_aggregation_rejects_runs_with_different_guard_definitions():
    pooled = pooled_of(DATA)
    stratified_guard = pooled_decide(pooled, verdict(DATA, 0.10, ALPHA))
    pooled_guard = sup.decide(
        pooled, required_strata=("pooled",), guards={"ni": pooled_of(DATA, 0.10)}, delta_min=0.02
    )
    tighter = pooled_decide(pooled, verdict(DATA, 0.05, ALPHA))
    for other in (pooled_guard, tighter):
        with pytest.raises(ValueError, match="differ"):
            sup.aggregate_runs(
                {"r1": stratified_guard, "r2": other},
                registered_run_ids=("r1", "r2"),
                rule="all",
                family_alpha=ALPHA,
            )
    same = pooled_decide(pooled, verdict(DATA, 0.10, ALPHA))
    out = sup.aggregate_runs(
        {"r1": stratified_guard, "r2": same},
        registered_run_ids=("r1", "r2"),
        rule="all",
        family_alpha=ALPHA,
    )
    assert out["outcome"] in (sup.SUPPORTED, sup.IMPROVED_BELOW_MIN)


def test_design_rates_are_counts_over_trials():
    r = design(trials=60)
    for res in r["results"].values():
        assert sum(res["outcomes"].values()) == r["trials"]
        improved = res["outcomes"].get(sup.SUPPORTED, 0) + res["outcomes"].get(
            sup.IMPROVED_BELOW_MIN, 0
        )
        assert res["improvement_rate"] == improved / r["trials"]


def test_certain_guard_go_rates_are_exactly_one():
    r = design(strata={"a": 200, "b": 200}, guard_margins=(0.99,), q_minus=0.05, trials=40)
    assert r["guard_go_rate"]["0.99"] == {"a": 1.0, "b": 1.0}
    assert r["guard_joint_go_rate"]["0.99"] == 1.0


def test_joint_guard_rate_is_the_product_of_independent_strata():
    r = design(
        strata={"a": 110, "b": 110}, p_a=0.08, q_plus=0.02, q_minus=0.02, guard_margins=(0.10,)
    )
    per = _exact_prob_upper_below(110, 0.02, 0.02, 0.10)
    count = round(r["guard_joint_go_rate"]["0.1"] * r["trials"])
    low, high = spw._clopper_pearson(count, r["trials"])
    assert low <= per**2 <= high
    assert r["guard_joint_go_rate"]["0.1"] <= min(r["guard_go_rate"]["0.1"].values())


def test_design_all_strata_equals_superiority_power_counts():
    kw = {
        "strata": {"a": 60, "b": 60},
        "p_a": 0.2,
        "q_plus": 0.0,
        "q_minus": 0.15,
        "margin": MARGIN,
        "alpha": ALPHA,
        "delta_min": 0.02,
        "trials": 80,
        "seed": 20261108,
    }
    d = spw.design_power(guard_margins=(), **kw)
    s = spw.superiority_power(**kw)
    assert d["results"]["all_strata"]["outcomes"] == {
        k: v["count"] for k, v in s["outcomes"].items() if v["count"]
    }


def test_guard_margins_with_colliding_labels_are_rejected():
    with pytest.raises(ValueError, match="distinct labels"):
        design(guard_margins=(0.05, 0.05000001), trials=2)
    with pytest.raises(ValueError, match="distinct labels"):
        design(guard_margins=(np.float32(0.05), 0.05), trials=2)


def test_aggregation_rejects_runs_scored_on_different_primary_metrics():
    spec_metric = {"name": "collision_at_fault", "kind": "binary", "higher_is_worse": True}
    base = _decision(True)
    other = {**_decision(True), "metric": spec_metric}
    with pytest.raises(ValueError, match="differ"):
        sup.aggregate_runs(
            {"r1": base, "r2": other},
            registered_run_ids=("r1", "r2"),
            rule="all",
            family_alpha=ALPHA,
        )


# --- registered gating strata: small strata reported, not gated ---------------------------


def mixed_rows(n_small_better=2, n_small_worse=2):
    return rows("big", 300, better=60) + rows(
        "small", 41, better=n_small_better, worse=n_small_worse
    )


def gated(data, gating=None, margin=0.10):
    return sup.decide(
        pooled_of(data),
        required_strata=("pooled",),
        guards={"ni": verdict(data, margin, ALPHA)},
        guard_strata=("big", "small"),
        gating_strata=gating,
        delta_min=0.02,
    )


def test_small_stratum_can_be_reported_instead_of_gated():
    data = mixed_rows()
    guard = verdict(data, 0.10, ALPHA)
    assert guard.strata["big"] == "GO" and guard.strata["small"] == "INSUFFICIENT"
    assert gated(data)["outcome"] == sup.GUARD_NOT_CERTIFIED
    d = gated(data, gating=("big",))
    assert d["outcome"] == sup.SUPPORTED
    assert d["gating_strata"] == ["big"]
    assert d["guards"]["ni"]["strata"]["small"]["state"] == "INSUFFICIENT"  # still reported


def test_regression_in_a_non_gating_stratum_still_blocks():
    data = mixed_rows(n_small_better=0, n_small_worse=30)
    guard = verdict(data, 0.10, ALPHA)
    assert guard.strata["small"] == "NO-GO"
    assert gated(data, gating=("big",))["outcome"] == sup.GUARD_REGRESSION_FLAGGED


def test_gating_strata_must_be_a_registered_nonempty_subset():
    data = mixed_rows()
    for gating, error in ((("other",), ValueError), ((), ValueError), ("big", TypeError)):
        with pytest.raises(error):
            gated(data, gating=gating)


def test_aggregation_rejects_runs_with_different_gating():
    data = mixed_rows()
    a, b = gated(data, gating=("big",)), gated(data, gating=("big", "small"))
    with pytest.raises(ValueError, match="differ"):
        sup.aggregate_runs(
            {"r1": a, "r2": b}, registered_run_ids=("r1", "r2"), rule="all", family_alpha=ALPHA
        )


def test_design_power_prices_gating_between_full_guard_and_no_guard():
    r = design(
        strata={"big": 300, "small": 41},
        p_a=0.08,
        q_plus=0.02,
        q_minus=0.07,
        guard_margins=(0.10,),
        gating_strata=("big",),
        trials=300,
    )
    x = {k: v["improvement_rate"] for k, v in r["results"].items()}
    assert x["pooled_guard@0.1"] < x["pooled_gating@0.1"] <= x["pooled"]
    assert r["scenario"]["gating_strata"] == ["big"]
    with pytest.raises(ValueError, match="gating"):
        design(strata={"a": 60, "b": 60}, gating_strata=("zzz",), trials=2)


def test_aggregation_rejects_runs_that_differ_only_in_guard_strata():
    both = mixed_rows()
    big = rows("big", 300, better=60)
    d1 = gated(both, gating=("big",))
    d2 = sup.decide(
        pooled_of(big),
        required_strata=("pooled",),
        guards={"ni": verdict(big, 0.10, ALPHA)},
        guard_strata=("big",),
        gating_strata=("big",),
        delta_min=0.02,
    )
    assert d1["gating_strata"] == d2["gating_strata"] and d1["guard_strata"] != d2["guard_strata"]
    with pytest.raises(ValueError, match="differ"):
        sup.aggregate_runs(
            {"r1": d1, "r2": d2}, registered_run_ids=("r1", "r2"), rule="all", family_alpha=ALPHA
        )
