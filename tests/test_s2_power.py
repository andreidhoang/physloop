"""Known answers for S2 operational power and design sensitivity (CPU-06 remainder, W4).

Oracles are independent of ``gate.s2_power``:

- literal regressions from the readiness spec and D-021 addendum: the protected
  ``paired_road_bounds([-1.0]*5)`` upper ``0.04364750021196295`` and the all-zero minima
  36/51/72/183 roads at alpha 0.05 for margins 0.10/0.07/0.05/0.02, with each previous
  integer failing, cross-checked against the closed form ``1 - (alpha/2) ** (1/R)``;
- deterministic laws whose verdict is certain, so the probability is exactly 0 or 1;
- an exact enumeration of a tiny binary sample space (12 roads, three road-difference
  values, multinomial weights) evaluated with the protected ``gate.verdict.verdict``;
- byte-for-byte equality of the irregular-layout generator with the locked
  ``gate.synthetic.generate`` on regular layouts.

Monte Carlo tolerance (predeclared): ``|rate - exact| <= 4.5 * sqrt(p(1-p)/n)`` per outcome.
Under a fresh seed the normal-approximation false-failure probability is about 7e-6 per
comparison (2e-5 for the three outcomes); with the fixed seed the test is deterministic.
"""

import math
import time
from math import comb

import numpy as np
import pytest

from gate import superiority as sup
from gate.s2_power import (
    LAWS,
    MINIMUM_ROADS_SEARCH_CAP,
    generate_rows,
    run_sensitivity,
    zero_difference_minimum_roads,
    zero_difference_upper,
)
from gate.superiority_power import design_power, superiority_power
from gate.synthetic import generate
from gate.verdict import paired_road_bounds, verdict

CONTINUOUS = "paired_continuous_uniform_shift_v1"
BINARY = "paired_binary_copy_v1"


def design(layout, *, margin=0.10, alpha=0.05, kind="continuous", higher_is_worse=False,
           looks=None, strata=None, superiority=None):
    return {
        "schema_version": 2,
        "design_id": "synthetic-design-not-registered",
        "required_strata": sorted(layout) if strata is None else strata,
        "margin": margin,
        "alpha": alpha,
        "metric": {
            "name": "synthetic_metric",
            "kind": kind,
            "higher_is_worse": higher_is_worse,
            "unit": "score_in_0_1" if kind == "continuous" else "event",
        },
        "layout": layout,
        "looks": looks,
        "superiority": superiority,
    }


def roads(n, scenes=(1,)):
    return [list(scenes) for _ in range(n)]


def continuous(worse_by=0.0, *, low=0.5, high=0.5, noise=0.0, road_copy=0.0,
               replicate_copy=0.0, effects=None, scenario_id="continuous"):
    return {
        "scenario_id": scenario_id,
        "information_role": "test assumption",
        "law": CONTINUOUS,
        "parameters": {
            "reference_low": low,
            "reference_high": high,
            "candidate_worse_by": worse_by,
            "noise_half_width": noise,
        },
        "dependence": {
            "road_copy_probability": road_copy,
            "replicate_copy_probability": replicate_copy,
        },
        "stratum_effects": {} if effects is None else effects,
    }


def binary(p_a, q_plus, q_minus, *, road_copy=0.0, replicate_copy=0.0, effects=None,
           scenario_id="binary"):
    return {
        "scenario_id": scenario_id,
        "information_role": "test assumption",
        "law": BINARY,
        "parameters": {"p_a": p_a, "q_plus": q_plus, "q_minus": q_minus},
        "dependence": {
            "road_copy_probability": road_copy,
            "replicate_copy_probability": replicate_copy,
        },
        "stratum_effects": {} if effects is None else effects,
    }


def outcome_rates(report, index=0):
    return {name: item["rate"] for name, item in report["scenarios"][index]["outcomes"].items()}


def diagnostic_codes(report):
    return [item["code"] for item in report["diagnostics"]]


# --- literal regressions (W4) ----------------------------------------------------------


def test_five_roads_of_full_improvement_bound_literal():
    assert paired_road_bounds([-1.0] * 5) == (-1.0, 0.04364750021196295)


@pytest.mark.parametrize(("margin", "minimum"), [(0.10, 36), (0.07, 51), (0.05, 72),
                                                 (0.02, 183)])
def test_all_zero_minimum_roads_and_previous_integer_fails(margin, minimum):
    assert zero_difference_minimum_roads(0.05, margin) == minimum
    assert zero_difference_upper(minimum, 0.05) < margin
    assert zero_difference_upper(minimum - 1, 0.05) >= margin
    for n in (minimum - 1, minimum):
        # Independent closed form; the protected bound pads outward by 1e-12.
        assert zero_difference_upper(n, 0.05) == pytest.approx(1 - 0.025 ** (1 / n), abs=2e-12)


@pytest.mark.parametrize("bad", [0, -1, True, 2.0])
def test_zero_difference_upper_rejects_noninteger_or_empty_road_counts(bad):
    with pytest.raises(ValueError):
        zero_difference_upper(bad, 0.05)


@pytest.mark.parametrize(("alpha", "margin"), [(0.0, 0.1), (0.05, 1.0), (True, 0.1)])
def test_zero_difference_minimum_rejects_invalid_probabilities(alpha, margin):
    with pytest.raises(ValueError):
        zero_difference_minimum_roads(alpha, margin)


# A7-05: the protected bound pads every upper endpoint outward by 1e-12, so the search for the
# all-zero minimum is capped (agent technical default) instead of looping or allocating a
# road list of ~3.7e12 entries for a margin the padded bound cannot certify.
def test_minimum_roads_search_returns_none_beyond_the_cap():
    assert MINIMUM_ROADS_SEARCH_CAP == 10**6
    # Closed form 1 - (alpha/2) ** (1/R) at R = 2e6: the minimum lies just above 2e6 roads.
    margin = -math.expm1(math.log(0.025) / 2_000_000)
    assert zero_difference_minimum_roads(0.05, margin) is None


@pytest.mark.parametrize("margin", [1e-12, 1e-13, 5e-324, 2e-12, 1e-9])
def test_margin_the_padded_bound_cannot_certify_within_the_cap_returns_promptly(margin):
    start = time.perf_counter()
    assert zero_difference_minimum_roads(0.05, margin) is None
    assert time.perf_counter() - start < 5.0


def test_explicit_search_cap_known_answers():
    assert zero_difference_minimum_roads(0.05, 0.10, max_roads=35) is None
    assert zero_difference_minimum_roads(0.05, 0.10, max_roads=36) == 36
    assert zero_difference_minimum_roads(0.05, 0.10, max_roads=1) is None


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, None])
def test_search_cap_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError):
        zero_difference_minimum_roads(0.05, 0.10, max_roads=bad)


def test_uncertifiable_tiny_margin_is_reported_not_searched_forever():
    report = run_sensitivity(design({"u": roads(3)}, margin=1e-13), [continuous()], 1, 1)
    [entry] = report["zero_difference_control"]
    assert entry["minimum_roads_at_alpha"] is None
    assert entry["minimum_roads_search_cap"] == MINIMUM_ROADS_SEARCH_CAP
    assert entry["certifiable"] is False
    assert diagnostic_codes(report) == [
        "zero_difference_control_not_certifiable",
        "zero_difference_minimum_not_found_within_cap",
    ]
    assert all(item["severity"] == "warning" for item in report["diagnostics"])


# --- deterministic known answers through the actual continuous gate path ---------------


def test_zero_difference_continuous_at_thirty_six_roads_always_goes():
    report = run_sensitivity(design({"u": roads(36)}), [continuous()], 5, 1)
    assert report["status"] == "passed"
    assert outcome_rates(report) == {"GO": 1.0, "NO-GO": 0.0, "INSUFFICIENT": 0.0}
    assert "zero_difference_control_not_certifiable" not in diagnostic_codes(report)


def test_zero_difference_continuous_at_thirty_five_roads_never_goes():
    report = run_sensitivity(design({"u": roads(35)}), [continuous()], 5, 1)
    assert outcome_rates(report) == {"GO": 0.0, "NO-GO": 0.0, "INSUFFICIENT": 1.0}
    [entry] = report["zero_difference_control"]
    assert entry["roads"] == 35 and entry["certifiable"] is False
    assert entry["upper"] == zero_difference_upper(35, 0.05)
    assert diagnostic_codes(report) == ["zero_difference_control_not_certifiable"]


def test_five_road_improvement_counterexample_goes_below_the_zero_difference_floor():
    # Higher-is-better score: reference 0, candidate 1 on every road (candidate better by 1).
    scenario = continuous(-1.0, low=0.0, high=0.0)
    report = run_sensitivity(design({"u": roads(5)}, margin=0.05), [scenario], 4, 7)
    assert report["status"] == "passed"
    assert outcome_rates(report)["GO"] == 1.0
    # The zero-difference control is a diagnostic, not a universal road-floor blocker.
    assert "zero_difference_control_not_certifiable" in diagnostic_codes(report)
    [entry] = report["zero_difference_control"]
    assert (entry["roads"], entry["alpha"], entry["margin"]) == (5, 0.05, 0.05)
    assert entry["minimum_roads_at_alpha"] == 72
    assert all(item["severity"] == "warning" for item in report["diagnostics"])


@pytest.mark.parametrize("layout", [roads(35, (3,)), roads(35, (1, 1)), roads(35, (2, 1, 4))])
def test_extra_replicates_or_scenes_do_not_add_independent_roads(layout):
    report = run_sensitivity(design({"u": layout}), [continuous()], 3, 2)
    assert outcome_rates(report)["INSUFFICIENT"] == 1.0
    summary = report["layout_summary"]["u"]
    assert summary["roads"] == 35
    assert summary["rollout_pairs"] == sum(sum(road) for road in layout)


def test_missing_required_stratum_cannot_go():
    layout = {"urban": roads(40), "highway": []}
    report = run_sensitivity(design(layout), [continuous()], 3, 3)
    assert outcome_rates(report) == {"GO": 0.0, "NO-GO": 0.0, "INSUFFICIENT": 1.0}
    assert "stratum_without_roads" in diagnostic_codes(report)


def test_candidate_worse_by_a_known_constant_gives_no_go_with_certainty():
    # Higher-is-worse continuous metric, candidate worse by 0.5 on all 40 roads:
    # Holm p = exp(-40 * 0.25 / 2) = exp(-5) < 0.05.
    scenario = continuous(0.5, low=0.2, high=0.4, noise=0.05)
    report = run_sensitivity(design({"u": roads(40)}, higher_is_worse=True), [scenario], 6, 4)
    assert outcome_rates(report)["NO-GO"] == 1.0


# --- sequential: actual per-look alpha, complete-road batches, registered looks only -----


def sequential(counts, weights=(0.5, 0.5), futility=None):
    return {
        "weights": list(weights),
        "futility_at": {} if futility is None else futility,
        "roads_per_look": counts,
    }


def test_sequential_uses_spent_alpha_per_look():
    # alpha .05 x .5 = .025 per look; all-zero threshold at .025 / margin .10 is 42 roads.
    report = run_sensitivity(
        design({"u": roads(80)}, looks=sequential({"u": [40, 40]})), [continuous()], 4, 5
    )
    scenario = report["scenarios"][0]
    assert outcome_rates(report)["GO"] == 1.0
    assert scenario["stopped_at_look"] == {"1": 0, "2": 4}
    assert scenario["expected_looks"] == 2.0
    assert scenario["expected_roads"] == 80.0
    assert report["look_alphas"] == [0.025, 0.025]
    looks = {(e["look"], e["roads"]): e["certifiable"] for e in report["zero_difference_control"]}
    assert looks == {(1, 40): False, (2, 80): True}
    assert zero_difference_minimum_roads(0.025, 0.10) == 42


def test_sequential_stops_early_and_reports_saved_roads():
    report = run_sensitivity(
        design({"u": roads(80)}, looks=sequential({"u": [42, 38]})), [continuous()], 3, 5
    )
    scenario = report["scenarios"][0]
    assert scenario["stopped_at_look"] == {"1": 3, "2": 0}
    assert scenario["expected_roads"] == 42.0
    assert scenario["expected_rollout_pairs"] == 42.0


def test_fixed_sample_at_forty_roads_goes_where_the_first_look_did_not():
    report = run_sensitivity(design({"u": roads(40)}), [continuous()], 3, 5)
    assert outcome_rates(report)["GO"] == 1.0


def test_no_unregistered_extra_look_is_taken():
    report = run_sensitivity(
        design({"u": roads(20)}, looks=sequential({"u": [10, 10]})), [continuous()], 3, 6
    )
    scenario = report["scenarios"][0]
    assert outcome_rates(report)["INSUFFICIENT"] == 1.0
    assert scenario["stopped_at_look"] == {"1": 0, "2": 3}
    assert max(int(k) for k in scenario["stopped_at_look"]) == 2


def test_registered_futility_stops_at_its_look():
    looks = sequential({"u": [10, 10]}, futility={"1": ["u"]})
    report = run_sensitivity(design({"u": roads(20)}, looks=looks), [continuous()], 3, 6)
    scenario = report["scenarios"][0]
    assert scenario["stopped_at_look"] == {"1": 3, "2": 0}
    assert outcome_rates(report)["INSUFFICIENT"] == 1.0


def test_closed_stratum_receives_no_further_roads():
    # Stratum a certifies at look 1 (42 roads); b needs both looks.
    looks = sequential({"a": [42, 38], "b": [40, 40]})
    report = run_sensitivity(
        design({"a": roads(80), "b": roads(80)}, looks=looks), [continuous()], 2, 8
    )
    scenario = report["scenarios"][0]
    assert outcome_rates(report)["GO"] == 1.0
    assert scenario["expected_roads"] == 42 + 80


def test_sequential_design_requires_every_stratum_in_every_look():
    with pytest.raises(ValueError, match="roads_per_look"):
        run_sensitivity(design({"u": roads(10)}, looks=sequential({"u": [10, 0]})),
                        [continuous()], 1, 0)
    with pytest.raises(ValueError, match="roads_per_look"):
        run_sensitivity(design({"u": roads(10)}, looks=sequential({"u": [4, 4]})),
                        [continuous()], 1, 0)


# --- exact enumeration oracle (binary) -----------------------------------------------


def _exact_binary_verdict_probabilities(n_roads, p_plus, p_minus, margin, alpha):
    exact = {"GO": 0.0, "NO-GO": 0.0, "INSUFFICIENT": 0.0}
    for n_plus in range(n_roads + 1):
        for n_minus in range(n_roads + 1 - n_plus):
            n_zero = n_roads - n_plus - n_minus
            weight = (
                comb(n_roads, n_plus) * comb(n_roads - n_plus, n_minus)
                * p_plus**n_plus * p_minus**n_minus * (1 - p_plus - p_minus) ** n_zero
            )
            diffs = [1] * n_plus + [-1] * n_minus + [0] * n_zero
            rows = [
                {"scene_id": f"s{i}", "road_id": f"r{i}", "stratum": "u",
                 "a_fail": d == -1, "b_fail": d == 1}
                for i, d in enumerate(diffs)
            ]
            exact[verdict(rows, margin, alpha, required_strata=("u",)).overall] += weight
    return exact


def test_binary_monte_carlo_matches_exact_enumeration_within_predeclared_tolerance():
    # Categories (neither, A-only, B-only, both) = (.1, .1, .7, .1): road difference +1
    # w.p. .7, -1 w.p. .1, 0 w.p. .2.
    exact = _exact_binary_verdict_probabilities(12, 0.7, 0.1, 0.9, 0.05)
    assert sum(exact.values()) == pytest.approx(1.0, abs=1e-12)
    assert min(exact.values()) > 0.25  # every outcome is exercised
    n_sim = 2000
    report = run_sensitivity(
        design({"u": roads(12)}, margin=0.9, kind="binary", higher_is_worse=True),
        [binary(0.2, 0.7, 0.1)],
        n_sim,
        20261109,
    )
    for name, item in report["scenarios"][0]["outcomes"].items():
        p = exact[name]
        assert abs(item["rate"] - p) <= 4.5 * math.sqrt(p * (1 - p) / n_sim), (name, item, p)
        assert item["standard_error"] == pytest.approx(
            math.sqrt(item["rate"] * (1 - item["rate"]) / n_sim)
        )
        low, high = item["interval"]
        assert 0 <= low <= item["rate"] <= high <= 1
    assert sum(item["count"] for item in report["scenarios"][0]["outcomes"].values()) == n_sim


def test_binary_success_metric_maps_failures_to_zero_values():
    # higher_is_worse False (a success indicator): the same failures, the same verdicts.
    worse = run_sensitivity(
        design({"u": roads(12)}, margin=0.9, kind="binary", higher_is_worse=True),
        [binary(0.2, 0.7, 0.1)], 50, 11,
    )
    better = run_sensitivity(
        design({"u": roads(12)}, margin=0.9, kind="binary", higher_is_worse=False),
        [binary(0.2, 0.7, 0.1)], 50, 11,
    )
    assert outcome_rates(worse) == outcome_rates(better)


def test_zero_discordance_binary_law_has_a_not_estimable_share_and_certain_go():
    report = run_sensitivity(
        design({"u": roads(36)}, kind="binary", higher_is_worse=True),
        [binary(0.1, 0.0, 0.0)], 4, 12,
    )
    derived = report["scenarios"][0]["derived"]
    assert derived["discordance"] == 0.0
    assert derived["mean_difference"] == 0.0
    assert derived["candidate_worse_share_of_discordant"] == "not_estimable"
    assert outcome_rates(report)["GO"] == 1.0


def test_binary_derived_quantities_known_answer():
    report = run_sensitivity(
        design({"u": roads(3)}, kind="binary", higher_is_worse=True),
        [binary(0.1, 0.06, 0.02)], 1, 0,
    )
    derived = report["scenarios"][0]["derived"]["by_stratum"]["u"]
    assert derived["discordance"] == pytest.approx(0.08)
    assert derived["mean_difference"] == pytest.approx(0.04)
    assert derived["candidate_worse_share_of_discordant"] == pytest.approx(0.75)
    assert derived["candidate_event_rate"] == pytest.approx(0.14)


# --- generator extension: regular-layout equality and irregular known answers -----------


def test_irregular_generator_reproduces_locked_synthetic_generator_on_regular_layouts():
    layout = {"a": roads(3, (3, 3)), "b": roads(2, (3, 3))}
    scenario = binary(0.1, 0.05, 0.02, road_copy=0.3, replicate_copy=0.6,
                      effects={"b": [0.1, 0.0]})
    for seed in (0, 1, 987654321):
        ours = generate_rows(design(layout, kind="binary", higher_is_worse=True), scenario, seed)
        locked = generate(seed, {"a": 3, "b": 2}, scenes_per_road=2, replicates=3, p_a=0.1,
                          q_plus=0.05, q_minus=0.02, road_correlation=0.3,
                          replicate_correlation=0.6, effects={"b": (0.1, 0.0)})
        assert ours == locked


def test_full_copy_makes_every_rollout_in_a_road_identical():
    layout = {"u": [[1, 3, 2], [2], [4, 1]]}
    rows = generate_rows(design(layout, kind="binary"),
                         binary(0.4, 0.3, 0.2, road_copy=1.0, replicate_copy=1.0), 5)
    assert len(rows) == 6 + 2 + 5
    for road in {row["road_id"] for row in rows}:
        assert len({(r["a_fail"], r["b_fail"]) for r in rows if r["road_id"] == road}) == 1


def test_replicate_copy_makes_replicates_of_a_scene_identical():
    layout = {"u": [[5, 5], [5]]}
    rows = generate_rows(design(layout), continuous(0.0, low=0.1, high=0.9, noise=0.05,
                                                    replicate_copy=1.0), 9)
    scenes = {row["scene_id"] for row in rows}
    assert len(scenes) == 3
    for scene in scenes:
        values = {(r["a_value"], r["b_value"]) for r in rows if r["scene_id"] == scene}
        assert len(values) == 1
    assert len({(r["a_value"], r["b_value"]) for r in rows}) == 3


def test_irregular_binary_marginals_match_the_declared_joint_law():
    layout = {"u": [[1 + (i % 3)] * (1 + (i % 2)) for i in range(3000)]}
    rows = generate_rows(design(layout, kind="binary"), binary(0.3, 0.2, 0.1), 21)
    n = len(rows)
    # (neither, A-only, B-only, both) = (1 - .3 - .2, .1, .2, .3 - .1).
    expected = {(False, False): 0.5, (True, False): 0.1, (False, True): 0.2, (True, True): 0.2}
    for key, p in expected.items():
        observed = sum((r["a_fail"], r["b_fail"]) == key for r in rows) / n
        assert abs(observed - p) <= 5 * math.sqrt(p * (1 - p) / n), (key, observed)


def test_irregular_layout_identities_and_order():
    rows = generate_rows(design({"u": [[2, 1], [1]]}), continuous(), 0)
    assert [(r["road_id"], r["scene_id"], r["replicate_id"]) for r in rows] == [
        ("u/road-000000", "u/road-000000/scene-0000", 0),
        ("u/road-000000", "u/road-000000/scene-0000", 1),
        ("u/road-000000", "u/road-000000/scene-0001", 0),
        ("u/road-000001", "u/road-000001/scene-0000", 0),
    ]


@pytest.mark.parametrize("higher_is_worse", [True, False])
def test_continuous_generator_known_means_and_bounds(higher_is_worse):
    layout = {"u": roads(4000)}
    scenario = continuous(0.1, low=0.3, high=0.7, noise=0.2)
    rows = generate_rows(design(layout, higher_is_worse=higher_is_worse), scenario, 13)
    a = np.array([r["a_value"] for r in rows])
    b = np.array([r["b_value"] for r in rows])
    assert a.min() >= 0.3 and a.max() <= 0.7
    assert b.min() >= 0 and b.max() <= 1
    gate_difference = b - a if higher_is_worse else a - b
    assert np.all(np.abs(gate_difference - 0.1) <= 0.2 + 1e-12)
    # Uniform means: a ~ U(.3,.7) has mean .5, sd .4/sqrt(12); difference has mean .1,
    # sd .4/sqrt(12). Tolerance 5 standard errors.
    se = 0.4 / math.sqrt(12) / math.sqrt(len(rows))
    assert abs(a.mean() - 0.5) <= 5 * se
    assert abs(gate_difference.mean() - 0.1) <= 5 * se


def test_continuous_zero_noise_difference_is_exact():
    rows = generate_rows(design({"u": roads(10)}), continuous(0.25, low=0.4, high=0.6), 3)
    assert all(r["a_value"] - r["b_value"] == pytest.approx(0.25, abs=1e-15) for r in rows)


def test_continuous_stratum_effect_overrides_only_its_stratum():
    layout = {"a": roads(5), "b": roads(5)}
    rows = generate_rows(design(layout), continuous(0.0, effects={"b": 0.2}), 3)
    assert {round(r["a_value"] - r["b_value"], 12) for r in rows if r["stratum"] == "a"} == {0.0}
    assert {round(r["a_value"] - r["b_value"], 12) for r in rows if r["stratum"] == "b"} == {0.2}


# --- unsupported or infeasible laws fail closed ----------------------------------------


@pytest.mark.parametrize(
    "scenario",
    [
        binary(0.01, 0.0, 0.02),  # q_minus > p_a: no joint law
        binary(0.5, 0.6, 0.0),  # p_a + q_plus > 1
        continuous(0.3, low=0.0, high=0.5),  # higher-is-better: b = a - d reaches below 0
        continuous(0.0, low=0.6, high=0.4),  # empty reference range
        continuous(0.0, noise=-0.1),
        continuous(0.0, effects={"other": 0.1}),  # effect names an unregistered stratum
    ],
)
def test_infeasible_laws_are_rejected(scenario):
    kind = "binary" if scenario["law"] == BINARY else "continuous"
    with pytest.raises(ValueError):
        run_sensitivity(design({"u": roads(3)}, kind=kind), [scenario], 1, 0)


def test_law_and_metric_kind_must_agree_no_binarizing():
    with pytest.raises(ValueError, match="unsupported"):
        run_sensitivity(design({"u": roads(3)}, kind="binary"), [continuous()], 1, 0)
    with pytest.raises(ValueError, match="unsupported"):
        run_sensitivity(design({"u": roads(3)}), [binary(0.1, 0.0, 0.0)], 1, 0)


def test_unknown_law_is_rejected():
    scenario = continuous()
    scenario["law"] = "normal_approximation"
    assert "normal_approximation" not in LAWS
    with pytest.raises(ValueError, match="unsupported"):
        run_sensitivity(design({"u": roads(3)}), [scenario], 1, 0)


# --- input validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["schema_version", "design_id", "required_strata", "margin", "alpha", "metric", "layout",
     "looks", "superiority"],
)
def test_every_design_field_is_required(field):
    value = design({"u": roads(3)})
    del value[field]
    with pytest.raises(ValueError, match=field):
        run_sensitivity(value, [continuous()], 1, 0)


@pytest.mark.parametrize(
    "field",
    ["scenario_id", "information_role", "law", "parameters", "dependence", "stratum_effects"],
)
def test_every_scenario_field_is_required(field):
    scenario = continuous()
    del scenario[field]
    with pytest.raises(ValueError, match=field):
        run_sensitivity(design({"u": roads(3)}), [scenario], 1, 0)


@pytest.mark.parametrize(
    ("layout", "strata"),
    [
        ({"u": roads(3), "v": roads(3)}, ["u"]),  # unregistered stratum in layout
        ({"u": roads(3)}, ["u", "v"]),  # registered stratum absent from layout
        ({"u": [[]]}, ["u"]),  # road without scenes
        ({"u": [[0]]}, ["u"]),  # scene without replicates
        ({"u": [[True]]}, ["u"]),
    ],
)
def test_invalid_layouts_are_rejected(layout, strata):
    with pytest.raises(ValueError, match="layout"):
        run_sensitivity(design(layout, strata=strata), [continuous()], 1, 0)


@pytest.mark.parametrize(("n_sim", "seed"), [(0, 0), (True, 0), (1.0, 0), (1, -1), (1, True)])
def test_simulation_count_and_seed_are_validated(n_sim, seed):
    with pytest.raises(ValueError):
        run_sensitivity(design({"u": roads(3)}), [continuous()], n_sim, seed)


def test_unknown_design_field_and_duplicate_scenarios_are_rejected():
    value = design({"u": roads(3)})
    value["default_margin"] = 0.1
    with pytest.raises(ValueError, match="unknown"):
        run_sensitivity(value, [continuous()], 1, 0)
    with pytest.raises(ValueError, match="scenario_id"):
        run_sensitivity(design({"u": roads(3)}), [continuous(), continuous()], 1, 0)
    with pytest.raises(ValueError, match="scenarios"):
        run_sensitivity(design({"u": roads(3)}), [], 1, 0)


def test_sequential_weights_must_be_valid():
    with pytest.raises(ValueError):
        run_sensitivity(
            design({"u": roads(4)}, looks=sequential({"u": [2, 2]}, weights=(0.5, 0.6))),
            [continuous()], 1, 0,
        )


# --- report contract -------------------------------------------------------------------


def test_report_contract_labels_and_determinism():
    args = (design({"u": roads(12)}, margin=0.9, kind="binary"),
            [binary(0.2, 0.7, 0.1, road_copy=0.2), binary(0.2, 0.5, 0.1, scenario_id="b2")],
            40, 99)
    first, second = run_sensitivity(*args), run_sensitivity(*args)
    assert first == second
    assert first["artifact_kind"] == "s2_power_sensitivity"
    assert first["schema_version"] == 1
    assert first["scope"] == "synthetic"
    assert first["classification"] == "assumption_not_measurement"
    assert first["producer"]["module"] == "gate.s2_power"
    assert set(first["input_hashes"]) == {"design", "scenarios"}
    assert first["n_sim"] == 40 and first["seed"] == 99
    assert first["analysis"] == "fixed_sample"
    assert first["inference_path"] == "gate.api.evaluate"
    assert first["monte_carlo"]["interval_method"] == "clopper_pearson_two_sided_0.95"
    assert "no guarantee" in first["monte_carlo"]["interval_note"]
    for scenario in first["scenarios"]:
        assert scenario["dependence"]["label"] == "copy_probability_not_icc"
        assert scenario["classification"] == "assumption_not_measurement"
    assert first["issues"] == []


def test_sequential_report_names_the_sequential_gate():
    report = run_sensitivity(
        design({"u": roads(4)}, looks=sequential({"u": [2, 2]})), [continuous()], 1, 0
    )
    assert report["analysis"] == "sequential"
    assert report["inference_path"] == "gate.sequential.SequentialGate"


def test_scenarios_share_common_random_numbers_by_trial():
    base = design({"u": roads(12)}, margin=0.9, kind="binary")
    together = run_sensitivity(
        base, [binary(0.2, 0.7, 0.1), binary(0.2, 0.7, 0.1, scenario_id="copy")], 60, 4
    )
    alone = run_sensitivity(base, [binary(0.2, 0.7, 0.1)], 60, 4)
    assert together["scenarios"][0]["outcomes"] == together["scenarios"][1]["outcomes"]
    assert together["scenarios"][0]["outcomes"] == alone["scenarios"][0]["outcomes"]


# --- F4: superiority and falsifier power alongside non-inferiority ---------------------------
#
# Oracles: the D-023 illustrative 441-road single-stratum cases recorded in
# evidence/superiority-rule-20261108-v2/power.json (rule v1, fixed sample, no guards; seed
# 20261108, 1000 trials, margin 0.05, alpha 0.05, delta_min 0.016; R441_b0_net3.0pp 850/150,
# R441_null_b0 ruled out 1000/1000, two strata 220/221 18/982), which the current rule v4
# reproduces through gate.superiority_power; gate.superiority_power.design_power on the same
# rows for pooled/guard/gating designs; and certain outcomes whose endpoints the protected
# all-zero bound fixes (zero_difference_upper(36, .05) < .10 <= zero_difference_upper(35, .05)).


def claim(delta_min, *, primary="registered_strata", pooled_label=None, guards=(),
          gating=()):
    return {"delta_min": delta_min, "primary": primary, "pooled_label": pooled_label,
            "guards": [dict(guard) for guard in guards], "gating_strata": list(gating)}


D023 = {"R441_b0_net3.0pp": ((0.08, 0.0, 0.030), {"IMPROVEMENT_SUPPORTED": 850,
                                                   "INSUFFICIENT": 150}),
        "R441_null_b0": ((0.08, 0.0, 0.0), {"MEANINGFUL_IMPROVEMENT_RULED_OUT": 1000}),
        # review F4-05: the three remaining recorded 441-road literals
        "R441_b0_net1.6pp_target": ((0.08, 0.0, 0.016), {
            "IMPROVEMENT_SUPPORTED": 188, "MEANINGFUL_IMPROVEMENT_RULED_OUT": 7,
            "INSUFFICIENT": 805}),
        "R441_b5_net9.5pp": ((0.15, 0.05, 0.145), {"IMPROVEMENT_SUPPORTED": 847,
                                                    "INSUFFICIENT": 153}),
        "R441_null_b5": ((0.08, 0.05, 0.05), {"MEANINGFUL_IMPROVEMENT_RULED_OUT": 7,
                                              "INSUFFICIENT": 993})}


def superiority_counts(scenario):
    return {name: value["count"] for name, value in scenario["superiority"]["outcomes"].items()
            if value["count"]}


def test_d023_single_stratum_441_road_superiority_and_falsifier_power_literals():
    scenarios = [binary(*law, scenario_id=name) for name, (law, _) in D023.items()]
    report = run_sensitivity(
        design({"a": roads(441)}, margin=0.05, kind="binary", higher_is_worse=True,
               superiority=claim(0.016)),
        scenarios, 1000, 20261108)
    assert report["superiority"]["status"] == "passed"
    assert report["superiority"]["inference_path"] == "gate.superiority.decide"
    assert report["superiority"]["rule_id"] == sup.RULE_ID
    for scenario in report["scenarios"]:
        expected = D023[scenario["scenario_id"]][1]
        assert superiority_counts(scenario) == expected
        improved = expected.get("IMPROVEMENT_SUPPORTED", 0)
        ruled = expected.get("MEANINGFUL_IMPROVEMENT_RULED_OUT", 0)
        section = scenario["superiority"]
        assert section["improvement_power"]["count"] == improved
        assert section["falsifier_power"]["count"] == ruled
        assert section["improvement_power"]["rate"] == improved / 1000
        # non-inferiority stays reported alongside, unchanged in shape
        assert set(scenario["outcomes"]) == {"GO", "NO-GO", "INSUFFICIENT"}


def test_small_trial_count_equals_superiority_power_on_the_same_seed():
    """Independent path: gate.synthetic.generate + gate.verdict.verdict + decide."""
    report = run_sensitivity(
        design({"a": roads(441)}, margin=0.05, kind="binary", higher_is_worse=True,
               superiority=claim(0.016)),
        [binary(0.08, 0.0, 0.016)], 60, 20261108)
    oracle = superiority_power(strata={"a": 441}, p_a=0.08, q_plus=0.0, q_minus=0.016,
                               margin=0.05, alpha=0.05, delta_min=0.016, trials=60,
                               seed=20261108)
    assert superiority_counts(report["scenarios"][0]) == {
        name: value["count"] for name, value in oracle["outcomes"].items() if value["count"]}


def test_two_strata_all_strata_claim_matches_the_recorded_collapse():
    """D-023 'two_strata_220_each_b0_net3pp' (220/221 roads): 18 of 1000 supported."""
    report = run_sensitivity(
        design({"a": roads(220), "b": roads(221)}, margin=0.05, kind="binary",
               higher_is_worse=True, superiority=claim(0.016)),
        [binary(0.08, 0.0, 0.030)], 1000, 20261108)
    assert superiority_counts(report["scenarios"][0]) == {"IMPROVEMENT_SUPPORTED": 18,
                                                          "INSUFFICIENT": 982}


@pytest.mark.parametrize(("n_roads", "outcome"), [
    (36, sup.RULED_OUT), (35, sup.INSUFFICIENT)])
def test_identical_outputs_are_ruled_out_exactly_at_the_zero_difference_floor(n_roads, outcome):
    assert zero_difference_upper(36, 0.05) < 0.10 <= zero_difference_upper(35, 0.05)
    report = run_sensitivity(design({"u": roads(n_roads)}, superiority=claim(0.10)),
                             [continuous()], 4, 2)
    section = report["scenarios"][0]["superiority"]
    assert superiority_counts(report["scenarios"][0]) == {outcome: 4}
    assert section["falsifier_power"]["rate"] == (1.0 if outcome == sup.RULED_OUT else 0.0)
    assert section["improvement_power"]["rate"] == 0.0


def test_certain_improvement_is_supported_and_not_ruled_out():
    # higher is better: reference 0.5, candidate 1.0 on every road (difference -0.5).
    # Protected bounds: 40 roads (-0.705, -0.207) -> improved, not ruled out at 0.10;
    # 10 roads (-0.861, +0.170) -> neither.
    assert paired_road_bounds([-0.5] * 40, 0.05)[1] < 0 < paired_road_bounds([-0.5] * 10,
                                                                              0.05)[1]
    report = run_sensitivity(design({"u": roads(40)}, superiority=claim(0.10)),
                             [continuous(-0.5)], 3, 0)
    assert superiority_counts(report["scenarios"][0]) == {sup.SUPPORTED: 3}
    few = run_sensitivity(design({"u": roads(10)}, superiority=claim(0.10)),
                          [continuous(-0.5)], 3, 0)
    assert superiority_counts(few["scenarios"][0]) == {sup.INSUFFICIENT: 3}


def test_certain_regression_counts_toward_the_ruled_out_claim_not_improvement():
    """Review F4-01: falsifier_power counts ``claims.ruled_out.holds`` (the D-023 per-stratum
    condition lower > -delta_min, as ``aggregate_runs`` counts claims), so a certain regression
    counts. Higher is better, candidate worse by 0.5 on 40 roads: differences +0.5, protected
    bounds (0.207, 0.705) -> NO-GO and lower 0.207 > -0.10 (ruled out)."""
    lower, upper = paired_road_bounds([0.5] * 40, 0.05)
    assert lower > 0 and upper > 0.10
    report = run_sensitivity(design({"u": roads(40)}, superiority=claim(0.10)),
                             [continuous(0.5)], 5, 0)
    scenario = report["scenarios"][0]
    assert scenario["outcomes"]["NO-GO"]["count"] == 5
    assert superiority_counts(scenario) == {sup.REGRESSION_FLAGGED: 5}
    section = scenario["superiority"]
    assert section["falsifier_power"]["count"] == 5
    assert section["falsifier_power"]["rate"] == 1.0
    assert section["improvement_power"]["count"] == 0
    definition = section["falsifier_power"]["definition"]
    for outcome in (sup.REGRESSION_FLAGGED, sup.GUARD_REGRESSION_FLAGGED,
                    sup.GUARD_NOT_CERTIFIED, sup.RULED_OUT, sup.IMPROVED_BELOW_MIN):
        assert outcome in definition
    assert "claims.ruled_out.holds" in definition
    assert "agent technical default (D-018)" in definition


def test_registered_stratum_without_roads_is_missing_never_improved():
    report = run_sensitivity(design({"u": roads(40), "v": []}, superiority=claim(0.10)),
                             [continuous(-0.5)], 2, 0)
    assert superiority_counts(report["scenarios"][0]) == {sup.INSUFFICIENT: 2}


def _design_power_counts(result, name):
    return {key: value for key, value in result["results"][name]["outcomes"].items()}


@pytest.mark.parametrize(("spec", "oracle_name"), [
    (claim(0.03), "all_strata"),
    (claim(0.03, primary="pooled", pooled_label="pooled"), "pooled"),
    (claim(0.03, primary="pooled", pooled_label="pooled",
           guards=[{"name": "same_endpoint_noninferiority", "margin": 0.1}],
           gating=["a", "b"]), "pooled_guard@0.1"),
    (claim(0.03, primary="pooled", pooled_label="pooled",
           guards=[{"name": "same_endpoint_noninferiority", "margin": 0.1}],
           gating=["a"]), "pooled_gating@0.1"),
])
def test_pooled_guard_and_gating_designs_equal_design_power_on_the_same_rows(spec, oracle_name):
    oracle = design_power(strata={"a": 60, "b": 61}, p_a=0.08, q_plus=0.01, q_minus=0.07,
                          margin=0.05, guard_margins=(0.1,), alpha=0.05, delta_min=0.03,
                          trials=60, seed=11, gating_strata=("a",))
    report = run_sensitivity(
        design({"a": roads(60), "b": roads(61)}, margin=0.05, kind="binary",
               higher_is_worse=True, superiority=spec),
        [binary(0.08, 0.01, 0.07)], 60, 11)
    assert superiority_counts(report["scenarios"][0]) == _design_power_counts(oracle,
                                                                              oracle_name)


@pytest.mark.parametrize(("gating", "outcome"), [
    (["a", "b"], sup.GUARD_NOT_CERTIFIED), (["a"], sup.SUPPORTED)])
def test_pooled_claim_with_a_flat_small_gating_stratum_is_not_certified(gating, outcome):
    # a: 40 roads at -0.5, b: 5 flat roads. Pooled bounds (-0.643, -0.178): improved; the
    # per-stratum guard at margin 0.10 is GO on a (upper -0.207) and INSUFFICIENT on b
    # (upper zero_difference_upper(5) = 0.522).
    assert paired_road_bounds([-0.5] * 40 + [0.0] * 5, 0.05)[1] < 0
    assert zero_difference_upper(5, 0.05) > 0.10
    guarded = claim(0.10, primary="pooled", pooled_label="pooled",
                    guards=[{"name": "same_endpoint_noninferiority", "margin": 0.10}],
                    gating=gating)
    report = run_sensitivity(design({"a": roads(40), "b": roads(5)}, superiority=guarded),
                             [continuous(-0.5, effects={"b": 0.0})], 2, 0)
    assert superiority_counts(report["scenarios"][0]) == {outcome: 2}


def test_sequential_superiority_is_decided_at_the_stopping_look():
    # all-zero differences: GO only at look 2 (alpha .025 each; 42 roads needed), then the
    # stratum's look-2 endpoints rule out a 0.10 improvement (zero_difference_upper(80, .025))
    assert zero_difference_upper(80, 0.025) < 0.10
    report = run_sensitivity(
        design({"u": roads(80)}, looks=sequential({"u": [40, 40]}), superiority=claim(0.10)),
        [continuous()], 3, 5)
    scenario = report["scenarios"][0]
    assert scenario["stopped_at_look"] == {"1": 0, "2": 3}
    assert superiority_counts(scenario) == {sup.RULED_OUT: 3}
    assert scenario["superiority"]["decided_at_look"] == {"1": 0, "2": 3}
    assert report["superiority"]["inference_path"] == (
        "gate.sequential.SequentialGate -> gate.superiority.decide")
    # a certain improvement closes at look 1 and is supported there
    better = run_sensitivity(
        design({"u": roads(80)}, looks=sequential({"u": [42, 38]}), superiority=claim(0.10)),
        [continuous(-0.5)], 2, 5)
    assert superiority_counts(better["scenarios"][0]) == {sup.SUPPORTED: 2}
    assert better["scenarios"][0]["superiority"]["decided_at_look"] == {"1": 2, "2": 0}


def test_design_schema_v1_reads_as_superiority_unregistered_never_defaulted():
    """Review F4-07: the superiority slot arrived with design schema 2. A v1 design (no slot)
    stays readable and its superiority section is blocked; v1 cannot carry the slot, v2 must,
    and other versions fail closed."""
    v2 = design({"u": roads(36)})
    v1 = {key: value for key, value in v2.items() if key != "superiority"}
    v1["schema_version"] = 1
    old, new = run_sensitivity(v1, [continuous()], 2, 1), run_sensitivity(v2, [continuous()], 2, 1)
    assert old["superiority"]["status"] == "blocked"
    assert [i["code"] for i in old["superiority"]["issues"]] == ["superiority_unregistered"]
    assert "schema_version 1" in old["superiority"]["issues"][0]["message"]
    assert old["scenarios"][0]["superiority"] is None
    assert old["scenarios"][0]["outcomes"] == new["scenarios"][0]["outcomes"]
    with pytest.raises(ValueError, match="superiority"):
        run_sensitivity({**v1, "superiority": None}, [continuous()], 1, 0)
    with pytest.raises(ValueError, match="superiority"):
        run_sensitivity({k: v for k, v in v2.items() if k != "superiority"}, [continuous()],
                        1, 0)
    for version in (0, 3, True, "2"):
        with pytest.raises(ValueError, match="schema_version"):
            run_sensitivity({**v2, "schema_version": version}, [continuous()], 1, 0)


def test_unregistered_superiority_is_blocked_not_defaulted():
    report = run_sensitivity(design({"u": roads(36)}), [continuous()], 2, 1)
    assert report["status"] == "passed"  # the non-inferiority simulation still ran
    assert report["superiority"]["status"] == "blocked"
    assert [i["code"] for i in report["superiority"]["issues"]] == ["superiority_unregistered"]
    assert report["scenarios"][0]["superiority"] is None


@pytest.mark.parametrize(("spec", "looks", "match"), [
    ({"delta_min": 0.1}, None, "primary"),
    ({**claim(0.1), "extra": 1}, None, "unknown"),
    (claim(0.0), None, "delta_min"),
    (claim(1.5), None, "delta_min"),
    (claim(True), None, "delta_min"),
    (claim(0.1, primary="best_stratum"), None, "primary"),
    (claim(0.1, primary="pooled"), None, "pooled_label"),
    (claim(0.1, pooled_label="pooled"), None, "pooled_label"),
    (claim(0.1, primary="pooled", pooled_label="u"), None, "pooled_label"),
    (claim(0.1, gating=["u"]), None, "gating_strata"),
    (claim(0.1, guards=[{"name": "g", "margin": 0.1}]), None, "gating_strata"),
    (claim(0.1, guards=[{"name": "g", "margin": 0.1}], gating=["zzz"]), None, "gating_strata"),
    (claim(0.1, guards=[{"name": "g", "margin": 1.0}], gating=["u"]), None, "margin"),
    (claim(0.1, guards=[{"name": "g"}], gating=["u"]), None, "margin"),
    (claim(0.1, guards=[{"name": "g", "margin": 0.1}] * 2, gating=["u"]), None, "unique"),
    (claim(0.1, guards=[{"name": "", "margin": 0.1}], gating=["u"]), None, "name"),
    (claim(0.1, primary="pooled", pooled_label="p"), "looks", "sequential"),
    (claim(0.1, guards=[{"name": "g", "margin": 0.1}], gating=["u"]), "looks", "sequential"),
    ("yes", None, "superiority"),
])
def test_superiority_design_is_validated_fail_closed(spec, looks, match):
    registered_looks = sequential({"u": [20, 20], "v": [20, 20]}) if looks else None
    value = design({"u": roads(40), "v": roads(40)}, looks=registered_looks, superiority=spec)
    with pytest.raises(ValueError, match=match):
        run_sensitivity(value, [continuous()], 1, 0)


def test_superiority_report_is_deterministic_and_describes_its_claims():
    args = (design({"u": roads(20)}, superiority=claim(0.10)), [continuous(-0.2, noise=0.3)],
            30, 4)
    first, second = run_sensitivity(*args), run_sensitivity(*args)
    assert first == second
    section = first["scenarios"][0]["superiority"]
    assert list(section["outcomes"]) == list(sup.OUTCOMES)
    assert sum(value["count"] for value in section["outcomes"].values()) == 30
    for name in ("improvement_power", "falsifier_power"):
        value = section[name]
        assert set(value) == {"count", "rate", "standard_error", "interval", "definition"}
    assert first["superiority"]["delta_min"] == 0.10
    assert "IMPROVEMENT_SUPPORTED_BELOW_MIN" in section["improvement_power"]["definition"]
