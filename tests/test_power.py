"""Agent-authored acceptance checks for S1 table completeness and labeling."""

import json
from math import ceil

import pytest
from scipy.stats import norm

from gate.power import power_tables
from gate.verdict import paired_road_bounds


@pytest.fixture(scope="module")
def tables():
    return power_tables()


def test_all_tables_are_strict_json_and_annotated(tables):
    json.dumps(tables, allow_nan=False)
    assert set(tables) == {
        "binary_sample_size",
        "noninferiority_margins",
        "continuous_score",
        "rate_ratio_planning",
        "replicate_tradeoff",
        "intersection_union",
        "gaussian_sequential",
        "operational_road_bounds",
        "reference_diagnostics",
    }
    for rows in tables.values():
        assert rows
        for row in rows:
            assert row["method"] and row["assumptions"]
            assert row["fixed_planning_vs_operational"]


def test_binary_table_has_known_answer_and_infeasible_joint_marginals(tables):
    rows = tables["binary_sample_size"]
    assert len(rows) == 15
    known = next(row for row in rows if row["base_rate"] == 0.08 and row["psi"] == 0.04)
    assert known["n_pairs_80_power_one_sided"] == 964
    assert known["difference_b_minus_a"] == pytest.approx(-0.016)
    impossible = next(row for row in rows if row["base_rate"] == 0.02 and row["psi"] == 0.08)
    assert impossible["status"] == "infeasible"
    assert impossible["n_pairs_80_power_one_sided"] is None
    assert "joint" in impossible["reason"]


def test_continuous_n_comes_from_unrounded_formula(tables):
    row = next(row for row in tables["continuous_score"] if row["sigma_d"] == 0.3)
    expected = ceil(((norm.isf(0.05) + norm.ppf(0.8)) * 0.3 / 0.07) ** 2)
    assert row["n_pairs_for_margin_007"] == expected
    assert round(100 * row["mde_n441"], 2) == 3.55


def test_replicate_and_iut_known_answers(tables):
    row = next(
        row
        for row in tables["replicate_tradeoff"]
        if row["m"] == 3 and row["rho"] == 0.6 and row["n"] == 441
    )
    assert row["single_replicate_scene_multiple"] == pytest.approx(1.363636363636)
    row = next(row for row in tables["intersection_union"] if row["k"] == 4)
    assert row["overall_power_at_080_per_stratum"] == pytest.approx(0.4096)


def test_gaussian_design_grid_and_effect_labels(tables):
    rows = tables["gaussian_sequential"]
    assert len(rows) == 12
    assert {(r["K"], r["alpha"], r["boundary_family"]) for r in rows} == {
        (k, a, method)
        for k in (2, 3)
        for a in (0.05, 0.025)
        for method in ("obrien_fleming", "pocock", "lan_demets_obf")
    }
    for row in rows:
        assert row["n_max_inflation"] >= 1
        assert row["expected_n_over_fixed_delta_zero"] <= row["n_max_inflation"]
        assert row["effect_convention"] == "benefit B-A; null=-Delta; equal=0"
        assert "not operational" in row["assumptions"]


def test_operational_minimum_matches_strict_verdict(tables):
    for row in tables["operational_road_bounds"]:
        n, alpha, margin = row["minimum_all_zero_roads"], row["alpha"], row["margin"]
        assert paired_road_bounds([0] * n, alpha)[1] < margin
        assert paired_road_bounds([0] * (n - 1), alpha)[1] >= margin
        assert row["fixed_planning_vs_operational"] == "operational"


def test_diagnostics_cannot_be_read_as_operational_evidence(tables):
    for row in tables["reference_diagnostics"]:
        assert row["fixed_planning_vs_operational"] == "diagnostic_only"
    bootstrap = next(r for r in tables["reference_diagnostics"] if r["method"] == "road_bootstrap")
    assert bootstrap["n_roads"] < bootstrap["n_scenes"]
    assert bootstrap["seed"] == 20261031
    assert "scene-weighted" in bootstrap["assumptions"]
    assert "different from the operational equal-road" in bootstrap["assumptions"]


def test_rate_ratio_is_labeled_independent_poisson(tables):
    for row in tables["rate_ratio_planning"]:
        assert "independent Poisson" in row["assumptions"]
        assert "no real-world" in row["assumptions"]
