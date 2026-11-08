"""Agent-authored S1 planning/reference tables, with no filesystem effects.

All grids are illustrative planning inputs, not estimates of real simulator
rates. The operational gate is deliberately priced separately from asymptotic
planning formulas. This module does not run the registered Monte Carlo study.
"""

from math import ceil, floor, log, log1p

from scipy.stats import norm

from gate import stats
from gate.verdict import paired_road_bounds


def _row(method, assumptions, classification="fixed_planning", **values):
    return {
        "method": method,
        "assumptions": assumptions,
        "fixed_planning_vs_operational": classification,
        **values,
    }


def _binary_tables():
    rows = []
    for base in (0.02, 0.04, 0.08, 0.12, 0.2):
        for psi in (0.021, 0.04, 0.08):
            candidate = base * 0.8
            effect = base * 0.2
            # The discordance must also admit the specified two marginals.
            feasible = effect <= psi <= min(base + candidate, 2 - base - candidate)
            rows.append(
                _row(
                    "Connor_one_sided_and_exact_McNemar_two_sided",
                    "independent paired binary scenes; no road or replicate design effect; "
                    "relative reduction 0.20 in candidate failure probability",
                    base_rate=base,
                    candidate_rate=candidate,
                    relative_cut=0.2,
                    psi=psi,
                    difference_b_minus_a=-effect,
                    alpha=0.05,
                    target_power=0.8,
                    status="feasible" if feasible else "infeasible",
                    reason=""
                    if feasible
                    else "discordance and marginals do not define a joint binary law",
                    n_pairs_80_power_one_sided=(
                        ceil(stats.connor_n(effect, psi, alpha=0.05, power=0.8, sided=1))
                        if feasible
                        else None
                    ),
                    exact_power_n441_two_sided=(
                        stats.exact_mcnemar_power(-effect, psi, 441, 0.05, sided=2)
                        if feasible
                        else None
                    ),
                )
            )
    return rows


def _sequential_tables():
    rows = []
    for K in (2, 3):
        for alpha in (0.05, 0.025):
            for method in ("obrien_fleming", "pocock", "lan_demets_obf"):
                design = stats.sequential_design(K, alpha, method, power=0.8)
                rows.append(
                    _row(
                        "canonical_gaussian_group_sequential",
                        "known variance, independent increments and equally spaced information; "
                        "upper efficacy stopping only, no futility; not operational gate boundaries",
                        classification="gaussian_sequential_planning",
                        boundary_family=method,
                        K=K,
                        alpha=alpha,
                        target_power=0.8,
                        boundaries=list(design["boundaries"]),
                        information_times=list(design["information_times"]),
                        effect_convention="benefit B-A; null=-Delta; equal=0",
                        n_max_inflation=design["n_max_inflation"],
                        n_max_if_fixed_n441=441 * design["n_max_inflation"],
                        expected_n_over_fixed_delta_minus_margin=design[
                            "expected_n_over_fixed_null"
                        ],
                        expected_n_over_fixed_delta_minus_half_margin=design[
                            "expected_n_over_fixed_midpoint"
                        ],
                        expected_n_over_fixed_delta_zero=design[
                            "expected_n_over_fixed_alternative"
                        ],
                        expected_n_if_fixed_n441_delta_minus_margin=441
                        * design["expected_n_over_fixed_null"],
                        expected_n_if_fixed_n441_delta_minus_half_margin=441
                        * design["expected_n_over_fixed_midpoint"],
                        expected_n_if_fixed_n441_delta_zero=441
                        * design["expected_n_over_fixed_alternative"],
                    )
                )
    return rows


def _operational_tables():
    rows = []
    for alpha in (0.05, 0.01, 0.015, 0.025):
        for margin in (0.01, 0.02, 0.05, 0.07):
            # Exact all-zero positive/negative component formula, followed by
            # the actual operation to include its numerical outward padding.
            n = max(1, floor(log(alpha / 2) / log1p(-margin)) + 1)
            while paired_road_bounds([0.0] * n, alpha)[1] >= margin:
                n += 1
            rows.append(
                _row(
                    "road_positive_negative_KL_all_zero",
                    "independent road means bounded in [-1,1]; all observed paired road "
                    "differences zero; alpha/2 per positive/negative component; strict upper<margin; "
                    "minimum is conditional on all-zero data, not 80% power",
                    classification="operational",
                    alpha=alpha,
                    margin=margin,
                    alpha_scope="fixed_total" if alpha == 0.05 else "registered_per_look_budget",
                    component_tail=alpha / 2,
                    minimum_all_zero_roads=n,
                    upper_at_minimum_roads=paired_road_bounds([0.0] * n, alpha)[1],
                    upper_at_nine_roads=paired_road_bounds([0.0] * 9, alpha)[1],
                    upper_at_220_roads=paired_road_bounds([0.0] * 220, alpha)[1],
                    upper_at_441_roads=paired_road_bounds([0.0] * 441, alpha)[1],
                    certifies_at_441_roads=paired_road_bounds([0.0] * 441, alpha)[1] < margin,
                )
            )
    return rows


def _diagnostic_tables():
    tango_lower, tango_upper = stats.tango_interval(12, 6, 441, alpha=0.05, sided=2)
    # Fixed illustrative paired scene differences; roads have unequal sizes.
    grouped = ((0, 0, 1), (-1, 0), (0,), (0, 1), (0, 0, -1), (0, 0))
    differences = [value for group in grouped for value in group]
    roads = [f"road-{i}" for i, group in enumerate(grouped) for _ in group]
    bootstrap_lower, bootstrap_upper = stats.road_bootstrap_interval(
        differences, roads, alpha=0.05, seed=20261031, draws=2000
    )
    return [
        _row(
            "Tango_score_interval",
            "IID paired binary observations; asymptotic score inversion; "
            "two-sided 95% interval; no clustering; not used to authorize GO",
            classification="diagnostic_only",
            b=12,
            c=6,
            n=441,
            estimate=(12 - 6) / 441,
            lower=tango_lower,
            upper=tango_upper,
            one_sided_p_candidate_regression=stats.exact_mcnemar_p(12, 6, sided=1),
            two_sided_p=stats.exact_mcnemar_p(12, 6, sided=2),
        ),
        _row(
            "road_bootstrap",
            "whole-road percentile resampling of pre-aggregated scene "
            "differences, scene-weighted estimand (different from the operational equal-road "
            "estimand); six synthetic roads are a diagnostic "
            "example only; no finite-sample or sequential coverage claim; not used to authorize GO",
            classification="diagnostic_only",
            n_roads=len(grouped),
            n_scenes=len(differences),
            seed=20261031,
            draws=2000,
            confidence=0.95,
            paired_scene_differences=differences,
            road_ids=roads,
            estimate=sum(differences) / len(differences),
            lower=bootstrap_lower,
            upper=bootstrap_upper,
        ),
    ]


def power_tables():
    """Return strict-JSON-compatible named row tables without writing files.

    All table rows carry method and assumptions. `fixed_planning_vs_operational`
    identifies planning formulas, operational all-zero bounds, and diagnostics.
    Infeasible binary joint laws use null result fields, never NaN or infinity.
    """
    tables = {"binary_sample_size": _binary_tables()}
    tables["noninferiority_margins"] = [
        _row(
            "normal_IID_binary_NI_planning",
            "true risk difference zero; IID paired scenes; "
            "variance approximated by discordance psi; not a Tango confidence bound",
            psi=psi,
            n=441,
            alpha=alpha,
            target_power=0.8,
            margin=stats.ni_margin(psi, 441, alpha, 0.8),
        )
        for psi in (0.021, 0.04, 0.08)
        for alpha in (0.05, 0.025)
    ]
    tables["continuous_score"] = [
        _row(
            "normal_known_variance_paired_continuous",
            "IID paired continuous scene-score "
            "differences; known sigma_d; one-sided alpha=.05; no cluster design effect",
            sigma_d=sigma,
            n=441,
            alpha=0.05,
            target_power=0.8,
            target_margin=0.07,
            mde_n441=stats.continuous_mde(sigma, 441, 0.05, 0.8),
            n_pairs_for_margin_007=ceil(((norm.isf(0.05) + norm.ppf(0.8)) * sigma / 0.07) ** 2),
        )
        for sigma in (0.1, 0.3, 0.5, 0.8)
    ]
    tables["rate_ratio_planning"] = [
        _row(
            "normal_log_rate_ratio",
            "independent Poisson event counts with equal exposure; "
            "large expected event counts; abstract simulation exposure; no real-world rate claim; "
            "ignores road pairing, overdispersion and covariance",
            base_rate=base,
            candidate_rate_ratio=0.8,
            null_ratio=1.0,
            alpha=0.05,
            target_power=0.8,
            exposure_per_arm=441,
            expected_reference_events=base * 441,
            expected_candidate_events=base * 0.8 * 441,
            power_at_exposure_441=stats.rate_ratio_power(base, 0.8, 441),
            exposure_per_arm_for_080_power=stats.rate_ratio_exposure(base, 0.8),
        )
        for base in (0.02, 0.04, 0.08, 0.12, 0.2)
    ]
    tables["replicate_tradeoff"] = [
        _row(
            "exchangeable_within_scene_replicates",
            "independent scenes; exchangeable replicates "
            "with nonnegative ICC rho; sigma=.3; no road clustering included",
            n=n,
            m=m,
            rho=rho,
            sigma=0.3,
            mean_variance=stats.replicate_variance(0.3, n, m, rho),
            design_effect=1 + (m - 1) * rho,
            single_replicate_scene_multiple=stats.equal_precision_scenes(m, rho),
            equal_precision_single_replicate_scenes=n * stats.equal_precision_scenes(m, rho),
            single_replicate_rollout_saving=1 - stats.equal_precision_scenes(m, rho) / m,
        )
        for n in (100, 441)
        for m in (1, 3, 5)
        for rho in (0.0, 0.3, 0.6, 0.9, 1.0)
    ]
    tables["intersection_union"] = [
        _row(
            "independent_stratum_power_product",
            "independent stratum decisions; disjoint IDs "
            "alone do not guarantee independence; GO requires every stratum",
            k=k,
            per_stratum_power=0.8,
            target_overall=0.8,
            overall_power_at_080_per_stratum=stats.overall_go_power([0.8] * k),
            per_stratum_power_for_080_overall=stats.per_stratum_power(0.8, k),
        )
        for k in (1, 2, 3, 4, 6)
    ]
    tables["gaussian_sequential"] = _sequential_tables()
    tables["operational_road_bounds"] = _operational_tables()
    tables["reference_diagnostics"] = _diagnostic_tables()
    return tables
