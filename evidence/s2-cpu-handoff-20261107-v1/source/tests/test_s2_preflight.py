"""Offline S2 admission tests. All approval records below are fictional fixtures."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from gate.s2_preflight import validate_s2_preflight


def attestation(**extra):
    return {
        "status": "recorded",
        "actor": "Huy",
        "source_ref": "fixture:manual-record",
        "recorded_at": "2026-11-02",
        **extra,
    }


def frozen(value):
    return {"value": value, "owner": "Huy", "source_ref": "fixture:owner", "frozen": True}


def manifest(phase="harmonizer_pilot"):
    full = phase == "full_sweep"
    return {
        "schema_version": 2,
        "stage": "S2",
        "phase": phase,
        "status": "preregistered",
        "registration": {"owner": "Huy", "source_ref": "fixture:preregistration", "frozen": True},
        "prerequisites": {
            "S1_evidence": "fixture:S1-qualified-evidence",
            "budget_decision_P005": attestation(
                tier="up_to_2500",
                project_cap_usd=2500,
                stage_cap_usd=500,
                pilot_cap_usd=30,
                approved_compute_gpu_hours_cap=200 if full else 8,
                approved_rate_per_gpu_hour_cap=2.5,
                approved_by="Huy",
            ),
            "full_sweep_release": attestation(
                approved_by="Huy",
                pilot_evidence_ref="fixture:completed-pilot",
                forecast_ref="fixture:measured-forecast",
                remaining_stage_cap_usd=450,
            )
            if full
            else {"status": "pending"},
            "challenge_closed_verified": attestation(),
            "latest_upstream_release_checked": attestation(release_merged=False),
            "license_acceptance_by_owner": attestation(approved_by="Huy"),
            "host_compatibility_verified": attestation(
                docker=True, nvidia_runtime=True, full_host_access=True
            ),
            "public_numbers_KCL": {"status": "pending"},
            "derived_weights_labels_KCL2": {"status": "pending"},
        },
        "planned": {
            "source": {
                "git_hash": "a" * 40,
                "config_sha256": "b" * 64,
                "lockfile_sha256": "c" * 64,
            },
            "sampling": {
                "scene_list": [f"scene-{i}" for i in range(441 if full else 5)],
                "road_list": [f"road-{i}" for i in range(441 if full else 5)],
                "replicates": 3 if full else 1,
                "seed_mode": "unseeded",
                "force_determinism": False,
                "target_population": "fixture roads",
                "estimand": "equal road",
                "road_independence_justification": "fixture only",
            },
            "simulator": {
                "branch": "e2e_challenge",
                "alpasim_sha": "d" * 40,
                "evaluator_sha": "e" * 40,
                "scorer_file_sha": "f" * 40,
                "n_sim_steps_requested": 200,
                "harmonizer": False if full else "compare_off_on",
                "renderer_image": "example/nre@sha256:" + "1" * 64,
                "rendering_flags": {},
                "traffic_mode": "log_replay",
                "camera_request": {"count": 4},
                "decoded_size": [1080, 1900],
                "run_method": "docker_compose",
                "topology": "fixture one-node",
                "concurrency": 1,
                "replicas": 1,
                "nre_workers": 1,
                "controller_identity": "fixture controller",
                "postprocessing_identity": "fixture",
            },
            "driver": {
                "expert_graph": False,
                "max_batch_size": 1,
                "resolved_alpamayo1_5_commit": "2" * 40,
                "policy_clock_source": "monotonic",
            },
            "measurement": {
                "score_definition": "scene_score",
                "metric_direction": "higher_is_better",
                "deviation_convention": "four_m_separately",
                "interim_looks": {"mode": "fixed"},
                "experiment_prediction": frozen(
                    "Fictional fixture prediction: added time below 1 second/step."
                ),
                "margin": frozen(0.07),
                "reliability_floor": frozen(0.8),
                "noise_floor_prediction": frozen(0.02),
                "stabilization_rule": frozen("fixture"),
                "strata_definition": frozen(["urban", "highway"]),
            },
            "resources": {
                "gpu_count": 1,
                "rate_per_gpu_hour_usd": 1.5,
                "wall_hours_cap": 4,
                "storage_cap_usd": 4,
                "egress_cap_usd": 2,
                "other_cap_usd": 0,
                "total_cap_usd": 12,
                "quoted_at": "2026-11-02",
                "gpu_edition": "RTX PRO 6000 Workstation",
            },
        },
        "realized": {},
    }


def codes(result):
    return {reason["code"] for reason in result["blocking_reasons"]}


@pytest.mark.parametrize("phase", ["harmonizer_pilot", "full_sweep"])
def test_complete_fictional_fixture_is_logically_ready_never_launch_authorized(phase):
    document = manifest(phase)
    before = deepcopy(document)
    result = validate_s2_preflight(document)
    assert result["ready_for_planning"], result["blocking_reasons"]
    assert result["cloud_launch_authorized"] is False
    assert result["attestation_authenticity"] == "not_verified"
    assert result["publication"]["public_numbers_eligible_by_attestation"] is False
    assert document == before
    json.dumps(result, allow_nan=False)


def test_pilot_does_not_require_full_sweep_thresholds_or_final_harmonizer():
    document = manifest()
    for key in (
        "margin",
        "reliability_floor",
        "noise_floor_prediction",
        "stabilization_rule",
        "strata_definition",
    ):
        del document["planned"]["measurement"][key]
    assert validate_s2_preflight(document)["ready_for_planning"]


@pytest.mark.parametrize(
    "path,value,expected",
    [
        (("status",), "draft", "not_preregistered"),
        (("prerequisites", "budget_decision_P005"), None, "budget_unapproved"),
        (("prerequisites", "budget_decision_P005", "approved_by"), "agent", "budget_unapproved"),
        (
            ("prerequisites", "license_acceptance_by_owner", "approved_by"),
            "agent",
            "license_unaccepted",
        ),
        (("prerequisites", "host_compatibility_verified", "docker"), False, "host_incompatible"),
        (("planned", "source", "git_hash"), "main", "unpinned_identity"),
        (("planned", "simulator", "renderer_image"), "nre:latest", "unpinned_identity"),
        (("planned", "sampling", "force_determinism"), True, "seed_mode_inconsistent"),
        (("planned", "sampling", "replicates"), 2, "replicate_count"),
        (("planned", "simulator", "harmonizer"), "pending", "harmonizer_unresolved"),
        (("planned", "driver", "expert_graph"), True, "graph_unqualified"),
        (("planned", "measurement", "margin", "owner"), "agent", "owner_choice_unfrozen"),
        (("planned", "measurement", "margin", "value"), float("nan"), "invalid_measurement"),
        (
            ("planned", "measurement", "interim_looks"),
            {"mode": "sequential"},
            "sequential_unqualified",
        ),
        (("planned", "resources", "total_cap_usd"), 1, "cost_exceeds_cap"),
        (("planned", "resources", "gpu_count"), True, "invalid_resource"),
    ],
)
def test_incomplete_or_inconsistent_full_plan_is_blocked(path, value, expected):
    document = manifest("full_sweep")
    cursor = document
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    result = validate_s2_preflight(document)
    assert not result["ready_for_planning"]
    assert expected in codes(result)


def test_estimated_cost_is_not_budget_approval():
    document = manifest()
    document["prerequisites"]["budget_decision_P005"] = {"status": "pending", "pilot_cap_usd": 1000}
    assert "budget_unapproved" in codes(validate_s2_preflight(document))


def test_realized_outputs_cannot_replace_requested_plan():
    document = manifest()
    document["realized"] = {"n_sim_steps_actual": 200}
    del document["planned"]["simulator"]["n_sim_steps_requested"]
    assert "missing_planned_field" in codes(validate_s2_preflight(document))
    document["planned"]["simulator"]["n_sim_steps_actual"] = 200
    assert "realized_in_plan" in codes(validate_s2_preflight(document))


def test_seeded_mode_requires_port_and_evidence():
    document = manifest()
    document["planned"]["sampling"].update(seed_mode="seeded", force_determinism=True)
    assert "seed_mode_unqualified" in codes(validate_s2_preflight(document))
    document["planned"]["driver"].update(
        determinism_port_commit="3" * 40, determinism_evidence_ref="fixture:determinism"
    )
    assert validate_s2_preflight(document)["ready_for_planning"]


def test_publication_gate_does_not_change_data_action_gate():
    document = manifest()
    document["prerequisites"]["public_numbers_KCL"] = attestation(approved_by="Huy")
    result = validate_s2_preflight(document)
    assert result["publication"]["public_numbers_eligible_by_attestation"]
    document["prerequisites"]["license_acceptance_by_owner"] = None
    assert "license_unaccepted" in codes(validate_s2_preflight(document))


@pytest.mark.parametrize(
    "document",
    [
        None,
        [],
        {},
        {"schema_version": 1},
        {"schema_version": 2, "planned": "bad", "prerequisites": []},
    ],
)
def test_malformed_input_is_reported_without_exception(document):
    result = validate_s2_preflight(document)
    assert not result["ready_for_planning"]
    assert result["blocking_reasons"]


def test_repo_draft_remains_blocked():
    path = Path(__file__).parents[1] / "experiments/S2_MANIFEST_DRAFT.json"
    result = validate_s2_preflight(json.loads(path.read_text()))
    assert not result["ready_for_planning"]
    assert not result["cloud_launch_authorized"]


@pytest.mark.parametrize("value", [None, [], {}, True, "pending", float("inf"), 10**400])
@pytest.mark.parametrize(
    "section", ["source", "sampling", "simulator", "driver", "measurement", "resources"]
)
def test_malformed_planned_sections_are_total(section, value):
    document = manifest("full_sweep")
    document["planned"][section] = value
    result = validate_s2_preflight(document)
    assert not result["ready_for_planning"]


def test_cost_includes_storage_egress_and_approved_pilot_cap():
    document = manifest()
    document["planned"]["resources"].update(storage_cap_usd=40, total_cap_usd=48)
    result = validate_s2_preflight(document)
    assert result["estimated_total_cost_usd"] == 48
    assert "cost_exceeds_cap" in codes(result)


def test_graph_qualification_is_tied_to_pinned_driver_and_warns_on_transfer():
    document = manifest("full_sweep")
    driver = document["planned"]["driver"]
    driver["expert_graph"] = True
    driver["graph_qualification"] = {
        "status": "passed",
        "bitwise": True,
        "force_determinism": True,
        "negative_controls_pass": True,
        "evidence_ref": "fixture:graph-bitwise",
        "resolved_alpamayo1_5_commit": driver["resolved_alpamayo1_5_commit"],
    }
    result = validate_s2_preflight(document)
    assert result["ready_for_planning"]
    assert "graph_transfer_assumed" in {warning["code"] for warning in result["warnings"]}
    driver["graph_qualification"]["resolved_alpamayo1_5_commit"] = "4" * 40
    assert "graph_unqualified" in codes(validate_s2_preflight(document))


@pytest.mark.parametrize("phase", ["harmonizer_pilot", "full_sweep"])
@pytest.mark.parametrize("value", [None, frozen("pending"), frozen(""), frozen(1)])
def test_every_experiment_requires_a_frozen_text_prediction(phase, value):
    document = manifest(phase)
    document["planned"]["measurement"]["experiment_prediction"] = value
    result = validate_s2_preflight(document)
    assert not result["ready_for_planning"]
    assert {"owner_choice_unfrozen", "invalid_measurement"} & codes(result)


@pytest.mark.parametrize("phase", ["harmonizer_pilot", "full_sweep"])
@pytest.mark.parametrize(
    "field,value", [("owner", "agent"), ("frozen", False), ("source_ref", "pending")]
)
def test_experiment_prediction_requires_owner_and_source(phase, field, value):
    document = manifest(phase)
    document["planned"]["measurement"]["experiment_prediction"][field] = value
    assert "owner_choice_unfrozen" in codes(validate_s2_preflight(document))


@pytest.mark.parametrize(
    "compute",
    [
        {"gpu_count": 1, "rate_per_gpu_hour_usd": 2, "wall_hours_cap": 40},
        {"gpu_count": 1, "rate_per_gpu_hour_usd": 20, "wall_hours_cap": 1},
        {"gpu_count": 4, "rate_per_gpu_hour_usd": 2, "wall_hours_cap": 3},
    ],
)
def test_pilot_money_allowance_cannot_extend_compute_or_rate(compute):
    document = manifest()
    document["prerequisites"]["budget_decision_P005"]["pilot_cap_usd"] = 100
    document["planned"]["resources"].update(compute, total_cap_usd=100)
    result = validate_s2_preflight(document)
    assert not result["ready_for_planning"]
    assert "compute_exceeds_cap" in codes(result)


@pytest.mark.parametrize(
    "field", ["approved_compute_gpu_hours_cap", "approved_rate_per_gpu_hour_cap"]
)
@pytest.mark.parametrize("value", [None, 0, -1, True, float("inf")])
def test_phase_compute_caps_must_be_explicit_positive_numbers(field, value):
    document = manifest()
    document["prerequisites"]["budget_decision_P005"][field] = value
    result = validate_s2_preflight(document)
    assert "phase_limits_unapproved" in codes(result)
    assert "budget_unapproved" not in codes(result)
    assert not result["ready_for_planning"]
    assert result["cloud_launch_authorized"] is False


def test_other_cost_reserve_is_included_without_spending_it_automatically():
    document = manifest()
    document["prerequisites"]["budget_decision_P005"]["pilot_cap_usd"] = 100
    document["planned"]["resources"].update(other_cap_usd=80, total_cap_usd=92)
    result = validate_s2_preflight(document)
    assert result["ready_for_planning"]
    assert result["estimated_total_cost_usd"] == 92
    assert not result["cloud_launch_authorized"]
    document["planned"]["resources"]["total_cap_usd"] = 91
    assert "cost_exceeds_cap" in codes(validate_s2_preflight(document))


@pytest.mark.parametrize("field", ["tax_cap_usd", "setup_cap_usd", "mystery_cap_usd"])
def test_unrecognized_cost_caps_cannot_be_silently_ignored(field):
    document = manifest()
    document["planned"]["resources"][field] = 1000
    assert "unknown_cost_field" in codes(validate_s2_preflight(document))


@pytest.mark.parametrize("value", [None, -1, True, float("nan"), "pending"])
def test_other_cost_reserve_is_required_and_nonnegative(value):
    document = manifest()
    document["planned"]["resources"]["other_cap_usd"] = value
    assert "invalid_resource" in codes(validate_s2_preflight(document))


def test_finite_numeric_inputs_with_overflowing_product_are_blocked_without_exception():
    document = manifest()
    document["planned"]["resources"].update(
        gpu_count=10**200, rate_per_gpu_hour_usd=10**200, wall_hours_cap=10**200
    )
    result = validate_s2_preflight(document)
    assert not result["ready_for_planning"]
    assert "invalid_resource" in codes(result)
    assert result["estimated_total_cost_usd"] is None
    json.dumps(result, allow_nan=False)


def test_positive_compute_cost_underflow_cannot_make_zero_budget_ready():
    document = manifest()
    document["prerequisites"]["budget_decision_P005"].update(
        tier="zero", project_cap_usd=0, stage_cap_usd=0, pilot_cap_usd=0
    )
    document["planned"]["resources"].update(
        rate_per_gpu_hour_usd=5e-324,
        wall_hours_cap=5e-324,
        storage_cap_usd=0,
        egress_cap_usd=0,
        other_cap_usd=0,
        total_cap_usd=0,
    )
    result = validate_s2_preflight(document)
    assert not result["ready_for_planning"]
    assert "invalid_resource" in codes(result)


def test_exact_pilot_caps_are_permitted_without_automatic_compute_extension():
    document = manifest()
    document["prerequisites"]["budget_decision_P005"]["pilot_cap_usd"] = 100
    document["planned"]["resources"].update(
        rate_per_gpu_hour_usd=2.5,
        wall_hours_cap=8,
        storage_cap_usd=0,
        egress_cap_usd=0,
        other_cap_usd=80,
        total_cap_usd=100,
    )
    result = validate_s2_preflight(document)
    assert result["ready_for_planning"]
    assert result["estimated_total_cost_usd"] == 100


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "pending"),
        ("actor", "agent"),
        ("approved_by", "agent"),
        ("pilot_evidence_ref", "pending"),
        ("forecast_ref", "pending"),
        ("remaining_stage_cap_usd", 0),
        ("remaining_stage_cap_usd", 501),
        ("remaining_stage_cap_usd", float("inf")),
    ],
)
def test_full_sweep_requires_owner_release_after_pilot(field, value):
    document = manifest("full_sweep")
    document["prerequisites"]["full_sweep_release"][field] = value
    assert "full_sweep_unreleased" in codes(validate_s2_preflight(document))


def test_full_sweep_cannot_spend_more_than_released_remaining_stage_cap():
    document = manifest("full_sweep")
    document["prerequisites"]["full_sweep_release"]["remaining_stage_cap_usd"] = 10
    assert "cost_exceeds_cap" in codes(validate_s2_preflight(document))


def test_full_sweep_missing_release_is_blocked_but_pilot_needs_none():
    full = manifest("full_sweep")
    del full["prerequisites"]["full_sweep_release"]
    assert "full_sweep_unreleased" in codes(validate_s2_preflight(full))
    pilot = manifest()
    del pilot["prerequisites"]["full_sweep_release"]
    assert validate_s2_preflight(pilot)["ready_for_planning"]
