"""Known answers for S4 pins, registration validation and override compilation (A8; L1/L2).

Every owner value below is a SYNTHETIC test input chosen to exercise the schema; none is a
registration, a recommendation or Huy's choice. Key citations are checked against the vendored,
hash-checked pinned sources in ``tests/fixtures/s4_upstream/`` by parsing their AST/text, never by
importing upstream.
"""

import ast
import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from teacher_rl import pins, registration
from teacher_rl.registration import compile_overrides, validate_registration

FIXTURES = Path(__file__).parent / "fixtures" / "s4_upstream"
PROVENANCE = json.loads((FIXTURES / "provenance.json").read_text())
SCENE_A = "clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb"
SCENE_B = "clipgt-01d503d4-449b-46fc-8d78-9085e70d3554"
SCENE_C = "clipgt-37113280-a033-4be4-afdd-cb199299e028"
REASON2 = "b" * 40  # synthetic revision; the real one is an admission-time input


def _source(fixture):
    """Return vendored bytes after checking sha256 and git blob against provenance."""
    for repo in PROVENANCE["repositories"].values():
        for entry in repo["files"]:
            if entry["fixture"] == fixture:
                data = (FIXTURES / fixture).read_bytes()
                assert hashlib.sha256(data).hexdigest() == entry["sha256"]
                blob = b"blob " + str(len(data)).encode() + b"\0" + data
                assert hashlib.sha1(blob).hexdigest() == entry["git_blob_sha"]
                return data
    raise AssertionError(f"{fixture} not in provenance")


GUARD = {"name": "hard_brake_guard", "s2_registration_id": "synthetic-s2-guard",
         "metric": {"name": "hard_brake_any", "kind": "binary", "higher_is_worse": True,
                    "unit": "indicator"},
         "margin": 0.1, "alpha": 0.05,
         # review F4-03: synthetic stand-in for the hash of the guard's S2 registration
         # without its run-specific candidate (teacher_rl.evaluate._s2_protocol_hash)
         "s2_protocol_hash": "d" * 64}


def _fixture_for(citation):
    repo, _, path = citation.partition(":")
    return f"{repo}/{path}.txt"


def valid():
    """A complete synthetic registration (values exercise the schema; not owner choices)."""
    return {
        "artifact_kind": "s4_run_registration",
        "schema_version": 2,
        "scope": "synthetic",
        "registration_id": "synthetic-s4-a8",
        "engineering": {
            "pins": {
                "alpagym": "972d160eed0e23d388497851504a3a233fec5879",
                "alpamayo_recipes": "1d99fc50370c96637157455da386323a44663c9d",
                "cosmos_rl": "d2a2c57c4bd6496482bc42da19a59b4432705eda",
                "training_alpasim": "46b401beaec27977cbb86826df8e39cec2d49e49",
                "alpamayo_r1": "4cda35d22bb257f0936ac272397627b9309ca211",
                "teacher_revision": "7aba8293c09993f2e125c6819df05d7fa3e873ea",
                "alpagym_uv_lock_sha256":
                    "5148f9b0f5f20b1cc3753dec78e165491c881b0217d095bba9a74d017fcd674e",
            },
            "cosmos_reason2_revision": REASON2,
            "base_experiment": "alpamayo_1_5_local_2gpu_smoke",
            "policy_model_path": "/data/ckpt/alpagym-bundle",
            "run_root": "/data/alpagym-runs",
            "config_route": "hydra_only",
        },
        "owner": {
            "group_size": 2,
            "rollouts_per_trainer_step": 2,
            "learning_rate": 0.0001,
            "lr_schedule": {"decay_type": "none", "decay_ratio": 0.0, "min_lr_factor": 0.0},
            "warmup": {"steps": 1, "start_factor": 0.0},
            "scene_set": [SCENE_A],
            "optimizer_steps_per_run": {"trainer_steps": 3, "epochs": 1, "mini_batch_rows": 1,
                                        "optimizer_iterations": 1},
            "checkpoint_retention": {"save_freq": 1, "max_keep": -1,
                                     "export_safetensors": True},
            "anchor": {"kind": "none"},
            "scene_roles": {"training": [SCENE_A], "development": [SCENE_B],
                            "confirmation": [SCENE_C]},
            "overlap_rule": {"rule_ref": "synthetic-rule-ref", "summary": "synthetic summary"},
            "evaluated_policy_path": "synthetic evaluated path",
            # Synthetic stand-ins (F4 slots); not owner choices.
            "improvement_decision_rule": {
                "rule_id": "synthetic-rule", "registration_ref": "synthetic-ref",
                "guards": [dict(GUARD)], "guard_strata": "primary_strata",
                "gating_strata": "all_guard_strata", "look": {"kind": "fixed"}},
            "smallest_improvement_of_interest": {"value": 0.03, "unit": "synthetic unit"},
            "seed_modes": {"train_seed": "unseeded", "dataloader_seed": 0,
                           "rollout_force_determinism": True,
                           "simulator": "upstream_random_unsettable"},
            "version_policy": {"on_policy": True, "allowed_outdated_steps": 0,
                               "prefetch_rollout": False},
            "reward_arm": {"arm": "A", "preset": "progress_safety"},
            "e07_pass_criterion": {"kind": "float32_tolerance", "tolerance": 1e-6},
            "parity_tolerances": {
                "log_density": {"abs": 0.0, "rel": 0.0, "justification_ref": "synthetic"},
                "action_trajectory": {"abs": 0.0, "rel": 0.0, "ulp": 0,
                                      "justification_ref": "synthetic"},
            },
            "video": {"render_video": False},
            "decode_device": "cuda_nvjpeg",
            "independent_training_runs": 3,
            "run_aggregation": {"rule": "all", "k": None, "family_alpha": 0.05},
            "prediction_ref": "experiments/2026-11-09-synthetic-s4.md",
            "response_to_alpagym_issue_10": {"response": "accept_as_shipped",
                                             "record_ref": "synthetic-issue-10-record"},
        },
    }


def codes(report, severity=None):
    return {i["code"] for i in report["issues"] if severity is None or i["severity"] == severity}


# ---------------------------------------------------------------- pinned fixture integrity


def test_vendored_sources_match_provenance_and_licences():
    assert set(PROVENANCE["repositories"]) == {
        "alpagym", "cosmos-rl", "alpamayo-recipes", "alpasim-gympin"}
    for name, repo in PROVENANCE["repositories"].items():
        assert repo["license"] == "Apache-2.0"
        lic = (FIXTURES / repo["license_file"]["fixture"]).read_bytes()
        assert hashlib.sha256(lic).hexdigest() == repo["license_file"]["sha256"]
        assert b"Apache License" in lic and b"Version 2.0, January 2004" in lic
        assert repo["commit"] in repo["license_file"]["source_url"]
        for entry in repo["files"]:
            assert entry["fixture"].startswith(f"{name}/")
            data = (FIXTURES / entry["fixture"]).read_bytes()
            assert len(data) == entry["bytes"]
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            blob = b"blob " + str(len(data)).encode() + b"\0" + data
            assert hashlib.sha1(blob).hexdigest() == entry["git_blob_sha"]
            assert repo["commit"] in entry["source_url"]


def test_vendored_commits_equal_the_engineering_pins():
    repos = PROVENANCE["repositories"]
    assert repos["alpagym"]["commit"] == pins.UPSTREAM_PINS["alpagym"]["commit"]
    assert repos["cosmos-rl"]["commit"] == pins.UPSTREAM_PINS["cosmos_rl"]["commit"]
    assert repos["alpamayo-recipes"]["commit"] == pins.UPSTREAM_PINS["alpamayo_recipes"]["commit"]
    assert repos["alpasim-gympin"]["commit"] == pins.UPSTREAM_PINS["training_alpasim"]["commit"]


# ---------------------------------------------------------------- pins


def test_pin_literals():
    assert pins.UPSTREAM_PINS["alpagym"]["commit"] == "972d160eed0e23d388497851504a3a233fec5879"
    assert pins.UPSTREAM_PINS["alpamayo_recipes"]["commit"] == (
        "1d99fc50370c96637157455da386323a44663c9d")
    assert pins.UPSTREAM_PINS["cosmos_rl"]["commit"] == "d2a2c57c4bd6496482bc42da19a59b4432705eda"
    assert pins.UPSTREAM_PINS["training_alpasim"]["commit"] == (
        "46b401beaec27977cbb86826df8e39cec2d49e49")
    assert pins.UPSTREAM_PINS["training_alpasim"]["tree_identical_commit"].startswith("10cae1c")
    assert pins.UPSTREAM_PINS["alpamayo_r1"]["commit"] == "4cda35d22bb257f0936ac272397627b9309ca211"
    assert pins.TEACHER["revision"] == "7aba8293c09993f2e125c6819df05d7fa3e873ea"
    assert pins.COSMOS_REASON2["revision"] is None
    assert pins.COSMOS_REASON2["observed_revision"]["value"].startswith("a9fae2cf")
    assert pins.ALPAGYM_LOCK["packages"] == {
        "torch": "2.8.0+cu128", "transformers": "4.57.1", "vllm": "0.11.0", "numpy": "2.2.6",
        "safetensors": "0.7.0"}


def test_preset_facts_match_vendored_yaml_text():
    for facts in pins.EXPERIMENT_PRESETS.values():
        text = _source(_fixture_for(facts["source"])).decode()
        assert f"expected_valid_steps: {facts['expected_valid_steps']}\n" in text
        assert f"    n_generation: {facts['n_generation']}\n" in text
        assert f"    train_batch_per_replica: {facts['train_batch_per_replica']}\n" in text
        assert f"      dp_shard_size: {facts['dp_shard_size']}\n" in text
        assert f"    n_sim_steps: {facts['n_sim_steps']}\n" in text
        assert f'extra_overrides: "{facts["extra_overrides"]}"\n' in text
        assert f"override /topology: {facts['topology']}\n" in text
    default = _source("alpagym/packages/host/src/alpagym_host/conf/default.yaml.txt").decode()
    assert "  - reward: distance_to_gt\n" in default
    assert "      grpo_ratio_clip_low: 0.2\n" in default
    assert "      grpo_ratio_clip_high: 0.2\n" in default


def test_validate_observed_pins_known_answer_and_mismatch():
    observed = {name: pin["commit"] for name, pin in pins.UPSTREAM_PINS.items()}
    observed.update(alpagym_uv_lock_sha256=pins.ALPAGYM_LOCK["uv_lock_sha256"],
                    teacher_revision=pins.TEACHER["revision"],
                    cosmos_reason2_revision=REASON2,
                    registered_cosmos_reason2_revision=REASON2)
    assert pins.validate_observed_pins(observed) == {"status": "passed", "issues": []}
    bad = dict(observed, training_alpasim="10cae1c2b943da72eabafad65df80f4e3c16d90d")
    assert codes(pins.validate_observed_pins(bad)) == {"pin_mismatch"}
    unregistered = dict(observed)
    del unregistered["registered_cosmos_reason2_revision"]
    assert "cosmos_reason2_revision_unregistered" in codes(
        pins.validate_observed_pins(unregistered))
    missing = dict(observed)
    del missing["alpagym"]
    assert codes(pins.validate_observed_pins(missing)) == {"pin_missing"}
    assert "unknown_field" in codes(pins.validate_observed_pins(dict(observed, extra="x")))


def test_pins_artifact_is_deterministic():
    first, second = pins.pins_artifact(), pins.pins_artifact()
    assert first == second
    assert first["artifact_kind"] == "s4_pins" and re.fullmatch(r"[0-9a-f]{64}",
                                                                 first["facts_hash"])


# ---------------------------------------------------------------- registration validation


def test_valid_synthetic_registration_passes_with_role_facts():
    report = validate_registration(valid())
    assert report["status"] == "passed", report["issues"]
    assert report["issues"] == []
    assert report["facts"]["role_scene_id_intersections"] == {
        "training&development": [], "training&confirmation": [],
        "development&confirmation": []}


@pytest.mark.parametrize("field", registration.OWNER_FIELDS)
def test_every_owner_slot_is_required(field):
    reg = valid()
    del reg["owner"][field]
    report = validate_registration(reg)
    assert report["status"] == "blocked"
    assert {(i["code"], i["path"]) for i in report["issues"]} >= {
        ("owner_slot_unregistered", f"$.owner.{field}")}


@pytest.mark.parametrize("field", registration.ENGINEERING_FIELDS)
def test_every_engineering_field_is_required(field):
    reg = valid()
    del reg["engineering"][field]
    assert ("field_missing", f"$.engineering.{field}") in {
        (i["code"], i["path"]) for i in validate_registration(reg)["issues"]}


NEGATIVE = [
    (("owner", "group_size"), 1, "group_size_below_two"),
    (("owner", "group_size"), True, "not_integer"),
    (("owner", "learning_rate"), float("nan"), "not_finite_number"),
    (("owner", "learning_rate"), 0.0, "out_of_range"),
    (("owner", "evaluated_policy_path"), "TBD", "placeholder_value"),
    (("owner", "evaluated_policy_path"), "____", "placeholder_value"),
    (("owner", "reward_arm"), {"arm": "B", "preset": "metrics"}, "reward_arm_not_approved"),
    (("owner", "reward_arm"), {"arm": "A", "preset": "distance_to_gt"},
     "reward_arm_not_approved"),
    (("owner", "decode_device"), "cpu_libjpeg", "no_upstream_key"),
    (("owner", "prediction_ref"), "experiments/../secrets.md", "malformed_prediction_ref"),
    (("owner", "scene_set"), ["clipgt-not-a-uuid"], "malformed_scene_id"),
    (("owner", "scene_set"), [SCENE_A, SCENE_A], "duplicate_scene_id"),
    (("owner", "e07_pass_criterion"), {"kind": "float32_tolerance"}, "field_missing"),
    (("owner", "e07_pass_criterion"), {"kind": "approximately"}, "unsupported_value"),
    (("owner", "anchor"), {"kind": "kl_to_reference", "kl_beta": 0.0,
                           "reference_reset_interval": 0}, "out_of_range"),
    (("owner", "lr_schedule"), {"decay_type": "step", "decay_ratio": 0.0,
                                "min_lr_factor": 0.0}, "unsupported_value"),
    (("owner", "checkpoint_retention"), {"save_freq": 1, "max_keep": 0,
                                         "export_safetensors": True}, "out_of_range"),
    # B3 re-verification: the run aggregation inputs are an owner slot (no defaults)
    (("owner", "run_aggregation"), {"rule": None, "k": None, "family_alpha": 0.05},
     "aggregation_rule_unregistered"),
    (("owner", "run_aggregation"), {"rule": "any", "k": None, "family_alpha": 0.05},
     "aggregation_rule_unregistered"),
    (("owner", "run_aggregation"), {"rule": "at_least_k", "k": None, "family_alpha": 0.05},
     "aggregation_k_invalid"),
    (("owner", "run_aggregation"), {"rule": "at_least_k", "k": 4, "family_alpha": 0.05},
     "aggregation_k_invalid"),
    (("owner", "run_aggregation"), {"rule": "at_least_k", "k": 0, "family_alpha": 0.05},
     "aggregation_k_invalid"),
    (("owner", "run_aggregation"), {"rule": "at_least_k", "k": True, "family_alpha": 0.05},
     "aggregation_k_invalid"),
    (("owner", "run_aggregation"), {"rule": "all", "k": 2, "family_alpha": 0.05},
     "aggregation_k_invalid"),
    (("owner", "run_aggregation"), {"rule": "all", "k": None, "family_alpha": None},
     "family_alpha_invalid"),
    (("owner", "run_aggregation"), {"rule": "all", "k": None, "family_alpha": 1.0},
     "family_alpha_invalid"),
    (("owner", "run_aggregation"), {"rule": "all", "k": None, "family_alpha": 0.0},
     "family_alpha_invalid"),
    (("owner", "run_aggregation"), {"rule": "all", "k": None, "family_alpha": 1},
     "family_alpha_invalid"),
    (("owner", "run_aggregation"), {"rule": "all", "k": None,
                                    "family_alpha": float("nan")}, "family_alpha_invalid"),
    (("owner", "run_aggregation"), {"rule": "all", "family_alpha": 0.05}, "field_missing"),
    (("owner", "run_aggregation"), "all", "not_object"),
    # F4: superiority guards, guard/gating strata and the look are owner slots (no defaults)
    (("owner", "improvement_decision_rule", "guards"), None, "not_guard_list"),
    (("owner", "improvement_decision_rule", "guards"), [{**GUARD, "margin": 0.0}],
     "out_of_range"),
    (("owner", "improvement_decision_rule", "guards"), [{**GUARD, "margin": 1.0}],
     "out_of_range"),
    (("owner", "improvement_decision_rule", "guards"), [{**GUARD, "alpha": 1}],
     "guard_alpha_invalid"),
    (("owner", "improvement_decision_rule", "guards"), [{**GUARD, "name": "Bad Name"}],
     "malformed_guard_name"),
    (("owner", "improvement_decision_rule", "guards"), [GUARD, GUARD], "duplicate_guard_name"),
    (("owner", "improvement_decision_rule", "guards"),
     [{k: v for k, v in GUARD.items() if k != "metric"}], "field_missing"),
    (("owner", "improvement_decision_rule", "guards"),
     [{**GUARD, "metric": {**GUARD["metric"], "kind": "ordinal"}}], "unsupported_value"),
    (("owner", "improvement_decision_rule", "guards"),
     [{**GUARD, "metric": {**GUARD["metric"], "higher_is_worse": "yes"}}], "not_boolean"),
    (("owner", "improvement_decision_rule", "guards"),
     [{**GUARD, "s2_registration_id": "TBD"}], "placeholder_value"),
    # review F4-03: the guard's S2 registration is bound by hash (no default)
    (("owner", "improvement_decision_rule", "guards"),
     [{k: v for k, v in GUARD.items() if k != "s2_protocol_hash"}], "field_missing"),
    (("owner", "improvement_decision_rule", "guards"),
     [{**GUARD, "s2_protocol_hash": "D" * 64}], "malformed_guard_protocol_hash"),
    (("owner", "improvement_decision_rule", "guards"),
     [{**GUARD, "s2_protocol_hash": "d" * 63}], "malformed_guard_protocol_hash"),
    (("owner", "improvement_decision_rule", "guards"),
     [{**GUARD, "s2_protocol_hash": None}], "malformed_guard_protocol_hash"),
    (("owner", "improvement_decision_rule", "guard_strata"), "all", "unsupported_value"),
    (("owner", "improvement_decision_rule", "guard_strata"), [], "not_strata_list"),
    (("owner", "improvement_decision_rule", "guard_strata"), ["urban", "urban"],
     "not_strata_list"),
    (("owner", "improvement_decision_rule", "gating_strata"), None, "unsupported_value"),
    (("owner", "improvement_decision_rule", "gating_strata"), [], "not_strata_list"),
    (("owner", "improvement_decision_rule", "look"), {"kind": "interim"}, "unsupported_value"),
    (("owner", "improvement_decision_rule", "look"), {"kind": "sequential"}, "field_missing"),
    (("owner", "improvement_decision_rule", "look"),
     {"kind": "sequential", "look_index": 0}, "out_of_range"),
    (("owner", "improvement_decision_rule", "look"),
     {"kind": "fixed", "look_index": 1}, "unknown_field"),
    (("owner", "improvement_decision_rule", "look"), "fixed", "unsupported_value"),
    # F4: the response to AlpaGym issue #10 (bf16 trainable masters) is an owner slot
    (("owner", "response_to_alpagym_issue_10"), {"response": "ignore", "record_ref": "x-1"},
     "unsupported_value"),
    (("owner", "response_to_alpagym_issue_10"), {"response": "accept_as_shipped"},
     "field_missing"),
    (("owner", "response_to_alpagym_issue_10"),
     {"response": "change_request", "record_ref": "____"}, "placeholder_value"),
    (("owner", "response_to_alpagym_issue_10"), "accept", "not_object"),
    (("engineering", "policy_model_path"), "relative/ckpt", "unsafe_model_path"),
    (("engineering", "policy_model_path"), "/data/../etc", "unsafe_model_path"),
    (("engineering", "policy_model_path"), "/data/ckpt'; rm", "unsafe_model_path"),
    (("engineering", "run_root"), "./tmp/alpagym-runs", "unsafe_run_root"),
    (("engineering", "run_root"), "/data/../runs", "unsafe_run_root"),
    (("engineering", "base_experiment"), "alpamayo_1_5_clrl_test_run",
     "unsupported_base_experiment"),
    (("engineering", "cosmos_reason2_revision"), "a9fae2cf", "cosmos_reason2_revision_unregistered"),
    (("engineering", "config_route"), "env_vars", "unsupported_value"),
    # v1 lacked the guard s2_protocol_hash (G3 bump): a v1 document is never upgraded
    (("schema_version",), 1, "unsupported_schema_version"),
    (("schema_version",), 3, "unsupported_schema_version"),
    (("schema_version",), 2.0, "unsupported_schema_version"),
    (("artifact_kind",), "s2_registration", "wrong_artifact_kind"),
]


@pytest.mark.parametrize(("path", "value", "code"), NEGATIVE)
def test_negative_controls_block(path, value, code):
    reg = valid()
    target = reg
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    report = validate_registration(reg)
    assert report["status"] == "blocked"
    assert code in codes(report, "blocking")


@pytest.mark.parametrize("field", registration.DECISION_RULE_FIELDS)
def test_every_decision_rule_slot_is_required(field):
    reg = valid()
    del reg["owner"]["improvement_decision_rule"][field]
    report = validate_registration(reg)
    assert report["status"] == "blocked"
    assert ("field_missing", f"$.owner.improvement_decision_rule.{field}") in {
        (i["code"], i["path"]) for i in report["issues"]}


def test_decision_rule_slot_known_answers():
    reg = valid()
    rule = reg["owner"]["improvement_decision_rule"]
    rule.update(guards=[], guard_strata="primary_strata", gating_strata="all_guard_strata",
                look={"kind": "sequential", "look_index": 2})
    assert validate_registration(reg)["status"] == "passed"  # an explicit empty guard list
    rule.update(guards=[GUARD], guard_strata=["urban", "highway"], gating_strata=["urban"])
    assert validate_registration(reg)["status"] == "passed"
    rule.update(gating_strata=["urban", "rural"])
    assert "gating_strata_not_subset" in codes(validate_registration(reg), "blocking")
    # with no guards the guard and gating strata take the inert tokens (no strata to register)
    rule.update(guards=[], guard_strata=["urban"], gating_strata="all_guard_strata")
    assert "guard_strata_without_guards" in codes(validate_registration(reg), "blocking")
    rule.update(guard_strata="primary_strata", gating_strata=["urban"])
    assert "guard_strata_without_guards" in codes(validate_registration(reg), "blocking")


def test_issue_10_response_is_recorded_and_a_change_request_blocks_compilation():
    out = compile_overrides(valid())
    assert out["status"] == "passed", out["issues"]
    entry = next(e for e in out["mapping"] if e["field"] == "response_to_alpagym_issue_10")
    assert entry["category"] == "harness_only" and entry["applied"] == "recorded"
    note = out["inherited_cosmos_defaults"]["alpagym_issue_10"]
    assert note["response"] == "accept_as_shipped"
    assert note["runtime_trainable_dtype"] == "unverified (E03 records it at load)"
    reg = valid()
    reg["owner"]["response_to_alpagym_issue_10"] = {"response": "change_request",
                                                    "record_ref": "synthetic-change-record"}
    assert validate_registration(reg)["status"] == "passed"  # a valid owner choice
    blocked = compile_overrides(reg)
    assert blocked["status"] == "blocked" and blocked["effective_config_hash"] is None
    assert "alpagym_issue_10_change_not_compilable" in codes(blocked, "blocking")


def test_pin_mismatch_and_unknown_fields_block():
    reg = valid()
    reg["engineering"]["pins"]["cosmos_rl"] = "e82b56f" + "0" * 33
    reg["owner"]["learning_rate_typo"] = 1.0
    reg["owner"]["seed_modes"]["simulator"] = "seeded_42"
    report = validate_registration(reg)
    assert {"pin_mismatch", "unknown_field", "no_upstream_key"} <= codes(report, "blocking")


def test_on_policy_lag_contradiction_and_training_role_mismatch():
    reg = valid()
    reg["owner"]["version_policy"]["allowed_outdated_steps"] = 2
    reg["owner"]["scene_roles"]["training"] = [SCENE_B]
    report = validate_registration(reg)
    assert {"on_policy_lag_contradiction", "training_role_differs_from_scene_set"} <= codes(
        report, "blocking")


def test_role_overlap_is_a_fact_and_warning_not_a_decision():
    reg = valid()
    reg["owner"]["scene_roles"]["confirmation"] = [SCENE_A, SCENE_C]
    report = validate_registration(reg)
    assert report["status"] == "passed"
    assert report["facts"]["role_scene_id_intersections"]["training&confirmation"] == [SCENE_A]
    assert codes(report, "warning") == {"role_scene_id_overlap"}


def test_fewer_than_three_runs_is_a_warning():
    reg = valid()
    reg["owner"]["independent_training_runs"] = 1
    report = validate_registration(reg)
    assert report["status"] == "passed"
    assert codes(report, "warning") == {"fewer_than_three_training_runs"}


def test_run_aggregation_rules_match_gate_superiority():
    from gate import superiority

    assert registration.RUN_AGGREGATION_RULES == superiority.RUN_RULES
    reg = valid()
    reg["owner"]["run_aggregation"] = {"rule": "at_least_k", "k": 3, "family_alpha": 0.01}
    assert validate_registration(reg)["status"] == "passed"
    reg["owner"]["independent_training_runs"] = 2  # k may not exceed the registered runs
    assert "aggregation_k_invalid" in codes(validate_registration(reg), "blocking")


def test_non_object_inputs_fail_closed():
    assert validate_registration([])["status"] == "blocked"
    reg = valid()
    reg["owner"] = "everything default"
    assert "not_object" in codes(validate_registration(reg))


# ---------------------------------------------------------------- compilation known answers

EXPECTED_OVERRIDES = [
    "experiment=alpamayo_1_5_local_2gpu_smoke",
    "reward=progress_safety",
    ("alpasim.wizard_args.extra_overrides='+cameras=4cam_1080 "
    "runtime.simulation_config.pose_reporting_interval_us=100000 eval.video.render_video=false'"),
    "cosmos.rollout.n_generation=2",
    "cosmos.rollout.prefetch_rollout=false",
    "cosmos.train.ckpt.enable_checkpoint=true",
    "cosmos.train.ckpt.export_safetensors=true",
    "cosmos.train.ckpt.max_keep=-1",
    "cosmos.train.ckpt.save_freq=1",
    "cosmos.train.max_num_steps=3",
    "cosmos.train.num_epochs=1",
    "cosmos.train.optm_decay_ratio=0.0",
    "cosmos.train.optm_decay_type='none'",
    "cosmos.train.optm_lr=0.0001",
    "cosmos.train.optm_min_lr_factor=0.0",
    "cosmos.train.optm_warmup_steps=1",
    "cosmos.train.train_batch_per_replica=2",
    "cosmos.train.train_policy.allowed_outdated_steps=0",
    "cosmos.train.train_policy.grpo_optimization_iterations=1",
    "cosmos.train.train_policy.kl_beta=0.0",
    "cosmos.train.train_policy.mini_batch=1",
    "cosmos.train.train_policy.on_policy=true",
    "cosmos.train.train_policy.reference_reset_interval=0",
    f"dataset.scene_ids=['{SCENE_A}']",
    "dataset.test_suite_id=null",
    "policy.inference.sampling.force_determinism=true",
    "policy.model.path='/data/ckpt/alpagym-bundle'",
    "run_root='/data/alpagym-runs'",
]


def test_compile_known_answer_overrides_and_counts():
    out = compile_overrides(valid())
    assert out["status"] == "passed", out["issues"]
    assert out["hydra"]["overrides"] == EXPECTED_OVERRIDES
    assert out["cosmos_toml_overrides"] == {}
    # 2 rollouts x 22 expected valid steps = 44 rows; 1 x ceil(44/1) = 44 per trainer step.
    assert out["derived"]["replay_rows_per_trainer_step"] == 44
    assert out["derived"]["max_optimizer_steps_per_trainer_step"] == 44
    assert out["derived"]["max_optimizer_steps_per_run"] == 132
    assert out["inherited_from_preset"]["expected_valid_steps"] == 22
    assert out["effective_config_hash"] == compile_overrides(valid())["effective_config_hash"]
    applied = {e["key"]: e["applied"] for e in out["mapping"]
               if e["category"] == "cosmos_toml_key"}
    assert applied == {"train.optm_warmup_start_factor": "upstream_default_matches",
                       "train.seed": "upstream_default_matches",
                       "train.train_policy.dataloader_seed": "upstream_default_matches"}


def test_smoke_shape_gives_the_plan_literal_of_22_optimizer_steps():
    reg = valid()
    reg["owner"]["rollouts_per_trainer_step"] = 1
    reg["owner"]["optimizer_steps_per_run"]["trainer_steps"] = 1
    out = compile_overrides(reg)
    assert out["derived"]["max_optimizer_steps_per_trainer_step"] == 22  # 02 S4.1 "up to 22"
    assert codes(out, "warning") == {"partial_group_trained"}


def test_optimizer_step_bound_on_runnable_shapes():
    # A8-G-01: only shapes the pinned host validator accepts. rollouts 4, mini_batch 2:
    # rows 4 x 22 = 88; ceil(88 / 2) = 44; x 2 iterations = 88; x 3 trainer steps = 264.
    reg = valid()
    reg["owner"]["rollouts_per_trainer_step"] = 4
    reg["owner"]["optimizer_steps_per_run"].update(mini_batch_rows=2, optimizer_iterations=2)
    out = compile_overrides(reg)
    assert out["status"] == "passed", out["issues"]
    assert out["derived"]["replay_rows_per_trainer_step"] == 88
    assert out["derived"]["max_optimizer_steps_per_trainer_step"] == 88
    assert out["derived"]["max_optimizer_steps_per_run"] == 264
    # rollouts 4, mini_batch 4: rows 88; 88 / 4 = 22; x 2 = 44 per trainer step.
    reg["owner"]["optimizer_steps_per_run"]["mini_batch_rows"] = 4
    assert compile_overrides(reg)["derived"]["max_optimizer_steps_per_trainer_step"] == 44


_CONFIG_VALIDATION = "alpagym/packages/host/src/alpagym_host/config_validation.py.txt"


def _pinned_geometry_accepts(train_batch, mini_batch, dp_shard_size, n_generation):
    """Run the pinned ``_validate_cosmos_grpo_batch_geometry`` AST node (stdlib-only body)."""
    from types import SimpleNamespace

    tree = ast.parse(_source(_CONFIG_VALIDATION))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_validate_cosmos_grpo_batch_geometry")
    namespace = {"__builtins__": {"ValueError": ValueError, "list": list, "str": str},
                 "CosmosRLConfig": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), _CONFIG_VALIDATION, "exec"),  # noqa: S102
         namespace)
    cosmos = SimpleNamespace(
        launch=SimpleNamespace(policy_replicas=1, rollout_replicas=1),
        train=SimpleNamespace(train_batch_per_replica=train_batch,
                              train_policy=SimpleNamespace(mini_batch=mini_batch)),
        policy=SimpleNamespace(parallelism=SimpleNamespace(dp_shard_size=dp_shard_size)),
        rollout=SimpleNamespace(n_generation=n_generation))
    try:
        namespace["_validate_cosmos_grpo_batch_geometry"](cosmos)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize(("rollouts", "mini_batch"), [
    (2, 1), (2, 2), (2, 4), (4, 2), (4, 3), (6, 3), (6, 4), (5, 5), (3, 2)])
def test_batch_geometry_agrees_with_the_pinned_host_validator(rollouts, mini_batch):
    """A8-G-01: a shape the pinned validator rejects before launch is blocked here too."""
    reg = valid()
    reg["owner"]["group_size"] = 2
    reg["owner"]["rollouts_per_trainer_step"] = rollouts
    reg["owner"]["optimizer_steps_per_run"]["mini_batch_rows"] = mini_batch
    preset = pins.EXPERIMENT_PRESETS[reg["engineering"]["base_experiment"]]
    accepted = _pinned_geometry_accepts(rollouts, mini_batch, preset["dp_shard_size"], 2)
    report = validate_registration(reg)
    assert (report["status"] == "passed") is accepted
    assert ("batch_geometry_rejected_upstream" in codes(report, "blocking")) is (not accepted)
    assert (compile_overrides(reg)["status"] == "passed") is accepted


def _leaf_paths(value, prefix=()):
    if type(value) is dict:
        for key, item in value.items():
            yield from _leaf_paths(item, (*prefix, key))
    if prefix:
        yield prefix


@pytest.mark.parametrize("bad", [[], {}, [1], {"x": []}])
def test_validation_never_raises_on_unhashable_or_odd_values(bad):
    """A8-G-02: every field replaced by a list/object returns a verdict, never an exception."""
    for path in list(_leaf_paths(valid())):
        reg = valid()
        target = reg
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = copy.deepcopy(bad)
        report = validate_registration(reg)  # must not raise
        assert report["status"] in ("passed", "blocked"), path
        compile_overrides(reg)  # must not raise either
    for path, value in ((("engineering", "base_experiment"), []),
                        (("owner", "reward_arm"), {"arm": [], "preset": "progress_safety"})):
        reg = valid()
        reg[path[0]][path[1]] = value
        assert validate_registration(reg)["status"] == "blocked"


def test_one_changed_owner_value_changes_the_effective_hash():
    base = compile_overrides(valid())["effective_config_hash"]
    reg = valid()
    reg["owner"]["learning_rate"] = 0.00001
    changed = compile_overrides(reg)
    assert changed["effective_config_hash"] != base
    assert "cosmos.train.optm_lr=1e-05" in changed["hydra"]["overrides"]


@pytest.mark.parametrize(("field", "value", "key"), [
    (("warmup", "start_factor"), 0.1, "train.optm_warmup_start_factor"),
    (("seed_modes", "train_seed"), 7, "train.seed"),
    (("seed_modes", "dataloader_seed"), 5, "train.train_policy.dataloader_seed"),
])
def test_cosmos_only_fields_need_the_h1_route(field, value, key):
    reg = valid()
    reg["owner"][field[0]][field[1]] = value
    blocked = compile_overrides(reg)
    assert blocked["status"] == "blocked"
    assert codes(blocked, "blocking") == {"requires_h1_toml_route"}
    assert blocked["effective_config_hash"] is None
    reg["engineering"]["config_route"] = "h1_resolved_toml"
    out = compile_overrides(reg)
    assert out["status"] == "passed"
    assert out["cosmos_toml_overrides"] == {key: value}


def test_kl_anchor_and_video_on_compile_to_their_keys():
    reg = valid()
    reg["owner"]["anchor"] = {"kind": "kl_to_reference", "kl_beta": 0.02,
                              "reference_reset_interval": 4}
    reg["owner"]["video"]["render_video"] = True
    overrides = compile_overrides(reg)["hydra"]["overrides"]
    assert "cosmos.train.train_policy.kl_beta=0.02" in overrides
    assert "cosmos.train.train_policy.reference_reset_interval=4" in overrides
    assert any(o.endswith("eval.video.render_video=true'") and "+cameras=4cam_1080" in o
               for o in overrides)


def test_blocked_registration_compiles_nothing():
    reg = valid()
    del reg["owner"]["parity_tolerances"]
    out = compile_overrides(reg)
    assert out["status"] == "blocked"
    assert "hydra" not in out and out["effective_config_hash"] is None


def test_harness_and_fixed_fields_never_reach_upstream():
    out = compile_overrides(valid())
    categories = {e["field"]: e["category"] for e in out["mapping"]}
    for field in ("scene_roles", "overlap_rule", "evaluated_policy_path",
                  "improvement_decision_rule", "smallest_improvement_of_interest",
                  "e07_pass_criterion", "parity_tolerances", "independent_training_runs",
                  "run_aggregation", "prediction_ref", "response_to_alpagym_issue_10"):
        assert categories[field] == "harness_only"
    assert categories["seed_modes.simulator"] == "fixed_by_upstream"
    assert categories["decode_device"] == "fixed_by_upstream"
    text = " ".join(out["hydra"]["overrides"])
    assert "synthetic" not in text and "experiments/" not in text


def test_compile_does_not_mutate_its_input():
    reg = valid()
    before = copy.deepcopy(reg)
    compile_overrides(reg)
    assert reg == before


# ---------------------------------------------------------------- key citations vs pinned AST


def _dataclasses(source):
    tree = ast.parse(source)
    classes = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            fields = {}
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields[item.target.id] = ast.unparse(item.annotation)
            bases = [ast.unparse(b) for b in node.bases]
            classes[node.name] = {"fields": fields, "bases": bases, "node": node}
    return classes


def _all_fields(classes, name):
    fields = {}
    for base in classes[name]["bases"]:
        if base in classes:
            fields.update(_all_fields(classes, base))
    fields.update(classes[name]["fields"])
    return fields


def _resolve(classes, root, dotted):
    current = root
    for part in dotted.split("."):
        fields = _all_fields(classes, current)
        assert part in fields, f"{dotted}: {part} is not a field of {current}"
        annotation = fields[part]
        names = [n for n in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", annotation) if n in classes]
        current = names[-1] if names else None
    return current


def test_every_hydra_key_resolves_in_the_pinned_alpagym_schema():
    config = _source("alpagym/packages/host/src/alpagym_host/config.py.txt").decode()
    classes = _dataclasses(config)
    out = compile_overrides(valid())
    hydra_entries = [e for e in out["mapping"] if e["category"] == "hydra_key"]
    assert hydra_entries
    for entry in hydra_entries:
        if entry["key"] in ("experiment", "reward"):
            _source(_fixture_for(entry["source"]))  # the preset/group file exists, hash-checked
            continue
        _resolve(classes, "RunConfigSchema", entry["key"])
        cited = entry["source"].split(";")[0]
        repo_path, _, anchor = cited.rpartition(":")
        assert repo_path == registration._G_CONFIG
        cls, field = anchor.split(".")
        assert field in _all_fields(classes, cls), entry
        assert entry["key"].endswith(field), entry


def test_every_cosmos_toml_key_and_default_resolve_in_pinned_cosmos_config():
    config = _source("cosmos-rl/cosmos_rl/policy/config/__init__.py.txt").decode()
    classes = _dataclasses(config)
    expected_defaults = {"train.optm_warmup_start_factor": "0.0", "train.seed": "None",
                         "train.train_policy.dataloader_seed": "0"}
    for key, (default, source) in registration.COSMOS_TOML_DEFAULTS.items():
        owner, _, field = key.rpartition(".")
        cls = _resolve(classes, "Config", owner)
        node = classes[cls]["node"]
        stmt = next(s for s in node.body if isinstance(s, ast.AnnAssign)
                    and s.target.id == field)
        call = stmt.value
        default_kw = next(k for k in call.keywords if k.arg == "default")
        assert ast.unparse(default_kw.value) == expected_defaults[key]
        assert repr(default) == expected_defaults[key]
        assert source.endswith(f"{cls}.{field}")


def test_reward_preset_and_video_key_exist_in_pinned_text():
    reward = _source("alpagym/packages/host/src/alpagym_host/conf/reward/"
                     "progress_safety.yaml.txt").decode()
    for metric, scale in (("progress", "1.0"), ("collision_any", "-10.0"),
                          ("offroad", "-5.0")):
        assert f"metric_name: {metric}\n    scale: {scale}\n" in reward
    assert "kind: distance_to_gt\n    scale: -0.01\n" in reward
    base = _source("alpasim-gympin/src/wizard/configs/base_config.yaml.txt").decode()
    eval_block = base[base.index("\neval:\n"):]
    assert "\n  video:\n    render_video: True\n" in eval_block
    policy = _source("alpagym/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/"
                     "policy/alpamayo_r1.yaml.txt").decode()
    assert "  device: cuda\n" in policy and "    force_determinism: true\n" in policy


def test_run_artifacts_writes_the_trainer_type_the_h1_route_must_keep():
    text = _source("alpagym/packages/host/src/alpagym_host/run_artifacts.py.txt").decode()
    assert '"trainer_type": "alpagym_grpo"' in text
    assert 'train_policy["mu_iterations"] = train_policy.pop("grpo_optimization_iterations")' in (
        text)


# ---------------------------------------------------------------- import discipline


@pytest.mark.parametrize("module", ["teacher_rl/pins.py", "teacher_rl/registration.py",
                                    "teacher_rl/records.py"])
def test_modules_do_not_import_torch_or_upstream_at_import(module):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / module).read_text())
    banned = {"torch", "numpy", "safetensors", "alpagym_runtime", "alpagym_host", "cosmos_rl"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            continue
        assert not names & banned, (module, names)


def test_s4_setup_draft_is_inert_and_consistent_with_the_pins():
    from teacher_rl.capture.entrypoint import validate_capture_config

    root = Path(__file__).resolve().parents[1]
    draft = json.loads((root / "experiments" / "S4_SETUP_DRAFT.json").read_text())
    assert draft["status"] == "draft"
    assert draft["cloud_launch_authorized"] is False
    assert draft["scientific_admission"] is False
    for name, pin in pins.UPSTREAM_PINS.items():
        assert draft["pins"][name]["commit"] == pin["commit"]
    assert draft["alpagym_lock"]["uv_lock_sha256"] == pins.ALPAGYM_LOCK["uv_lock_sha256"]
    assets = {a["role"]: a for a in draft["assets"]}
    assert assets["teacher"]["revision"] == pins.TEACHER["revision"]
    assert assets["vlm_config_and_processor"]["revision"] is None
    assert assets["vlm_config_and_processor"]["required"] is True
    assert all(a["authorization_ref"] is None for a in draft["assets"] if "authorization_ref" in a)
    assert draft["host_requirements"]["gpu_count"] == 2
    suggestion = draft["capture"]["capture_config_suggestion"]
    assert suggestion["label"].startswith("agent technical default (D-018)")
    assert validate_capture_config(suggestion["config"]) == suggestion["config"]
    blank = " ".join(draft["registration"]["owner_inputs_blank"])
    for field in registration.OWNER_FIELDS:
        assert field in blank
    # F4: the superiority sub-slots are listed blank, never filled in the draft
    for field in (*registration.DECISION_RULE_FIELDS, *registration.GUARD_FIELDS):
        assert field in blank
