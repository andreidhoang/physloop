"""Known answers and negative controls for teacher_rl.ops (B3 item 1; lane-r3 sections 2-3).

Literal argv below is transcribed from lane-r3 section 3 steps 1-5 and section 2, not from
the module. No test runs a process: plans are compiled and inspected only.
"""

import copy
import inspect

import pytest

from gate import s2_bootstrap
from scripts.rehearse_s4 import (
    REASON2,
    RENDERER,
    synthetic_bootstrap_record,
    synthetic_inventory,
    synthetic_rental_record,
)
from teacher_rl import ops

ALPAGYM = "972d160eed0e23d388497851504a3a233fec5879"
ALPASIM = "46b401beaec27977cbb86826df8e39cec2d49e49"
TREE = "9b838542a6365bd34dc48b9c6eb795a94162fb8d"
TEACHER = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
LOCK = "5148f9b0f5f20b1cc3753dec78e165491c881b0217d095bba9a74d017fcd674e"
CO = "/data/src/alpagym"
CACHE = "/data/cache/alpagym-alpasim"
# sha256("https://github.com/NVlabs/alpasim.git@" + ALPASIM)[:12], computed with
# `printf %s ... | shasum -a 256` (AlpaGym alpasim_dependency.py:37-41; repo_url from the
# AlpaGym pyproject.toml alpasim-grpc git source at 972d160)
SIM = f"{CACHE}/62400020a95f"
EXPECTED_IDS = [
    "H0_directories", "H1_uv_version", "H2_compose_version", "H3_gpu_inventory",
    "H4_redis_version", "H5_git_lfs_version", "H6_disk_free",
    "G1_clone", "G2_checkout", "G3_status_before", "G4_lock_before", "G5_uv_version",
    "G6_uv_sync_frozen", "G7_lock_after", "G8_status_after", "G9_pip_freeze",
    "G10_check_env",
    "S1_cache_dir", "S2_cache_fresh",
    "I1_ngc_login", "I2_renderer_pull", "I3_renderer_inspect", "I4_ngc_logout",
    "M1_teacher_download", "M2_cosmos_reason2_cache",
    "C1_convert_release", "N1_nccl_p2p", "R1_alpasim_head", "R2_alpasim_tree",
    "R3_alpasim_venv_freeze",
]


def plan(inventory=None, record=None, rental=None, phases=ops.PHASES):
    return ops.build_s4_bootstrap_plan(inventory or synthetic_inventory(),
                                       record or synthetic_bootstrap_record(),
                                       rental or synthetic_rental_record(),
                                       phases=list(phases))


def stage(compiled, stage_id):
    return next(s for s in compiled["stages"] if s["id"] == stage_id)


def codes(compiled):
    return {i["code"] for i in compiled["issues"]}


def test_reused_s2_compiler_parts_keep_their_signatures():
    # Drift guard: gate/s2_bootstrap.py is edited by other rounds; a renamed or re-shaped
    # private part must break here loudly, not silently change the S4 plan.
    names = ("_Builder", "_file_rule", "_hf_cache_path", "_inspect_probe", "_issue",
             "_min_version", "_plan_content_issues", "_probe", "_seal", "_stdout", "UNITS",
             "TIMEOUTS", "ENV_ALLOWLIST", "RunResult")
    for name in names:
        assert hasattr(s2_bootstrap, name), name
    assert list(inspect.signature(s2_bootstrap._Builder).parameters) == [
        "inventory", "record", "unknown_components"]
    assert list(inspect.signature(s2_bootstrap._Builder.add).parameters)[:6] == [
        "self", "stage_id", "phase", "purpose", "anchor", "argv"]
    assert set(s2_bootstrap._STAGE_FIELDS) == set(stage(plan(), "G6_uv_sync_frozen"))


def test_inventory_fixture_passes_and_demand_never_counts_unknown_as_zero():
    report = ops.validate_s4_inventory(synthetic_inventory())
    assert report["status"] == "passed", report["issues"]
    assert report["disk_demand"] == {
        # tokenizer.json: size 700 known, sha256 unverified -> its bytes count
        "known_bytes_by_role": {"cosmos_reason2": 500 + 700, "teacher": 1000 + 2000},
        "known_bytes": 4200,
        "unknown_components": ["asset:teacher:model-00002-of-00002.safetensors",
                               "image:renderer"]}


def test_full_plan_stage_order_and_gpu_count():
    compiled = plan()
    assert compiled["compiled"] is True, compiled["issues"]
    assert compiled["artifact_kind"] == "s4_bootstrap_plan"
    assert [s["id"] for s in compiled["stages"]] == EXPECTED_IDS
    assert compiled["gpu_count"] == 2
    assert compiled["execution_status"] == "qualify_on_host"
    assert compiled["secrets_required"] == ["hf_token", "ngc_api_key"]
    assert ops.plan_content_issues(compiled) == []
    disk = stage(compiled, "H6_disk_free")["expected_outputs"][0]
    # review B3-15: the S2 285 GB filesystem floor on the model and Docker filesystems, and
    # every S4-specific consumer listed as unknown (never zero)
    assert disk["requirements"] == [
        {"root": "teacher_snapshot", "path": "/data/models/alpamayo-1.5-10b", "bytes": 3000,
         "filesystem_floor_bytes": 285_000_000_000},
        {"root": "hf_home", "path": "/data/hf", "bytes": 1200, "filesystem_floor_bytes": 0},
        {"root": "docker_data_root", "path": "/var/lib/docker", "bytes": 0,
         "filesystem_floor_bytes": 285_000_000_000}]
    assert disk["unknown_components"] == [
        "alpagym_venv_and_uv_cache", "alpasim_cached_checkout_and_venv",
        "asset:teacher:model-00002-of-00002.safetensors", "checkpoint_exports_and_resume_bundles",
        "converted_bundle_files", "image:renderer", "locally_built_alpasim_images",
        "run_directory_logs_and_capture_streams"]
    assert stage(compiled, "H3_gpu_inventory")["expected_outputs"][0]["pattern"] == (
        r"\A(?:\d+, [^\n]+\n){2}\Z")


def test_literal_argv_from_lane_r3():
    compiled = plan()
    assert stage(compiled, "G1_clone")["argv"] == [
        "git", "clone", "https://github.com/NVlabs/alpagym", CO]
    assert stage(compiled, "G2_checkout")["argv"] == ["git", "-C", CO, "checkout", "--detach",
                                                      ALPAGYM]
    assert stage(compiled, "G6_uv_sync_frozen")["argv"] == ["uv", "sync", "--frozen",
                                                            "--all-packages"]
    assert stage(compiled, "G6_uv_sync_frozen")["cwd"] == CO
    assert stage(compiled, "S1_cache_dir")["argv"] == ["install", "-d", CACHE]
    assert stage(compiled, "S2_cache_fresh")["argv"] == ["find", CACHE, "-mindepth", "1",
                                                         "-maxdepth", "1"]
    assert stage(compiled, "S2_cache_fresh")["expected_outputs"][0]["pattern"] == r"\A\Z"
    assert stage(compiled, "R1_alpasim_head")["argv"] == ["git", "-C", SIM, "rev-parse", "HEAD"]
    assert stage(compiled, "R1_alpasim_head")["expected_outputs"][0]["pattern"] == (
        rf"\A{ALPASIM}\n\Z")
    assert stage(compiled, "R2_alpasim_tree")["argv"] == ["git", "-C", SIM, "rev-parse",
                                                          "HEAD^{tree}"]
    assert stage(compiled, "R2_alpasim_tree")["expected_outputs"][0]["pattern"] == (
        rf"\A{TREE}\n\Z")
    assert stage(compiled, "I2_renderer_pull")["argv"] == ["docker", "pull", RENDERER]
    assert stage(compiled, "M1_teacher_download")["argv"] == [
        "uv", "run", "--no-sync", "hf", "download", "nvidia/Alpamayo-1.5-10B", "config.json",
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "--revision",
        TEACHER, "--local-dir", "/data/models/alpamayo-1.5-10b"]
    assert stage(compiled, "M2_cosmos_reason2_cache")["argv"] == [
        "uv", "run", "--no-sync", "hf", "download", "nvidia/Cosmos-Reason2-8B", "config.json",
        "tokenizer.json", "--revision", REASON2]
    assert stage(compiled, "C1_convert_release")["argv"] == [
        "uv", "run", "--no-sync", "python",
        f"{CO}/packages/policies/alpamayo_r1/scripts/convert_release_to_alpagym_checkpoint.py",
        "--input", "/data/models/alpamayo-1.5-10b", "--output", "/data/ckpt/alpagym-bundle",
        "--vlm-name-or-path", "nvidia/Cosmos-Reason2-8B", "--overwrite"]
    assert stage(compiled, "C1_convert_release")["env_set"]["HF_HUB_OFFLINE"] == "1"
    assert stage(compiled, "N1_nccl_p2p")["argv"] == [
        "uv", "run", "--no-sync", "torchrun", "--nproc-per-node=2", "install/check_nccl_p2p.py"]
    assert stage(compiled, "R3_alpasim_venv_freeze")["argv"] == [
        "uv", "pip", "freeze", "--python", f"{SIM}/.venv/bin/python"]


def test_lock_hash_is_checked_before_and_after_the_frozen_sync():
    compiled = plan()
    for stage_id in ("G4_lock_before", "G7_lock_after"):
        rule = stage(compiled, stage_id)["expected_outputs"][0]
        assert rule == {"kind": "file", "root": "alpagym_checkout", "path": f"{CO}/uv.lock",
                        "size_bytes": None, "sha256": LOCK}
    ids = [s["id"] for s in compiled["stages"]]
    assert ids.index("G4_lock_before") < ids.index("G6_uv_sync_frozen") < ids.index(
        "G7_lock_after")
    freeze = [r["pattern"] for r in stage(compiled, "G9_pip_freeze")["expected_outputs"]]
    assert r"(?m)^torch==2\.8\.0\+cu128$" in freeze
    assert r"(?m)^vllm==0\.11\.0$" in freeze


def test_gated_stages_require_terms_and_acquisition_authority():
    compiled = plan()
    assert stage(compiled, "M1_teacher_download")["requires"] == [
        "acquire:teacher", "host_setup", "terms:teacher"]
    assert stage(compiled, "M2_cosmos_reason2_cache")["requires"] == [
        "acquire:cosmos_reason2", "host_setup", "terms:cosmos_reason2"]
    assert stage(compiled, "M2_cosmos_reason2_cache")["secret_env"] == {"HF_TOKEN": "hf_token"}
    assert stage(compiled, "I1_ngc_login")["stdin_secret"] == "ngc_api_key"
    assert all("host_setup" in s["requires"] for s in compiled["stages"])
    assert compiled["authorizations"]["terms:cosmos_reason2"] == "SYNTHETIC-terms-cosmos-reason2"


def test_deterministic_and_hash_bound():
    a, b = plan(), plan()
    assert a == b
    body = {k: v for k, v in a.items() if k != "plan_sha256"}
    from gate.s2_identity import identity_hash
    assert identity_hash(body) == a["plan_sha256"]
    tampered = copy.deepcopy(a)
    tampered["stages"][0]["argv"].append("--oops")
    assert "plan_hash_mismatch" in {i["code"] for i in ops.plan_content_issues(tampered)}


def test_phase_subset_compiles_only_its_stages():
    compiled = plan(phases=["nccl", "host"])
    assert [s["id"][0] for s in compiled["stages"]] == ["H"] * 7 + ["N"]
    assert compiled["issues"] == []  # the C1 caveat appears only when C1 is compiled


def assert_refused(compiled, code):
    assert compiled["compiled"] is False
    assert compiled["stages"] == []
    assert code in codes(compiled), compiled["issues"]


def test_missing_terms_reference_refuses():
    inventory = synthetic_inventory()
    inventory["assets"][1]["authorization_ref"] = None
    assert_refused(plan(inventory=inventory), "authorization_missing")
    assert plan(inventory=inventory, phases=["host", "alpagym"])["compiled"] is True


def test_missing_acquisition_authority_refuses():
    record = synthetic_bootstrap_record()
    del record["authorizations"]["acquire:teacher"]
    assert_refused(plan(record=record), "authorization_missing")


def test_unauthorized_phase_refuses():
    record = synthetic_bootstrap_record(phases=("host",))
    assert_refused(plan(record=record), "phase_not_authorized")


@pytest.mark.parametrize("gpus", [1, 4])
def test_gpu_count_comes_from_the_rental_record_and_must_be_two(gpus):
    assert_refused(plan(rental=synthetic_rental_record(gpu_count=gpus)), "gpu_count_mismatch")


def test_unadmitted_rental_record_refuses():
    rental = synthetic_rental_record()
    rental["max_hourly_usd"] = "200"  # window exposure exceeds the recorded headroom
    assert_refused(plan(rental=rental), "rental_record_not_admitted")


def test_cosmos_revision_has_no_default():
    inventory = synthetic_inventory()
    inventory["assets"][1]["revision"] = None
    report = ops.validate_s4_inventory(inventory)
    assert report["valid"] is True and report["status"] == "blocked"
    assert_refused(plan(inventory=inventory), "revision_unresolved")
    assert plan(inventory=inventory, phases=["alpagym", "nccl"])["compiled"] is True


@pytest.mark.parametrize("mutate, code", [
    (lambda i: i["images"][0].update(reference="nvcr.io/nvidia/nre/nre-ga:26.04"),
     "mutable_reference"),
    (lambda i: i["sources"]["alpagym"].update(commit="0" * 40), "pin_mismatch"),
    (lambda i: i.update(alpagym_uv_lock_sha256="0" * 64), "pin_mismatch"),
    (lambda i: i["sources"]["training_alpasim"].update(tree="0" * 40), "pin_mismatch"),
    (lambda i: i["assets"][0].update(revision="0" * 40), "pin_mismatch"),
    (lambda i: i["assets"][0]["files"][0].update(size_bytes=0), "invalid_size"),
    (lambda i: i["assets"][0]["files"][0].update(path="../escape"), "unsafe_relative_path"),
    (lambda i: i["host_paths"].update(evidence_dir="/data/src/alpagym/evidence"),
     "host_path_overlap"),
    (lambda i: i["host_paths"].update(hf_home="relative/hf"), "invalid_host_path"),
    (lambda i: i["assets"][1].update(authorization_ref="hf_" + "a" * 34),
     "secret_like_value"),
    (lambda i: i["assets"].pop(), "invalid_inventory"),
])
def test_inventory_negative_controls_refuse(mutate, code):
    inventory = synthetic_inventory()
    mutate(inventory)
    report = ops.validate_s4_inventory(inventory)
    assert code in {i["code"] for i in report["issues"]}, report["issues"]
    assert_refused(plan(inventory=inventory), "inventory_invalid")


def test_invalid_phase_selection_and_record_refuse():
    assert_refused(plan(phases=["launch"]), "invalid_phases")
    assert_refused(plan(phases=["host", "host"]), "invalid_phases")
    record = synthetic_bootstrap_record()
    record["authorizations"]["acquire:teacher"] = "TBD"
    assert_refused(plan(record=record), "invalid_action_record")
    record = synthetic_bootstrap_record()
    record["authorizations"]["acquire:everything"] = "SYNTHETIC-x"
    assert_refused(plan(record=record), "invalid_action_record")


def test_scope_mismatch_refuses():
    rental = synthetic_rental_record()
    rental["scope"] = "actual"
    assert_refused(plan(rental=rental), "scope_mismatch")


def test_non_dict_inputs_never_raise():
    compiled = ops.build_s4_bootstrap_plan(None, None, None, phases=["host"])
    assert compiled["compiled"] is False and compiled["issues"]


def test_setup_draft_operations_section_is_inert_and_matches_the_module():
    import json
    from pathlib import Path

    draft = json.loads((Path(__file__).resolve().parents[1] / "experiments"
                        / "S4_SETUP_DRAFT.json").read_text())
    assert draft["status"] == "draft" and draft["cloud_launch_authorized"] is False
    ops_section = draft["operations"]
    assert ops_section["status"] == "draft" and ops_section["authorization"] is None
    assert ops_section["bootstrap_plan"]["phases"] == list(ops.PHASES)
    assert ops_section["bootstrap_plan"]["action_record_authorizations"] == list(
        ops.AUTHORIZATION_KEYS)
    assert ops_section["bootstrap_plan"]["executable_by"] is None
    for key in ("stall_seconds", "deadline_epoch"):
        assert ops_section["watchdog"][key] is None
    assert ops_section["forecast"]["measured_inputs"] is None
    handoff = ops_section["evaluation_handoff"]
    assert all(handoff[k] is None for k in ("run_registry_run_ids", "aggregation_rule",
                                            "aggregation_k", "family_alpha"))
    assert ops_section["launch"]["nurec_authorization_refs"] is None  # Huy's references


# --- review B3-04: R1-R3 verify the AlpaSim checkout AlpaGym actually uses ------------------


def test_record_phase_targets_alpagym_content_addressed_cache_checkout():
    assert ops.ALPAGYM_ALPASIM_REPO_URL == "https://github.com/NVlabs/alpasim.git"
    assert ops.alpasim_cached_checkout(CACHE) == SIM
    compiled = plan()
    for stage_id in ("R1_alpasim_head", "R2_alpasim_tree", "R3_alpasim_venv_freeze"):
        assert SIM in " ".join(stage(compiled, stage_id)["argv"]), stage_id
    assert "/data/src/alpasim-train" not in str(compiled)


def test_cache_dir_fixture_and_launch_env_agree():
    # equality of inventory and launch cache dirs is by construction (rehearsal), not enforced
    from teacher_rl import launch
    assert "XDG_CACHE_HOME" not in launch.ENV_ALLOWLIST
    assert synthetic_inventory()["host_paths"]["alpasim_cache_dir"] == CACHE


# --- review B3-10: gated snapshot, bundle, caches and evidence never overlap ----------------


@pytest.mark.parametrize("name, value", [
    ("converted_bundle", "/data/models/alpamayo-1.5-10b"),  # converter --overwrite in place
    ("converted_bundle", "/data/models/alpamayo-1.5-10b/bundle"),
    ("hf_home", "/data/models"),
    ("download_scratch", "/data/evidence/scratch"),
    ("evidence_dir", "/data/hf/evidence"),
    ("alpasim_cache_dir", "/data/scratch/alpasim"),
])
def test_data_paths_must_not_overlap(name, value):
    inventory = synthetic_inventory()
    inventory["host_paths"][name] = value
    report = ops.validate_s4_inventory(inventory)
    assert "host_path_overlap" in {i["code"] for i in report["issues"]}, report["issues"]
    assert_refused(plan(inventory=inventory), "inventory_invalid")


# --- review B3-17: a structurally broken plan yields issues, never a KeyError ---------------


@pytest.mark.parametrize("key", ["roots", "stages", "authorizations", "compiled"])
def test_plan_content_issues_on_a_plan_missing_a_field(key):
    broken = copy.deepcopy(plan())
    del broken[key]
    issues = ops.plan_content_issues(broken)
    assert "invalid_plan" in {i["code"] for i in issues}


def test_plan_content_issues_on_a_plan_with_an_extra_field():
    broken = copy.deepcopy(plan())
    broken["extra"] = 1
    assert "invalid_plan" in {i["code"] for i in ops.plan_content_issues(broken)}


# --- review B3-18: offline Cosmos-Reason2-8B resolution by repo name is a host question ------


def test_c1_is_l3_with_the_offline_cache_caveat():
    compiled = plan()
    assert compiled["compiled"] is True
    assert stage(compiled, "C1_convert_release")["level"] == "L3"
    caveats = [i for i in compiled["issues"]
               if i["code"] == "hf_offline_cache_resolution_unverified"]
    assert len(caveats) == 1 and caveats[0]["severity"] == "info"
    assert caveats[0]["path"] == "stages.C1_convert_release"


# --- review B3-19: the rental record must attest every licence the plan consumes ------------


def test_rental_record_must_attest_each_terms_reference_used():
    rental = synthetic_rental_record()
    rental["license_attestation_refs"] = ["SYNTHETIC-terms-renderer", "SYNTHETIC-terms-teacher"]
    compiled = plan(rental=rental)
    assert_refused(compiled, "license_attestation_missing")
    assert "rental_record.license_attestation_refs" in {i["path"] for i in compiled["issues"]}
    rental["license_attestation_refs"] = []
    assert_refused(plan(rental=rental), "license_attestation_missing")
    # phases that consume no licence compile under a record attesting none
    assert plan(rental=rental, phases=["host", "alpagym", "alpasim", "nccl"])["compiled"]


# --- A2X: S4 plans carry their registered kind and run through the one guest runner ---------


def test_plan_records_s4_kind_and_names_the_guest_runner():
    compiled = plan()
    assert compiled["compiler"] == {"inventory_kind": "s4_runtime",
                                    "function": "teacher_rl.ops.build_s4_bootstrap_plan"}
    assert compiled["executable_by"] == "scripts/s2_guest_bootstrap.py"
    assert s2_bootstrap.validate_plan(compiled) == []
    assert ops._PLAN_FIELDS == s2_bootstrap.plan_fields("s4_runtime")
    refused = ops.build_s4_bootstrap_plan(None, None, None, phases=["host"])
    assert refused["compiler"] == compiled["compiler"] and refused["executable_by"] is None


def test_s4_roots_stay_outside_the_alpagym_checkout_under_validate_plan():
    forged = copy.deepcopy(plan())
    forged["roots"]["evidence_dir"] = f"{CO}/evidence"
    forged.pop("plan_sha256")
    from gate.s2_identity import identity_hash
    forged["plan_sha256"] = identity_hash(forged)
    issues = s2_bootstrap.validate_plan(forged)
    assert {(i["code"], i["path"]) for i in issues} >= {("invalid_plan", "roots.evidence_dir")}
