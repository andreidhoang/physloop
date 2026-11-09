"""Agent-authored known answers for the S2 bootstrap plan (CPU-09; R2 phases A-E).

Stage order, argv and gates are literal transcriptions of R2 section (1) and the pinned AlpaSim
source it cites; rule semantics are checked on tiny synthetic files. No real tool is run: every
probe goes through a fake runner, and no network, provider or credential is touched.
"""

import ast
import hashlib
import os
import re
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest
from test_s2_runtime import CUDA_REF, NRE_REF, fixture_inventory

from gate.s2_bootstrap import (
    TIMEOUTS,
    RunResult,
    build_execution_plan,
    evaluate_rule,
    validate_plan,
    version_tuple,
)
from gate.s2_identity import identity_hash

ALPASIM = "e121e59141dc391f6c295513d6dadc840a419149"
A15 = "36aeb4c5938cbc2eb2aed33b22434773da4ab639"
PATH = ("/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:"
        "/sbin:/bin")
ALL_KEYS = {
    "host_setup": "SYNTHETIC-D-0xx host setup",
    "acquire:renderer": "SYNTHETIC-D-0xx NRE pull",
    "acquire:teacher": "SYNTHETIC-D-0xx teacher download",
    "acquire:vlm_config": "SYNTHETIC-D-0xx Cosmos config download",
    "acquire:nurec_scenes": "SYNTHETIC-D-0xx five scenes",
    "acquire:harmonizer": "SYNTHETIC-D-0xx Harmonizer download",
}
CLIP1 = "10000000-0000-4000-8000-000000000001"
SCENE1 = f"sample_set/26.04_release/{CLIP1}/{CLIP1}.usdz"
EXPECTED_IDS = [
    "A0_directories", "A1_host_probe", "A2a_apt_update", "A2b_apt_install",
    "A3a_docker_version", "A3b_compose_version", "A3c_buildx_version", "A4a_nvidia_ctk_version",
    "A4b_gpu_container_check", "A5_driver_version", "A6a_uv_fetch", "A6b_uv_extract",
    "A6c_uv_install", "A6d_uv_version", "A6e_python_install", "A6f_bundle_archive",
    "A6g_bundle_unpack", "A6h_runtime_sync", "A6i_runtime_imports", "A7a_rustup_fetch",
    "A7b_rustup_install", "A9a_data_root", "A9b_disk_free",
    "B1a_clone", "B1b_checkout", "B1c_rev_parse", "B1d_status", "B2a_hood_dir", "B2b_hood_copy",
    "B3a_uv_lock", "B3b_lock_archive",
    "C1_uv_sync", "C2_compile_protos", "C3_import_check", "C4a_uv_version",
    "C4b_python_version", "C4c_rustc_version", "C4d_uv_export", "C4e_uv_pip_freeze",
    "C4f_stub_hashes",
    "D1_ngc_login", "D2a_renderer_pull", "D2b_renderer_inspect", "D2c_ngc_logout",
    "D3a_base_build", "D3b_base_inspect", "D4_base_freeze", "D5_compose_pull_flag",
    "E1_teacher_download", "E2a_vlm_config_cache", "E2b_processor_cache", "E3_scene_catalog",
    "E3a_scene_dir",
    "E3_scene1_download", "E3_scene1_place", "E3_scene2_download", "E3_scene2_place",
    "E3_scene3_download", "E3_scene3_place", "E3_scene4_download", "E3_scene4_place",
    "E3_scene5_download", "E3_scene5_place", "E4_harmonizer_download",
]


# Synthetic rental resources (a real record carries the approved rental's values; none are
# chosen by the code). Shapes follow gate.s2_host_fit.REQUIREMENT_FIELDS.
REQUIRED_RESOURCES = {
    "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "gpu_count": 1,
    "min_vram_bytes": 100_000_000_000, "driver_branch": "580", "min_cuda_capability": "12.0",
    "min_guest_ram_bytes": 96_000_000_000, "min_vcpu": 8,
    "min_docker_data_root_free_bytes": 100_000_000_000,
    "min_data_root_free_bytes": 300_000_000_000,
}


def action(phases="ABCDE", keys=None, resources=REQUIRED_RESOURCES):
    return {
        "artifact_kind": "s2_bootstrap_action_record",
        "schema_version": 1,
        "action_record_ref": "SYNTHETIC-action-record",
        "authorized_phases": list(phases),
        "authorizations": dict(ALL_KEYS if keys is None else keys),
        "required_resources": dict(resources) if resources is not None else None,
    }


def plan(inventory=None, record=None, phases="ABCDE"):
    return build_execution_plan(inventory or fixture_inventory(), record or action(),
                                phases=list(phases))


def stage(compiled, stage_id):
    return next(s for s in compiled["stages"] if s["id"] == stage_id)


def codes(compiled):
    return sorted({i["code"] for i in compiled["issues"] if i["severity"] == "error"})


# --- compiled plan known answers --------------------------------------------------------


def test_full_plan_stage_order_is_the_r2_sequence():
    compiled = plan()
    assert compiled["compiled"] is True, compiled["issues"]
    assert [s["id"] for s in compiled["stages"]] == EXPECTED_IDS
    assert compiled["phases"] == ["A", "B", "C", "D", "E"]
    assert compiled["execution_status"] == "qualify_on_host"
    assert compiled["secrets_required"] == ["hf_token", "ngc_api_key"]
    assert compiled["authorizations"] == {
        **ALL_KEYS,
        "terms:renderer": "SYNTHETIC-ngc-nre-terms",
        "terms:teacher": "SYNTHETIC-openmdw-terms",
        "terms:vlm_config": "SYNTHETIC-cosmos-access",
        "terms:nurec_scenes": "SYNTHETIC-nurec-terms",
        "terms:harmonizer": "SYNTHETIC-harmonizer-terms",
    }
    assert validate_plan(compiled) == []


def test_literal_argv_for_source_anchored_stages():
    compiled = plan()
    argv = {s["id"]: s["argv"] for s in compiled["stages"]}
    assert argv["A2b_apt_install"] == [
        "apt-get", "install", "-y", "git", "build-essential", "ca-certificates", "curl"]
    assert argv["A4b_gpu_container_check"] == [
        "docker", "run", "--rm", "--gpus", "all", CUDA_REF, "nvidia-smi"]
    assert argv["A7a_rustup_fetch"] == [
        "curl", "--proto", "=https", "--tlsv1.2", "-sSf", "-o", "/data/scratch/rustup-init.sh",
        "https://sh.rustup.rs"]
    assert argv["A7b_rustup_install"] == [
        "sh", "/data/scratch/rustup-init.sh", "-y", "--default-toolchain", "1.91.0"]
    assert argv["B1a_clone"] == ["git", "clone", "https://github.com/NVlabs/alpasim",
                                 "/data/alpasim"]
    assert stage(compiled, "B1a_clone")["env_set"]["GIT_LFS_SKIP_SMUDGE"] == "1"
    assert argv["B1b_checkout"] == ["git", "-C", "/data/alpasim", "checkout", "--detach", ALPASIM]
    assert argv["B2b_hood_copy"] == [
        "cp", "-a", "/data/alpasim/data/nre-artifacts/ego-hoods", "/data/physloop-data/nre-artifacts/"]
    assert argv["B3a_uv_lock"] == ["uv", "lock"]
    assert stage(compiled, "B3a_uv_lock")["cwd"] == "/data/alpasim"
    assert argv["C1_uv_sync"] == ["uv", "sync", "--frozen", "--extra", "wizard", "--extra",
                                  "driver"]
    assert argv["C2_compile_protos"] == ["uv", "run", "--no-sync", "compile-protos"]
    assert stage(compiled, "C2_compile_protos")["cwd"] == "/data/alpasim/src/grpc"
    assert argv["C3_import_check"] == [
        "uv", "run", "--no-sync", "python", "-c",
        ("import torch,transformers,alpamayo1_5,alpasim_driver; print(torch.__version__,"
         "torch.version.cuda,torch.cuda.get_arch_list(),transformers.__version__)")]
    assert argv["D1_ngc_login"] == ["docker", "login", "nvcr.io", "-u", "$oauthtoken",
                                    "--password-stdin"]
    assert stage(compiled, "D1_ngc_login")["stdin_secret"] == "ngc_api_key"
    assert stage(compiled, "D1_ngc_login")["cleanup_argv"] == ["docker", "logout", "nvcr.io"]
    assert argv["D2a_renderer_pull"] == ["docker", "pull", NRE_REF]
    assert argv["D2c_ngc_logout"] == ["docker", "logout", "nvcr.io"]
    assert argv["D3a_base_build"] == ["docker", "build", "-t", "alpasim-base:0.89.0",
                                      "/data/alpasim"]
    assert argv["D4_base_freeze"] == [
        "docker", "run", "--rm", "--entrypoint", "uv", "alpasim-base:0.89.0", "pip", "freeze",
        "--python", "/repo/.venv/bin/python"]
    assert argv["E1_teacher_download"][:5] == ["uv", "run", "--no-sync", "hf", "download"]
    assert argv["E1_teacher_download"][5] == "nvidia/Alpamayo-1.5-10B"
    assert argv["E1_teacher_download"][-4:] == [
        "--revision", "7aba8293c09993f2e125c6819df05d7fa3e873ea", "--local-dir",
        "/data/models/alpamayo-1.5-10b"]
    assert argv["E3_scene1_download"] == [
        "uv", "run", "--no-sync", "hf", "download", "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec",
        SCENE1, "--repo-type", "dataset", "--revision", "1" * 40, "--local-dir", "/data/scratch/nurec"]
    assert argv["E3_scene1_place"] == [
        "mv", f"/data/scratch/nurec/{SCENE1}",
        "/data/physloop-data/nre-artifacts/all-usdzs/00000000-0000-4000-8000-000000000001.usdz"]
    assert argv["E4_harmonizer_download"] == [
        "uv", "run", "--no-sync", "hf", "download", "nvidia/Harmonizer",
        "harmonizer_nontemporal.pt", "--revision", "20ca33d4612b1e98e0526b3a7ee604af5b289f58",
        "--local-dir", "/data/models/harmonizer"]
    assert argv["B3b_lock_archive"] is None and argv["A9b_disk_free"] is None


def test_teacher_download_lists_exactly_the_inventory_files():
    compiled = plan()
    argv = stage(compiled, "E1_teacher_download")["argv"]
    files = [f["path"] for f in fixture_inventory()["assets"][0]["files"]]
    assert argv[6:-4] == files
    rules = stage(compiled, "E1_teacher_download")["expected_outputs"]
    shards = [r for r in rules if r["sha256"] is not None]
    assert (shards[0]["path"], shards[0]["size_bytes"]) == (
        "/data/models/alpamayo-1.5-10b/model-00001-of-00005.safetensors", 4_928_204_944)
    assert len(shards) == 5 and len(rules) == 10


def test_lock_gate_requires_alpamayo15_pin():
    rule = next(r for r in stage(plan(), "B3a_uv_lock")["expected_outputs"]
                if r["kind"] == "uv_lock_git_rev")
    assert rule == {"kind": "uv_lock_git_rev", "root": "checkout",
                    "path": "/data/alpasim/uv.lock", "package": "alpamayo1-5",
                    "repository": "https://github.com/NVlabs/alpamayo1.5.git", "commit": A15}


def test_ego_hood_receipts_are_the_pinned_blob_bytes():
    rules = stage(plan(), "B2b_hood_copy")["expected_outputs"]
    assert rules == [
        {"kind": "file", "root": "data_root",
         "path": "/data/physloop-data/nre-artifacts/ego-hoods/hyperion_8_1/"
                 "camera_front_wide_120fov.png",
         "size_bytes": 425_624,
         "sha256": "b7c36c0d179e82cc0d820a202fb9b0017e4006aa815cc850fbd3ea1848d9a8cd"},
        {"kind": "file", "root": "data_root",
         "path": "/data/physloop-data/nre-artifacts/ego-hoods/hyperion_8/"
                 "camera_front_wide_120fov.png",
         "size_bytes": 808_775,
         "sha256": "04fa3f354940d03984a37eb755ebf845be4b4bdae39c178b8de990d0b045392d"},
    ]


def test_every_stage_is_anchored_listed_and_never_a_shell_string():
    for s in plan()["stages"]:
        assert re.match(r"R2 [A-E]\d", s["source_anchor"]), s["id"]
        assert s["level"] in ("L2", "L3")
        assert type(s["timeout_s"]) is int and s["timeout_s"] > 0
        if s["argv"] is None:
            assert s["expected_outputs"], s["id"]
            continue
        assert type(s["argv"]) is list and all(type(a) is str and a for a in s["argv"])
        joined = " ".join(s["argv"])
        if not s["reviewed_script"]:
            for token in ("|", ";", "&&", "$(", "`", ">"):
                assert token not in joined, (s["id"], token)
        assert s["env_allowlist"] == ["LANG", "LC_ALL"]
        assert s["env_set"]["PATH"] == PATH and s["env_set"]["HOME"] == "/root"


def test_gated_stages_require_terms_and_acquisition_authority():
    compiled = plan()
    requires = {s["id"]: s["requires"] for s in compiled["stages"]}
    assert requires["D1_ngc_login"] == ["acquire:renderer", "host_setup", "terms:renderer"]
    assert requires["D2a_renderer_pull"] == ["acquire:renderer", "host_setup", "terms:renderer"]
    assert requires["E1_teacher_download"] == ["acquire:teacher", "host_setup", "terms:teacher"]
    assert requires["E2a_vlm_config_cache"] == [
        "acquire:vlm_config", "host_setup", "terms:vlm_config"]
    assert requires["E2b_processor_cache"] == ["host_setup"]
    assert requires["E3_scene3_download"] == [
        "acquire:nurec_scenes", "host_setup", "terms:nurec_scenes"]
    assert requires["E4_harmonizer_download"] == [
        "acquire:harmonizer", "host_setup", "terms:harmonizer"]
    assert requires["B1a_clone"] == ["host_setup"]
    secret = {s["id"]: s["secret_env"] for s in compiled["stages"] if s["secret_env"]}
    assert secret == {name: {"HF_TOKEN": "hf_token"} for name in (
        "E2a_vlm_config_cache", "E3_scene1_download", "E3_scene2_download",
        "E3_scene3_download", "E3_scene4_download", "E3_scene5_download")}
    assert [s["id"] for s in compiled["stages"] if s["records_first_delivery"]] == [
        f"E3_scene{k}_download" for k in range(1, 6)]


def test_vlm_cache_paths_follow_hf_hub_layout_at_pinned_revision():
    rules = stage(plan(), "E2a_vlm_config_cache")["expected_outputs"]
    assert rules[0] == {
        "kind": "file", "root": "hf_home",
        "path": "/data/hf-home/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/"
                "a9fae2cf89dc64db96b12860417f0eb403013bb9/chat_template.json",
        "size_bytes": 5_499, "sha256": None}
    assert stage(plan(), "E2a_vlm_config_cache")["level"] == "L3"
    assert stage(plan(), "E2a_vlm_config_cache")["env_set"]["HF_HOME"] == "/data/hf-home"


def test_plan_is_deterministic_and_hash_bound():
    first, second = plan(), plan()
    assert first == second
    body = {k: v for k, v in first.items() if k != "plan_sha256"}
    assert first["plan_sha256"] == identity_hash(body)
    tampered = deepcopy(first)
    tampered["stages"][1]["argv"].append("--evil")
    assert "plan_hash_mismatch" in {i["code"] for i in validate_plan(tampered)}


def test_phase_subset_compiles_only_its_stages_and_keys():
    compiled = plan(record=action("BC", {"host_setup": "SYNTHETIC-ref"}), phases="CB")
    assert compiled["compiled"] is True, compiled["issues"]
    assert compiled["phases"] == ["B", "C"]
    assert [s["id"][0] for s in compiled["stages"]] == ["B"] * 8 + ["C"] * 9
    assert compiled["authorizations"] == {"host_setup": "SYNTHETIC-ref"}
    assert compiled["secrets_required"] == []


def test_registry_digest_base_is_pulled_by_digest():
    inventory = fixture_inventory()
    base_ref = "registry.example.test/alpasim-base@sha256:" + "d" * 64
    inventory["images"][1] = {
        "role": "alpasim_base", "kind": "registry_digest", "reference": base_ref,
        "platform": "linux/amd64", "size_bytes": "unverified", "compressed_bytes": "unverified",
        "authorization_ref": None}
    compiled = plan(inventory=inventory, record=action("D"), phases="D")
    assert compiled["compiled"] is True, compiled["issues"]
    ids = [s["id"] for s in compiled["stages"]]
    assert ids == ["D1_ngc_login", "D2a_renderer_pull", "D2b_renderer_inspect", "D2c_ngc_logout",
                   "D3a_base_pull", "D3b_base_inspect", "D4_base_freeze", "D5_compose_pull_flag"]
    assert stage(compiled, "D3a_base_pull")["argv"] == ["docker", "pull", base_ref]
    assert stage(compiled, "D4_base_freeze")["argv"][5] == base_ref


def test_upstream_default_harmonizer_has_no_download_stage_but_carries_terms():
    inventory = fixture_inventory()
    inventory["harmonizer_source"] = "upstream_default_download"
    keys = {k: v for k, v in ALL_KEYS.items() if k != "acquire:harmonizer"}
    compiled = plan(inventory=inventory, record=action("E", keys), phases="E")
    assert compiled["compiled"] is True, compiled["issues"]
    assert "E4_harmonizer_download" not in [s["id"] for s in compiled["stages"]]
    assert "harmonizer_downloaded_per_on_arm_unit" in {i["code"] for i in compiled["issues"]}
    # Review A2-F12: the per-unit download is licensed use; its terms record lives in the plan.
    assert compiled["authorizations"]["terms:harmonizer"] == "SYNTHETIC-harmonizer-terms"
    assert "acquire:harmonizer" not in compiled["authorizations"]
    inventory["assets"][4]["authorization_ref"] = None
    refused = plan(inventory=inventory, record=action("E", keys), phases="E")
    assert_refused(refused, "authorization_missing")


# --- refusals (negative controls) -------------------------------------------------------


def assert_refused(compiled, code):
    assert compiled["compiled"] is False
    assert compiled["stages"] == []
    assert code in codes(compiled), compiled["issues"]
    assert "plan_sha256" in compiled


def test_missing_terms_reference_refuses_the_whole_plan():
    inventory = fixture_inventory()
    inventory["assets"][0]["authorization_ref"] = None
    assert_refused(plan(inventory=inventory), "authorization_missing")
    # Phases that do not acquire the teacher still compile.
    compiled = plan(inventory=inventory, record=action("ABCD"), phases="ABCD")
    assert compiled["compiled"] is True


def test_missing_acquisition_authority_refuses():
    keys = {k: v for k, v in ALL_KEYS.items() if k != "acquire:vlm_config"}
    compiled = plan(record=action("ABCDE", keys))
    assert_refused(compiled, "authorization_missing")
    paths = {i["path"] for i in compiled["issues"] if i["code"] == "authorization_missing"}
    assert paths == {"action_record.authorizations.acquire:vlm_config"}


def test_phase_without_recorded_authority_refuses():
    assert_refused(plan(record=action("ABCD")), "phase_not_authorized")
    assert_refused(plan(record=action("ABCDE", {})), "authorization_missing")


@pytest.mark.parametrize("phases", [[], ["F"], ["A", "A"], "ABC", [1]])
def test_invalid_phase_selection_refuses(phases):
    compiled = build_execution_plan(fixture_inventory(), action(), phases=phases)
    assert_refused(compiled, "invalid_phases")


def test_secret_in_argv_refuses():
    inventory = fixture_inventory()
    record = action()
    record["authorizations"]["host_setup"] = "hf_" + "Q" * 34
    assert_refused(plan(inventory=inventory, record=record), "invalid_action_record")
    compiled = plan()
    # $oauthtoken is NVIDIA's literal username, not a secret.
    assert "$oauthtoken" in stage(compiled, "D1_ngc_login")["argv"]
    assert validate_plan(compiled) == []
    leaky = deepcopy(compiled)
    leaky["stages"][0]["argv"].append("--token=abc")
    leaky["plan_sha256"] = identity_hash({k: v for k, v in leaky.items() if k != "plan_sha256"})
    assert "secret_in_argv" in {i["code"] for i in validate_plan(leaky)}
    leaky = deepcopy(compiled)
    leaky["stages"][0]["env_set"]["HF_TOKEN"] = "hf_" + "Z" * 34
    leaky["plan_sha256"] = identity_hash({k: v for k, v in leaky.items() if k != "plan_sha256"})
    assert "secret_in_argv" in {i["code"] for i in validate_plan(leaky)}


def test_inventory_secret_path_refuses():
    inventory = fixture_inventory()
    inventory["host_paths"]["teacher_snapshot"] = "/data/hf_" + "Ab1" * 12
    assert_refused(plan(inventory=inventory), "inventory_invalid")


def test_mutable_tag_inventory_refuses():
    inventory = fixture_inventory()
    inventory["images"][0]["reference"] = "nvcr.io/nvidia/nre/nre-ga:26.04"
    assert_refused(plan(inventory=inventory), "inventory_invalid")


def test_relative_path_escape_inventory_refuses():
    inventory = fixture_inventory()
    inventory["assets"][3]["files"][0]["path"] = "../../etc/shadow.usdz"
    assert_refused(plan(inventory=inventory), "inventory_invalid")


def test_owner_inputs_needed_by_requested_phases_refuse():
    inventory = fixture_inventory()
    inventory["harmonizer_source"] = None
    assert_refused(plan(inventory=inventory), "owner_input_missing")
    inventory = fixture_inventory()
    inventory["assets"][3]["files"] = []
    assert_refused(plan(inventory=inventory), "owner_input_missing")
    inventory = fixture_inventory()
    inventory["toolchain"]["rust_toolchain"] = None
    assert_refused(plan(inventory=inventory), "toolchain_unset")
    # B and D need neither the toolchain nor scene rows.
    assert plan(inventory=inventory, record=action("BD"), phases="BD")["compiled"] is True


def test_unset_host_path_or_probe_image_refuses():
    inventory = fixture_inventory()
    inventory["host_paths"]["hf_home"] = None
    assert_refused(plan(inventory=inventory), "host_path_unset")
    inventory = fixture_inventory()
    inventory["images"][2]["reference"] = None
    assert_refused(plan(inventory=inventory), "image_reference_unset")


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(extra=1),
    lambda r: r.update(schema_version=2),
    lambda r: r.update(action_record_ref="TBD"),
    lambda r: r["authorizations"].update({"acquire:everything": "x"}),
    lambda r: r["authorizations"].update({"host_setup": "  "}),
])
def test_malformed_action_record_refuses(mutate):
    record = action()
    mutate(record)
    assert_refused(plan(record=record), "invalid_action_record")


def test_garbage_inputs_never_raise():
    for inventory, record in ((None, action()), (fixture_inventory(), None), ([], [])):
        compiled = build_execution_plan(inventory, record, phases=["A"])
        assert compiled["compiled"] is False and compiled["stages"] == []


# --- rule semantics ---------------------------------------------------------------------

LOCK_PINNED = f'''version = 1
requires-python = ">=3.11, <3.13"

[[package]]
name = "alpamayo1-5"
version = "0.1.0"
source = {{ git = "https://github.com/NVlabs/alpamayo1.5.git#{A15}" }}

[[package]]
name = "torch"
version = "2.8.0"
source = {{ registry = "https://pypi.org/simple" }}
'''
LOCK_REV_FORM = LOCK_PINNED.replace(
    "alpamayo1.5.git#", f"alpamayo1.5.git?rev={A15}#")


def lock_rule(tmp_path, text, **overrides):
    (tmp_path / "uv.lock").write_text(text)
    rule = {"kind": "uv_lock_git_rev", "root": "checkout", "path": str(tmp_path / "uv.lock"),
            "package": "alpamayo1-5", "repository": "https://github.com/NVlabs/alpamayo1.5.git",
            "commit": A15, **overrides}
    return evaluate_rule(rule, roots={"checkout": str(tmp_path)})


@pytest.mark.parametrize("text", [LOCK_PINNED, LOCK_REV_FORM])
def test_uv_lock_gate_accepts_the_pinned_commit(tmp_path, text):
    result = lock_rule(tmp_path, text)
    assert result["ok"] is True, result
    assert result["recorded"]["commit"] == A15


@pytest.mark.parametrize("text,reason", [
    (LOCK_PINNED.replace(A15, "11a0e01c" + "0" * 32), "commit_mismatch"),
    (LOCK_PINNED.replace("NVlabs/alpamayo1.5", "evil/alpamayo1.5"), "repository_mismatch"),
    (LOCK_PINNED.replace('name = "alpamayo1-5"', 'name = "alpamayo1-6"'), "package_count"),
    (LOCK_PINNED + LOCK_PINNED.split("\n\n", 1)[1], "package_count"),
    (LOCK_PINNED.replace("[[package]]", "[[package", 1), "unparseable"),
    (LOCK_PINNED.replace(f"#{A15}", ""), "commit_mismatch"),
])
def test_uv_lock_gate_rejects_mismatch(tmp_path, text, reason):
    result = lock_rule(tmp_path, text)
    assert result["ok"] is False
    assert result["reason"] == reason


def file_rule(path, root, size=None, sha=None):
    return {"kind": "file", "root": "r", "path": str(path), "size_bytes": size, "sha256": sha}


def test_file_rule_known_answer_and_records(tmp_path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"abc")
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    roots = {"r": str(tmp_path)}
    ok = evaluate_rule(file_rule(target, tmp_path, 3, digest), roots=roots)
    assert ok["ok"] is True and ok["recorded"] == {"size_bytes": 3, "sha256": digest}
    assert evaluate_rule(file_rule(target, tmp_path, None, None), roots=roots)["ok"] is True
    assert evaluate_rule(file_rule(target, tmp_path, 4, None), roots=roots)["reason"] == (
        "size_mismatch")
    assert evaluate_rule(file_rule(target, tmp_path, 3, "0" * 64), roots=roots)["reason"] == (
        "sha256_mismatch")
    assert evaluate_rule(file_rule(tmp_path / "missing", tmp_path), roots=roots)["reason"] == (
        "missing")


def test_file_rule_rejects_symlink_escape_and_lexical_escape(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret").write_bytes(b"s")
    os.symlink(outside / "secret", root / "link")
    roots = {"r": str(root)}
    assert evaluate_rule(file_rule(root / "link", root), roots=roots)["reason"] == (
        "outside_root")
    assert evaluate_rule(file_rule(outside / "secret", root), roots=roots)["reason"] == (
        "outside_root")
    # A symlink that stays inside the root (HF cache snapshots) is followed.
    (root / "blob").write_bytes(b"b")
    os.symlink(root / "blob", root / "inside")
    assert evaluate_rule(file_rule(root / "inside", root, 1), roots=roots)["ok"] is True


def test_file_regex_and_stdout_rules(tmp_path):
    (tmp_path / "HEAD").write_text(ALPASIM + "\n")
    roots = {"r": str(tmp_path)}
    rule = {"kind": "file_regex", "root": "r", "path": str(tmp_path / "HEAD"),
            "pattern": rf"\A{ALPASIM}\n\Z"}
    assert evaluate_rule(rule, roots=roots)["ok"] is True
    (tmp_path / "HEAD").write_text("ref: refs/heads/main\n")
    assert evaluate_rule(rule, roots=roots)["ok"] is False
    regex = {"kind": "stdout_regex", "pattern": r"\A\Z", "required": True}
    assert evaluate_rule(regex, roots={}, stdout=b"")["ok"] is True
    assert evaluate_rule(regex, roots={}, stdout=b"?? x\n")["ok"] is False
    optional = {"kind": "stdout_regex", "pattern": "--pull", "required": False}
    result = evaluate_rule(optional, roots={}, stdout=b"no such flag")
    assert result["ok"] is True and result["recorded"] == {"matched": False}


@pytest.mark.parametrize("stdout,minimum,ok", [
    (b"29.0.3\n", "23.0.1", True),
    (b"23.0.1\n", "23.0.1", True),
    (b"23.0.0\n", "23.0.1", False),
    (b"driver_version\n580.95.05\n", "580", True),
    (b"driver_version\n570.86.10\n", "580", False),
    (b"no version here\n", "1.0.0", False),
])
def test_min_version_rule(stdout, minimum, ok):
    rule = {"kind": "stdout_min_version", "pattern": r"(?m)^(\d+\.\d+(?:\.\d+)?)$",
            "minimum": minimum}
    assert evaluate_rule(rule, roots={}, stdout=stdout)["ok"] is ok


def test_version_tuple_pads():
    assert version_tuple("580") == (580, 0, 0)
    assert version_tuple("0.9.17") == (0, 9, 17)
    assert version_tuple("1.13.5") > version_tuple("1.13")


def make_usdz(path, metadata):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.yaml", metadata)
        archive.writestr("scene.usda", "#usda 1.0\n")


@pytest.mark.parametrize("metadata,ok", [
    ("version: 1\nuuid: 00000000-0000-4000-8000-000000000001\n", True),
    ("uuid: '00000000-0000-4000-8000-000000000001'\n", True),
    ("uuid: 00000000-0000-4000-8000-000000000002\n", False),
    ("nested:\n  uuid: 00000000-0000-4000-8000-000000000001\n", False),
    ("uuid: 00000000-0000-4000-8000-000000000001\nuuid: x\n", False),
])
def test_usdz_metadata_uuid_rule(tmp_path, metadata, ok):
    make_usdz(tmp_path / "s.usdz", metadata)
    rule = {"kind": "usdz_metadata_uuid", "root": "r", "path": str(tmp_path / "s.usdz"),
            "uuid": "00000000-0000-4000-8000-000000000001"}
    assert evaluate_rule(rule, roots={"r": str(tmp_path)})["ok"] is ok


def test_usdz_rule_rejects_non_zip(tmp_path):
    (tmp_path / "s.usdz").write_bytes(b"not a zip")
    rule = {"kind": "usdz_metadata_uuid", "root": "r", "path": str(tmp_path / "s.usdz"),
            "uuid": "00000000-0000-4000-8000-000000000001"}
    assert evaluate_rule(rule, roots={"r": str(tmp_path)})["ok"] is False


def test_archive_rule_copies_once_and_refuses_conflicts(tmp_path):
    source, archive = tmp_path / "src", tmp_path / "archive"
    source.mkdir()
    archive.mkdir()
    (source / "uv.lock").write_bytes(b"lock-1")
    rule = {"kind": "archive_file", "root": "r", "path": str(source / "uv.lock"),
            "name": "alpasim-uv.lock"}
    roots = {"r": str(source)}
    first = evaluate_rule(rule, roots=roots, archive_dir=str(archive))
    assert first["ok"] is True
    assert first["recorded"]["sha256"] == hashlib.sha256(b"lock-1").hexdigest()
    assert (archive / "alpasim-uv.lock").read_bytes() == b"lock-1"
    assert evaluate_rule(rule, roots=roots, archive_dir=str(archive))["ok"] is True
    (source / "uv.lock").write_bytes(b"lock-2")
    conflict = evaluate_rule(rule, roots=roots, archive_dir=str(archive))
    assert conflict["ok"] is False and conflict["reason"] == "archive_conflict"
    assert (archive / "alpasim-uv.lock").read_bytes() == b"lock-1"


def test_tree_rule_records_sorted_hashes(tmp_path):
    (tmp_path / "v0").mkdir()
    (tmp_path / "v0" / "b_pb2.py").write_bytes(b"b")
    (tmp_path / "v0" / "a_pb2.pyi").write_bytes(b"a")
    (tmp_path / "v0" / "__init__.py").write_bytes(b"")
    rule = {"kind": "tree", "root": "r", "path": str(tmp_path / "v0"),
            "suffixes": ["_pb2.py", "_pb2_grpc.py", ".pyi"]}
    result = evaluate_rule(rule, roots={"r": str(tmp_path)})
    assert result["ok"] is True
    assert result["recorded"]["files"] == [
        {"path": "a_pb2.pyi", "size_bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest()},
        {"path": "b_pb2.py", "size_bytes": 1, "sha256": hashlib.sha256(b"b").hexdigest()},
    ]
    (tmp_path / "empty").mkdir()
    rule["path"] = str(tmp_path / "empty")
    assert evaluate_rule(rule, roots={"r": str(tmp_path)})["ok"] is False


def space(path, size, floor=0):
    return {"root": "r", "path": str(path), "bytes": size, "filesystem_floor_bytes": floor}


def test_free_space_rule_sums_per_filesystem(tmp_path):
    roots = {"r": str(tmp_path)}
    small = {"kind": "free_space", "unknown_components": [], "requirements": [
        space(tmp_path / "not-yet" / "a", 1), space(tmp_path, 1)]}
    result = evaluate_rule(small, roots=roots)
    assert result["ok"] is True
    assert [d["required_bytes"] for d in result["recorded"]["devices"]] == [2]
    assert result["recorded"]["demand_status"] == "passed"
    huge = {"kind": "free_space", "unknown_components": [], "requirements": [
        space(tmp_path, 10 ** 18)]}
    assert evaluate_rule(huge, roots=roots)["reason"] == "insufficient_free_space"
    # Missing required fields fail closed.
    assert evaluate_rule({"kind": "free_space", "requirements": [
        {"root": "r", "path": str(tmp_path), "bytes": 1}]}, roots=roots)["ok"] is False


def test_probe_rule_uses_injected_runner_only():
    calls = []

    def runner(argv, *, cwd, env, stdin, timeout_s):
        calls.append((argv, cwd, env, stdin, timeout_s))
        return RunResult(0, b"sha256:" + b"a" * 64 + b" linux/amd64\n", b"", False)

    rule = {"kind": "probe", "argv": ["docker", "image", "inspect", "x"], "cwd": "/",
            "pattern": r"(?m)^sha256:[0-9a-f]{64} linux/amd64$", "timeout_s": 30}
    assert evaluate_rule(rule, roots={}, runner=runner, env={"PATH": "/bin"})["ok"] is True
    assert calls == [(["docker", "image", "inspect", "x"], "/", {"PATH": "/bin"}, None, 30)]
    assert evaluate_rule(rule, roots={}, runner=None)["ok"] is False


def test_unknown_rule_kind_fails_closed():
    result = evaluate_rule({"kind": "trust_me"}, roots={})
    assert result["ok"] is False and result["reason"] == "unknown_rule"


def test_runner_timeout_for_the_probe_outlasts_the_probe_worst_case():
    """Cross-file oracle: the probe's own 20 min budget plus its KILL grace (parsed from the
    script text) must end before the runner's stage timeout kills it mid-cleanup."""
    text = (Path(__file__).parents[1] / "scripts/vast_s2_host_probe.sh").read_text()

    def value(name):
        return int(re.search(rf"^{name}=(\d+)$", text, flags=re.MULTILINE).group(1))

    margin = int(re.search(r"cleanup_budget_s \+ (\d+)\)\)", text).group(1))
    kill_after = (value("longest_check_s") + value("check_kill_after_s")
                  + value("cleanup_budget_s") + margin)
    assert kill_after == 695
    assert "--kill-after=\"${outer_kill_after_s}s\" 20m" in text
    assert TIMEOUTS["probe_script"] > 20 * 60 + kill_after


# --- review A2 regressions --------------------------------------------------------------


def test_disk_gate_never_counts_unknown_sizes_as_zero(tmp_path):
    """Review A2-F5: unknowns travel with the rule and its record; the D-018 floor is checked
    on the data_root and docker_data_root filesystems; known arithmetic overflow refuses."""
    compiled = plan()
    rule = stage(compiled, "A9b_disk_free")["expected_outputs"][0]
    clips = [f"10000000-0000-4000-8000-00000000000{k}" for k in range(1, 6)]
    assert rule["unknown_components"] == sorted(
        [f"asset:nurec_scenes:sample_set/26.04_release/{c}/{c}.usdz" for c in clips]
        + ["component:alpasim_host_python_env", "component:export_temporary_copy",
           "component:per_unit_raw_outputs", "component:physloop_guest_runtime",
           "image:alpasim_base", "image:public_cuda_probe", "image:renderer"])
    floors = {r["root"]: (r["bytes"], r["filesystem_floor_bytes"]) for r in rule["requirements"]}
    # Hand sums: teacher 22,157,316,944; Cosmos 11,499,960 + Qwen 11,499,994 = 22,999,954;
    # Harmonizer 1,448,843,112; scenes and images unknown (0 known bytes) but floor-checked.
    assert floors == {
        "teacher_snapshot": (22_157_316_944, 0), "hf_home": (22_999_954, 0),
        "data_root": (0, 285_000_000_000), "docker_data_root": (0, 285_000_000_000),
        "harmonizer_dir": (1_448_843_112, 0)}
    # Evaluate: known demand fits but the unknowns are recorded as unverified, not passed.
    roots = {"r": str(tmp_path)}
    fits = {"kind": "free_space", "unknown_components": ["image:renderer"],
            "requirements": [space(tmp_path, 1)]}
    result = evaluate_rule(fits, roots=roots)
    assert result["ok"] is True
    assert result["recorded"]["demand_status"] == "unverified"
    assert result["recorded"]["unknown_components"] == ["image:renderer"]
    small_fs = {"kind": "free_space", "unknown_components": [],
                "requirements": [space(tmp_path, 1, floor=10 ** 18)]}
    failed = evaluate_rule(small_fs, roots=roots)
    assert failed["ok"] is False and failed["reason"] == "filesystem_below_floor"


def test_known_demand_above_capacity_refuses_to_compile():
    inventory = fixture_inventory()
    inventory["disk"]["capacity_bytes"] = 20_000_000_000
    assert_refused(plan(inventory=inventory), "disk_insufficient_for_known_demand")
    # Below the request floor alone stays an inventory blocking issue; A9b checks the host.
    inventory["disk"]["capacity_bytes"] = 250_000_000_000
    assert plan(inventory=inventory)["compiled"] is True


def test_anchors_name_the_source_of_every_flag():
    """Review A2-F6: flags not in R2/ONBOARDING are cited to the probe or labelled."""
    compiled = plan()
    assert "vast_s2_host_probe.sh::docker_engine" in stage(compiled, "A3a_docker_version")["source_anchor"]
    assert "vast_s2_host_probe.sh::gpu_identity" in stage(compiled, "A5_driver_version")["source_anchor"]
    anchor = stage(compiled, "A7a_rustup_fetch")["source_anchor"]
    assert "| sh" in anchor and "agent change" in anchor and "-o" in anchor


REPO = Path(__file__).resolve().parents[1]
PHYSLOOP_ANCHOR = re.compile(r"physloop:(?P<path>[^\s:;()]+)(?P<rest>[^\s;()]*)"
                             r"(?:\s+argv\s+(?P<token>[^\s;()]+))?")


def _python_symbol(path, symbol):
    """The AST node defining ``symbol`` (``name`` or ``Class.method``) in ``path``, or None."""
    nodes = ast.parse(path.read_text(encoding="utf-8")).body
    node = None
    for part in symbol.split("."):
        node = next((n for n in nodes if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part), None)
        if node is None:
            return None
        nodes = node.body
    return node


def _shell_symbol(path, symbol):
    """A shell function, a ``check <category> <name>`` id or a top-level assignment."""
    name = re.escape(symbol)
    return re.search(rf"(?m)^(?:{name}\(\)\s*\{{|\s*check \S+ {name}\s|{name}=)",
                     path.read_text(encoding="utf-8")) is not None


def test_physloop_anchors_name_an_existing_file_and_symbol():
    """Review A2R2: PhysLoop's own files are cited by symbol, never by line, so edits elsewhere
    in the cited file cannot make an anchor stale; upstream repo@sha:path:line anchors stay."""
    anchors = [(s["id"], m) for s in plan()["stages"]
               for m in PHYSLOOP_ANCHOR.finditer(s["source_anchor"])]
    assert {stage_id for stage_id, _ in anchors} >= {
        "A1_host_probe", "A3a_docker_version", "A5_driver_version", "D2b_renderer_inspect",
        "E3_scene_catalog"}
    for stage_id, match in anchors:
        symbol = re.fullmatch(r"::([A-Za-z_][\w.]*)", match["rest"])
        assert symbol, (stage_id, match[0], "physloop anchors cite path::symbol, not a line")
        path = REPO / match["path"]
        assert path.is_file(), (stage_id, match["path"])
        if path.suffix == ".py":
            node = _python_symbol(path, symbol[1])
            assert node is not None, (stage_id, match[0])
            if match["token"]:
                constants = {n.value for n in ast.walk(node)
                             if isinstance(n, ast.Constant) and isinstance(n.value, str)}
                assert match["token"] in constants, (stage_id, match[0])
        else:
            assert path.suffix == ".sh", (stage_id, match[0])
            assert _shell_symbol(path, symbol[1]), (stage_id, match[0])
            assert match["token"] is None, (stage_id, match[0])
    e3 = stage(plan(), "E3_scene_catalog")["source_anchor"]
    assert "physloop:gate/s2_launch.py::_command_group argv +nurec_scenes=curated_val" in e3
    assert "alpasim@e121e59:src/wizard/configs/nurec_scenes/curated_val.yaml:9-11" in e3


def stage_template(**overrides):
    base = {
        "id": "S1", "phase": "B", "purpose": "t", "source_anchor": "R2 B1", "argv": ["true"],
        "cwd": "/", "env_allowlist": [], "env_set": {}, "secret_env": {}, "stdin_secret": None,
        "requires": ["host_setup"], "expected_outputs": [], "idempotency_check": None,
        "cleanup_argv": None, "timeout_s": 10, "level": "L2", "records_first_delivery": False,
        "reviewed_script": False, "attempt_scoped": False}
    base.update(overrides)
    return base


def resealed(compiled, **changes):
    forged = dict(deepcopy(compiled), **changes)
    forged.pop("plan_sha256", None)
    forged["plan_sha256"] = identity_hash(forged)
    return forged


@pytest.mark.parametrize("bad_stage", [
    stage_template(id=["S1"]),
    stage_template(secret_env={"HF_TOKEN": ["hf_token"]}),
    stage_template(stdin_secret=["ngc_api_key"]),
    stage_template(requires=[["host_setup"]]),
    stage_template(expected_outputs=[{"kind": "file", "root": ["checkout"], "path": "/x"}]),
    stage_template(env_set={"A": ["b"]}),
    stage_template(secret_env="HF_TOKEN"),
])
def test_validate_plan_never_raises_on_unhashable_values(bad_stage):
    """Review A2-F7."""
    compiled = plan(record=action("B", {"host_setup": "SYNTHETIC-ref"}), phases="B")
    forged = resealed(compiled, stages=[bad_stage])
    issues = validate_plan(forged)
    assert issues and all(i["severity"] == "error" for i in issues)
    forged = resealed(compiled, secrets_required=[["hf_token"]])
    assert validate_plan(forged)


def test_build_never_raises_on_unhashable_inventory_values():
    inventory = fixture_inventory()
    inventory["images"][0]["role"] = []
    assert_refused(plan(inventory=inventory), "inventory_invalid")
    inventory = fixture_inventory()
    inventory["assets"][2]["files"][0]["path"] = []
    assert_refused(plan(inventory=inventory), "inventory_invalid")


def zip_with(flag_bits=None, method=None):
    import io
    import struct
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("metadata.yaml", "uuid: 00000000-0000-4000-8000-000000000001\n")
    data = bytearray(buffer.getvalue())
    for signature, flag_offset, method_offset in ((b"PK\x03\x04", 6, 8), (b"PK\x01\x02", 8, 10)):
        at = data.index(signature)
        if flag_bits is not None:
            struct.pack_into("<H", data, at + flag_offset, flag_bits)
        if method is not None:
            struct.pack_into("<H", data, at + method_offset, method)
    return bytes(data)


@pytest.mark.parametrize("blob", [zip_with(flag_bits=1), zip_with(method=93)])
def test_usdz_rule_fails_closed_on_encrypted_or_unsupported_members(tmp_path, blob):
    """Review A2-F4: zipfile raises RuntimeError/NotImplementedError here."""
    (tmp_path / "s.usdz").write_bytes(blob)
    rule = {"kind": "usdz_metadata_uuid", "root": "r", "path": str(tmp_path / "s.usdz"),
            "uuid": "00000000-0000-4000-8000-000000000001"}
    result = evaluate_rule(rule, roots={"r": str(tmp_path)})
    assert result["ok"] is False and result["reason"] == "unreadable"


def test_evaluate_rule_never_raises_from_a_failing_probe_runner():
    def runner(argv, **kwargs):
        raise RuntimeError("runner exploded")

    rule = {"kind": "probe", "argv": ["x"], "cwd": "/", "pattern": "y", "timeout_s": 5}
    result = evaluate_rule(rule, roots={}, runner=runner)
    assert result["ok"] is False and result["reason"] == "rule_error"


def test_option_like_file_name_refuses_and_metacharacters_arrive_literally():
    """Review A2-F8/F10: '-x' names are refused; a spaced/metacharacter name is one argv
    element, its rule path validates and nothing is shell-evaluated."""
    inventory = fixture_inventory()
    inventory["assets"][2]["files"][1]["path"] = "--force-download"
    assert_refused(plan(inventory=inventory), "inventory_invalid")
    inventory = fixture_inventory()
    name = "odd name;$(touch pwned)'.json"
    inventory["assets"][2]["files"][1]["path"] = name
    compiled = plan(inventory=inventory)
    assert compiled["compiled"] is True, compiled["issues"]
    assert validate_plan(compiled) == []
    argv = stage(compiled, "E2b_processor_cache")["argv"]
    assert argv.count(name) == 1 and argv[7] == name
    paths = [r["path"] for r in stage(compiled, "E2b_processor_cache")["expected_outputs"]]
    assert paths[1].endswith("/89644892e4d85e24eaac8bacfd4f463576704203/" + name)


def test_host_probe_output_directory_is_attempt_scoped():
    """Review A2-F9: the probe refuses an existing directory (exit 73), so each attempt gets
    its own directory; only attempt-scoped stages may carry the token."""
    a1 = stage(plan(), "A1_host_probe")
    assert a1["attempt_scoped"] is True
    assert a1["argv"][2] == "/data/evidence/s2-bootstrap/host-probe-attempt-__ATTEMPT__"
    assert a1["expected_outputs"][0]["path"] == (
        "/data/evidence/s2-bootstrap/host-probe-attempt-__ATTEMPT__/result.txt")
    assert a1["idempotency_check"] == a1["expected_outputs"]
    assert [s["id"] for s in plan()["stages"] if s["attempt_scoped"]] == ["A1_host_probe"]
    compiled = plan(record=action("B", {"host_setup": "SYNTHETIC-ref"}), phases="B")
    forged = resealed(compiled, stages=[stage_template(argv=["x", "/a/__ATTEMPT__"])])
    assert "invalid_stage" in {i["code"] for i in validate_plan(forged)}
    forged = resealed(compiled, stages=[stage_template(attempt_scoped=True)])
    assert "invalid_stage" in {i["code"] for i in validate_plan(forged)}


def test_plan_declares_units():
    """Review A2-F11."""
    compiled = plan()
    assert compiled["units"] == {"timeout_s": "seconds", "size_bytes": "bytes",
                                 "bytes": "bytes", "filesystem_floor_bytes": "bytes"}
    refused = plan(record=action("ABCD"))
    assert refused["units"] == compiled["units"]
    stripped = {k: v for k, v in compiled.items() if k not in ("units", "plan_sha256")}
    stripped["plan_sha256"] = identity_hash(stripped)
    assert "invalid_plan" in {i["code"] for i in validate_plan(stripped)}


# Scene catalog rule (review A2-F13), mirroring alpasim@e121e59 sceneset.py: CSVs are
# concatenated, uuid must be unique across files, rows are filtered by scene_id and the newest
# last_modified row is the wizard's choice.
HEADER = "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
U1, U2 = "00000000-0000-4000-8000-000000000001", "00000000-0000-4000-8000-000000000009"


def catalog_row(uuid, scene, path, modified="2026-03-16 22:00:27", repo="huggingface",
                revision="26.04"):
    return f"{uuid},{scene},26.4.96-91b06fb8,{path},{modified},{repo},{revision}\n"


def catalog_rule(tmp_path, texts, **overrides):
    catalogs = []
    for index, text in enumerate(texts):
        path = tmp_path / f"cat{index}.csv"
        path.write_text(text)
        catalogs.append({"root": "checkout", "path": str(path),
                         "sha256": hashlib.sha256(text.encode()).hexdigest()})
    rule = {"kind": "scene_catalog", "catalogs": catalogs, "scene_id": f"clipgt-{CLIP1}",
            "uuid": U1, "scene_path": SCENE1, **overrides}
    return evaluate_rule(rule, roots={"checkout": str(tmp_path)})


def test_scene_catalog_rule_known_answers(tmp_path):
    scene = f"clipgt-{CLIP1}"
    good = HEADER + catalog_row(U1, scene, SCENE1)
    result = catalog_rule(tmp_path, [HEADER + catalog_row(U2, "clipgt-other", "x.usdz"), good])
    assert result["ok"] is True, result
    assert result["recorded"]["hf_revision"] == "26.04"
    assert result["recorded"]["uuid"] == U1
    cases = [
        ([HEADER + catalog_row(U2, scene, SCENE1)], "uuid_mismatch"),
        ([HEADER + catalog_row(U1, scene, "other/path.usdz")], "path_mismatch"),
        ([HEADER + catalog_row(U1, "clipgt-other", SCENE1)], "scene_not_in_catalog"),
        ([good, good], "duplicate_uuid"),
        ([HEADER + catalog_row(U1, scene, SCENE1, repo="swiftstack")], "not_huggingface"),
        ([HEADER.replace("hf_revision", "rev") + catalog_row(U1, scene, SCENE1)],
         "catalog_header_mismatch"),
        # The wizard keeps the newest artifact per scene_id: the registered uuid is stale.
        ([good + catalog_row(U2, scene, SCENE1, modified="2026-08-05 19:15:07")],
         "uuid_mismatch"),
        ([good + catalog_row(U2, scene, SCENE1)], "ambiguous_scene"),
    ]
    for index, (texts, reason) in enumerate(cases):
        sub = tmp_path / f"case{index}"
        sub.mkdir()
        result = catalog_rule(sub, texts)
        assert result["ok"] is False and result["reason"] == reason, (index, result)
    sub = tmp_path / "tampered"
    sub.mkdir()
    result = catalog_rule(sub, [good], catalogs=[{"root": "checkout", "path": str(sub / "c.csv"),
                                                  "sha256": "0" * 64}])
    assert result["ok"] is False


def test_scene_catalog_stage_checks_the_pinned_curated_val_catalogs():
    compiled = plan()
    catalog = stage(compiled, "E3_scene_catalog")
    assert catalog["argv"] is None and len(catalog["expected_outputs"]) == 5
    first = catalog["expected_outputs"][0]
    # sha256 of the git blobs at alpasim@e121e59 (5cc40ad1..., 0dabd6be...); curated_val.yaml
    # lists exactly these two scene catalogs.
    assert first == {
        "kind": "scene_catalog",
        "catalogs": [
            {"root": "checkout", "path": "/data/alpasim/data/scenes/sim_scenes.csv",
             "sha256": "b1cc2d2fd0235fcfbaac7258a2943e1076376f7f1d51c64553e1a73b5797a7e0"},
            {"root": "checkout", "path": "/data/alpasim/data/scenes/sim_scenes_2604.csv",
             "sha256": "cf0df2b74d2129a3a43d0a4fe5d120b73bf4f63ee70fa7d534db5635eedbf872"}],
        "scene_id": f"clipgt-{CLIP1}", "uuid": "00000000-0000-4000-8000-000000000001",
        "scene_path": SCENE1}
    ids = [s["id"] for s in compiled["stages"]]
    assert ids.index("E3_scene_catalog") < ids.index("E3_scene1_download")


# --- A2X: registered plan kinds (one compiler entry per inventory kind) ---------------------


def test_plan_records_its_registered_inventory_kind_and_compiler():
    from gate.s2_bootstrap import PLAN_KINDS

    assert sorted(PLAN_KINDS) == ["s2_reexport", "s2_runtime", "s4_runtime"]
    compiled = plan()
    assert compiled["compiler"] == {"inventory_kind": "s2_runtime",
                                    "function": "gate.s2_bootstrap.build_execution_plan"}
    refused = build_execution_plan(None, None, phases=["A"])
    assert refused["compiled"] is False and refused["compiler"] == compiled["compiler"]


@pytest.mark.parametrize("compiler, code", [
    ({"inventory_kind": "s9_runtime", "function": "gate.s2_bootstrap.build_execution_plan"},
     "unknown_plan_kind"),
    (None, "unknown_plan_kind"),
    ("s2_runtime", "unknown_plan_kind"),
    ({"inventory_kind": "s2_runtime", "function": "teacher_rl.ops.build_s4_bootstrap_plan"},
     "compiler_mismatch"),
    ({"inventory_kind": "s4_runtime", "function": "teacher_rl.ops.build_s4_bootstrap_plan"},
     "compiler_mismatch"),
    ({"inventory_kind": "s2_runtime", "function": "gate.s2_bootstrap.build_execution_plan",
      "extra": 1}, "compiler_mismatch"),
])
def test_validate_plan_refuses_unknown_kinds_and_compiler_mismatches(compiler, code):
    forged = resealed(plan(phases="B"), compiler=compiler)
    issues = validate_plan(forged)
    assert issues and issues[0]["code"] == code, issues


# --- F3: guest runtime stages, delivered-host fit, sealed source tree --------------------

UV_SHA = "b4dfaef47d491a7296981f8374a4595f55dbf84e8937c8ecd2983574d8bb3da6"
UV_URL = ("https://github.com/astral-sh/uv/releases/download/0.12.24/"
          "uv-x86_64-unknown-linux-gnu.tar.gz")
UV_ARCHIVE = "/data/scratch/uv-0.12.24-x86_64-unknown-linux-gnu.tar.gz"
UV_DIR = "/data/scratch/uv-0.12.24-x86_64-unknown-linux-gnu"
PY = "cpython-3.12.11-linux-x86_64-gnu"
GUEST = ["uv", "run", "--frozen", "--no-dev", "--no-sync", "--no-config", "python"]
BUNDLE = "/data/scratch/physloop-source-" + "b" * 64 + ".tar"


def test_guest_runtime_stages_literal_argv_and_rules():
    compiled = plan()
    argv = {s["id"]: s["argv"] for s in compiled["stages"]}
    assert argv["A0_directories"] == [
        "install", "-d", "/data/evidence/s2-bootstrap", "/data/scratch", "/data/physloop-data",
        UV_DIR, "/root/.local/bin"]
    assert argv["A6a_uv_fetch"] == [
        "curl", "--proto", "=https", "--proto-redir", "=https", "--tlsv1.2", "-sSfL", "-o",
        UV_ARCHIVE, UV_URL]
    assert stage(compiled, "A6a_uv_fetch")["expected_outputs"] == [
        {"kind": "file", "root": "download_scratch", "path": UV_ARCHIVE,
         "size_bytes": 17_249_236, "sha256": UV_SHA}]
    assert argv["A6b_uv_extract"] == [
        "tar", "-x", "-z", "-f", UV_ARCHIVE, "-C", UV_DIR, "--no-same-owner",
        "--strip-components", "1"]
    assert argv["A6c_uv_install"] == ["install", "-m", "0755", f"{UV_DIR}/uv",
                                      "/root/.local/bin/uv"]
    probe = stage(compiled, "A6c_uv_install")["expected_outputs"][0]
    assert probe["argv"] == ["/root/.local/bin/uv", "--version"]
    assert re.search(probe["pattern"], "uv 0.12.24 (abc 2026-10-08)\n")
    assert not re.search(probe["pattern"], "uv 0.12.240\n")
    assert not re.search(stage(compiled, "A6d_uv_version")["expected_outputs"][0]["pattern"],
                         "uv 0.13.5\n")
    assert argv["A6e_python_install"] == ["uv", "python", "install", "--no-config", "--no-bin",
                                          PY]
    found = stage(compiled, "A6e_python_install")["expected_outputs"][0]
    assert found["argv"] == ["uv", "python", "find", "--no-config", "--managed-python",
                             "--no-python-downloads", PY]
    assert re.search(found["pattern"],
                     f"/root/.local/share/uv/python/{PY}/bin/python3.12\n")
    assert not re.search(found["pattern"], "/usr/bin/python3.12\n")
    assert argv["A6f_bundle_archive"] is None
    assert stage(compiled, "A6f_bundle_archive")["expected_outputs"] == [
        {"kind": "file", "root": "download_scratch", "path": BUNDLE, "size_bytes": 4_096,
         "sha256": "c" * 64}]
    # --touch (F3-2): unpacked files get the unpack time, not the archive's mtime 0, so a
    # __pycache__ entry compiled from an earlier unpack never validates against new bytes.
    assert argv["A6g_bundle_unpack"] == ["tar", "-x", "-f", BUNDLE, "-C", "/opt/physloop",
                                         "--no-same-owner", "--touch"]
    tree = {"kind": "source_tree", "root": "physloop_root", "path": "/opt/physloop",
            "bundle_sha256": "b" * 64}
    assert stage(compiled, "A6g_bundle_unpack")["expected_outputs"][1] == tree
    assert argv["A6h_runtime_sync"] == [
        "uv", "sync", "--frozen", "--no-dev", "--no-install-project", "--no-build",
        "--managed-python", "--no-python-downloads", "--no-config", "--python", PY]
    assert stage(compiled, "A6h_runtime_sync")["cwd"] == "/opt/physloop"
    assert stage(compiled, "A6h_runtime_sync")["expected_outputs"] == [tree]
    assert argv["A6i_runtime_imports"][:7] == GUEST and argv["A6i_runtime_imports"][7] == "-c"
    assert stage(compiled, "A6i_runtime_imports")["cwd"] == "/opt/physloop"
    pattern = stage(compiled, "A6i_runtime_imports")["expected_outputs"][0]["pattern"]
    good = "3.12.11 6.0.3 25.0.1 7.36.2 /opt/physloop/.venv /opt/physloop/gate torch=absent\n"
    assert re.search(pattern, good)
    for bad in (good.replace("3.12.11", "3.10.12"), good.replace("6.0.3", "6.0.2"),
                good.replace("torch=absent", "torch=present"),
                good.replace("/opt/physloop/.venv", "/usr"),
                good.replace("/opt/physloop/gate", "/tmp/gate")):
        assert not re.search(pattern, bad), bad
    # Stage order inside phase A: the runtime comes after the host prerequisites.
    ids = [s["id"] for s in compiled["stages"]]
    assert ids.index("A5_driver_version") < ids.index("A6a_uv_fetch") < ids.index(
        "A6i_runtime_imports") < ids.index("A7a_rustup_fetch")
    assert ids[:2] == ["A0_directories", "A1_host_probe"]


def test_runtime_import_check_program_runs_against_this_checkout(tmp_path):
    """The -c program is real Python: run it here with the local interpreter (no tool)."""
    import subprocess
    import sys

    program = stage(plan(), "A6i_runtime_imports")["argv"][-1]
    # -I implies -E, so PYTHONDONTWRITEBYTECODE is ignored: -B keeps gate/__pycache__ unwritten
    # (F3-7), and the program asserts the flag took effect before importing anything.
    result = subprocess.run([sys.executable, "-I", "-B", "-c", "import sys; "
                             "assert sys.flags.dont_write_bytecode, 'bytecode writes enabled'; "
                             "sys.path.insert(0, '" + str(REPO) + "'); exec(compile("
                             + repr(program) + ", 'check', 'exec'))"],
                            capture_output=True, text=True, timeout=60, check=False,
                            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
    assert result.returncode == 0, result.stderr
    fields = result.stdout.split()
    # The local interpreter is whatever 3.12.x uv picked (CI: the runner's system 3.12, F3-4);
    # the exact guest pin 3.12.11 is asserted only in the A6i regex test.
    assert re.fullmatch(r"3\.12\.\d+", fields[0]), fields[0]
    assert fields[1:5] == ["6.0.3", "25.0.1", "7.36.2", fields[4]]
    assert fields[5] == str(REPO / "gate")
    assert fields[6] in ("torch=absent", "torch=present")


def test_host_fit_rule_rides_on_the_attempt_scoped_probe():
    a1 = stage(plan(), "A1_host_probe")
    fit = a1["expected_outputs"][1]
    assert fit == {"kind": "host_fit", "root": "evidence_dir",
                   "path": "/data/evidence/s2-bootstrap/host-probe-attempt-__ATTEMPT__",
                   "requirements": REQUIRED_RESOURCES}
    assert a1["idempotency_check"] == a1["expected_outputs"]


def test_phase_a_refuses_without_required_resources_bundle_or_uv_checksum():
    assert_refused(plan(record=action(resources=None)), "host_requirements_unset")
    # Phases that do not probe the host need no rental resources.
    record = action("B", {"host_setup": "SYNTHETIC-ref"}, resources=None)
    assert plan(record=record, phases="B")["compiled"] is True
    inventory = fixture_inventory()
    inventory["guest_runtime"]["source_bundle"]["archive_sha256"] = None
    assert_refused(plan(inventory=inventory), "source_bundle_unset")
    assert plan(inventory=inventory, record=action("BCDE"), phases="BCDE")["compiled"] is True
    inventory = fixture_inventory()
    inventory["guest_runtime"]["uv"]["archive_sha256"] = "unverified"
    assert_refused(plan(inventory=inventory), "uv_archive_unverified")
    inventory = fixture_inventory()
    inventory["guest_runtime"]["uv"]["archive_sha256"] = "0" * 64
    assert_refused(plan(inventory=inventory), "inventory_invalid")


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("required_resources"),
    lambda r: r["required_resources"].update(gpu_count=0),
    lambda r: r["required_resources"].update(gpu_name="TBD"),
    lambda r: r["required_resources"].pop("min_vcpu"),
    lambda r: r.update(required_resources=[]),
])
def test_malformed_required_resources_refuse(mutate):
    record = action()
    mutate(record)
    assert_refused(plan(record=record), "invalid_action_record")


def write_probe(directory, *, cap=True, vcpu="15\n"):
    from test_s2_host_fit import CAP_LOG, CHECKS, DATA_DF, DOCKER_DF, FREE_LOG, GPU_LOG

    directory.mkdir(parents=True)
    files = {"checks.tsv": CHECKS, "gpu_identity.log": GPU_LOG, "cpu_count.log": vcpu,
             "memory_bytes.log": FREE_LOG, "docker_root_free.log": DOCKER_DF,
             "data_root_free.log": DATA_DF}
    if cap:
        files["gpu_compute_cap.log"] = CAP_LOG
    for name, text in files.items():
        (directory / name).write_text(text)
    return directory


def fit_rule(path):
    return {"kind": "host_fit", "root": "r", "path": str(path),
            "requirements": dict(REQUIRED_RESOURCES)}


def test_host_fit_rule_known_answers(tmp_path):
    roots = {"r": str(tmp_path)}
    probe = write_probe(tmp_path / "attempt-1")
    ok = evaluate_rule(fit_rule(probe), roots=roots)
    assert ok["ok"] is True, ok
    assert ok["recorded"]["facts"]["vcpu"] == 15
    assert ok["recorded"]["automatic_replacement"] is False
    short = write_probe(tmp_path / "attempt-2", vcpu="4\n")
    failed = evaluate_rule(fit_rule(short), roots=roots)
    assert failed["ok"] is False and failed["reason"] == "host_fit_blocked"
    assert failed["detail"] == ["vcpu_short"]
    assert failed["recorded"]["issues"][0]["path"] == "facts.vcpu"
    # Today's probe has no compute_cap check: the fact is unverified and A1 fails closed.
    nocap = write_probe(tmp_path / "attempt-3", cap=False)
    assert evaluate_rule(fit_rule(nocap), roots=roots)["detail"] == ["host_fact_unverified"]
    assert evaluate_rule(fit_rule(tmp_path / "absent"), roots=roots)["reason"] == "missing"
    assert evaluate_rule(fit_rule("/etc"), roots=roots)["reason"] == "outside_root"
    (tmp_path / "file").write_text("x")
    assert evaluate_rule(fit_rule(tmp_path / "file"), roots=roots)["reason"] == (
        "not_a_directory")
    linked = write_probe(tmp_path / "attempt-4", cap=False)
    (linked / "gpu_compute_cap.log").symlink_to(probe / "gpu_compute_cap.log")
    assert evaluate_rule(fit_rule(linked), roots=roots)["detail"] == ["host_fact_unverified"]
    bad = dict(fit_rule(probe), requirements={"gpu_count": 1})
    assert evaluate_rule(bad, roots=roots)["detail"] == ["invalid_requirements"]


def test_host_fit_rule_path_follows_the_attempt_substitution():
    from gate.s2_bootstrap import substitute_attempt

    a1 = substitute_attempt(stage(plan(), "A1_host_probe"), 3)
    assert a1["expected_outputs"][1]["path"] == (
        "/data/evidence/s2-bootstrap/host-probe-attempt-3")
    assert a1["idempotency_check"][1]["path"] == a1["expected_outputs"][1]["path"]


def sealed_tree(tmp_path):
    """A tiny unpacked bundle: manifest + two files; returns (root, rule, roots)."""
    root = tmp_path / "physloop"
    (root / "gate").mkdir(parents=True)
    (root / "gate" / "a.py").write_bytes(b"abc")
    (root / ".python-version").write_bytes(b"3.12\n")
    manifest = {
        "artifact_kind": "s2_source_manifest", "schema_version": 1, "scope": "synthetic",
        "producer": {"module": "test", "source_sha256": None}, "inputs": {},
        "units": {"size_bytes": "bytes"}, "issues": [],
        "files": [{"path": ".python-version", "size_bytes": 5,
                   "sha256": hashlib.sha256(b"3.12\n").hexdigest()},
                  {"path": "gate/a.py", "size_bytes": 3,
                   "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}
    from gate.s2_identity import canonical_json_bytes

    (root / "physloop-source-manifest.json").write_bytes(canonical_json_bytes(manifest))
    rule = {"kind": "source_tree", "root": "p", "path": str(root),
            "bundle_sha256": identity_hash(manifest)}
    return root, rule, {"p": str(root)}


def test_source_tree_rule_known_answer_and_allowed_guest_dirs(tmp_path):
    root, rule, roots = sealed_tree(tmp_path)
    ok = evaluate_rule(rule, roots=roots)
    assert ok["ok"] is True and ok["recorded"]["files_checked"] == 2
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "x.py").write_bytes(b"x")
    (root / "gate" / "__pycache__").mkdir()
    (root / "gate" / "__pycache__" / "a.cpython-310.pyc").write_bytes(b"x")
    assert evaluate_rule(rule, roots=roots)["ok"] is True


@pytest.mark.parametrize("damage, reason", [
    (lambda root: (root / "gate" / "extra.py").write_bytes(b"x"), "tree_mismatch"),
    (lambda root: (root / "gate" / ".venv").mkdir() or (root / "gate" / ".venv" / "x").write_bytes(
        b"x"), "tree_mismatch"),  # only the top-level .venv is the guest's
    (lambda root: (root / "gate" / "a.py").write_bytes(b"abd"), "tree_mismatch"),
    (lambda root: (root / "gate" / "a.py").unlink(), "tree_mismatch"),
    (lambda root: (root / "gate" / "l.py").symlink_to(root / "gate" / "a.py"),
     "symlink_in_tree"),
    (lambda root: (root / "linked").symlink_to(root / "gate", target_is_directory=True),
     "symlink_in_tree"),
    (lambda root: (root / "physloop-source-manifest.json").write_bytes(b"{}"),
     "manifest_invalid"),
    (lambda root: (root / "physloop-source-manifest.json").unlink(), "manifest_unreadable"),
])
def test_source_tree_rule_negative_controls(tmp_path, damage, reason):
    root, rule, roots = sealed_tree(tmp_path)
    damage(root)
    result = evaluate_rule(rule, roots=roots)
    assert result["ok"] is False and result["reason"] == reason, result


def test_source_tree_rule_detail_names_the_differences(tmp_path):
    root, rule, roots = sealed_tree(tmp_path)
    (root / "gate" / "extra.py").write_bytes(b"x")
    (root / "gate" / "a.py").write_bytes(b"abd")
    (root / ".python-version").unlink()
    detail = evaluate_rule(rule, roots=roots)["detail"]
    assert detail == {"extra": ["gate/extra.py"], "changed": ["gate/a.py"],
                      "missing": [".python-version"]}
    root, rule, roots = sealed_tree(tmp_path / "again")
    assert evaluate_rule(dict(rule, bundle_sha256="0" * 64), roots=roots)["reason"] == (
        "bundle_sha256_mismatch")


V1, V2 = b"VALUE = 'AAAA'\n", b"VALUE = 'BBBB'\n"  # same size: only bytes differ (F3-2)


def bytecode_tree(tmp_path, *, source_ns, pyc_ns, compiled_ns=None):
    """A sealed tree whose gate/m.py is V2 while __pycache__ holds bytecode compiled from V1.

    The V1 bytecode is compiled with the source mtime ``compiled_ns`` (default ``source_ns``:
    the first unpack) and the pyc file's own mtime is set to ``pyc_ns``; then V2 (same size)
    replaces the source at ``source_ns`` - a second bundle unpacked over kept storage.
    Returns (root, rule, roots, pyc).
    """
    import py_compile

    from gate.s2_identity import canonical_json_bytes

    root = tmp_path / "physloop"
    (root / "gate").mkdir(parents=True)
    module = root / "gate" / "m.py"
    module.write_bytes(V1)
    first = source_ns if compiled_ns is None else compiled_ns
    os.utime(module, ns=(first, first))
    pyc = Path(py_compile.compile(str(module), doraise=True,
                                  invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP))
    os.utime(pyc, ns=(pyc_ns, pyc_ns))
    module.write_bytes(V2)
    os.utime(module, ns=(source_ns, source_ns))
    manifest = {
        "artifact_kind": "s2_source_manifest", "schema_version": 1, "scope": "synthetic",
        "producer": {"module": "test", "source_sha256": None}, "inputs": {},
        "units": {"size_bytes": "bytes"}, "issues": [],
        "files": [{"path": "gate/m.py", "size_bytes": len(V2),
                   "sha256": hashlib.sha256(V2).hexdigest()}]}
    (root / "physloop-source-manifest.json").write_bytes(canonical_json_bytes(manifest))
    rule = {"kind": "source_tree", "root": "p", "path": str(root),
            "bundle_sha256": identity_hash(manifest)}
    return root, rule, {"p": str(root)}, pyc


def load_value(module):
    """VALUE as CPython's own source loader resolves it (it reads __pycache__ when valid)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("f3_stale_probe", module)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded.VALUE


SECOND = 1_700_000_000 * 10**9


@pytest.mark.parametrize("source_ns, pyc_ns", [
    (0, SECOND),  # the archive's mtime 0 restored on both unpacks (no --touch)
    (SECOND + 500_000_000, SECOND + 100_000_000),  # same second, pyc written before the unpack
])
def test_source_tree_rule_refuses_bytecode_that_would_shadow_the_sealed_source(
        tmp_path, source_ns, pyc_ns):
    root, rule, roots, pyc = bytecode_tree(tmp_path, source_ns=source_ns, pyc_ns=pyc_ns)
    # The defect, shown with CPython's own loader: the sealed bytes say BBBB, AAAA runs.
    assert load_value(root / "gate" / "m.py") == "AAAA"
    result = evaluate_rule(rule, roots=roots)
    assert result["ok"] is False and result["reason"] == "stale_bytecode", result
    assert result["detail"] == ["gate/__pycache__/" + pyc.name]


def test_source_tree_rule_accepts_bytecode_compiled_after_the_unpack(tmp_path):
    # --touch: the unpack gives V2 a fresh mtime, so the V1 bytecode no longer validates.
    root, rule, roots, _ = bytecode_tree(tmp_path, compiled_ns=0, source_ns=SECOND + 7 * 10**9,
                                         pyc_ns=SECOND)
    assert load_value(root / "gate" / "m.py") == "BBBB"
    assert evaluate_rule(rule, roots=roots)["ok"] is True
    # Bytecode compiled from the sealed bytes after the unpack is the interpreter's own cache.
    import py_compile

    module = root / "gate" / "m.py"
    py_compile.compile(str(module), doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    assert evaluate_rule(rule, roots=roots)["ok"] is True


def test_source_tree_rule_refuses_unchecked_hash_bytecode_and_symlinked_cache(tmp_path):
    import importlib.util

    root, rule, roots, pyc = bytecode_tree(tmp_path, compiled_ns=0,
                                           source_ns=SECOND + 7 * 10**9, pyc_ns=SECOND)
    # Flags 0b01: hash-based and never checked against the source - it always loads.
    pyc.write_bytes(importlib.util.MAGIC_NUMBER + b"\x01\x00\x00\x00" + b"\x00" * 8)
    result = evaluate_rule(rule, roots=roots)
    assert result["ok"] is False and result["reason"] == "stale_bytecode", result
    pyc.write_bytes(importlib.util.MAGIC_NUMBER + b"\x03\x00\x00\x00" + b"\x00" * 8)
    assert evaluate_rule(rule, roots=roots)["ok"] is True  # checked hash: CPython verifies it
    pyc.unlink()
    pyc.symlink_to(root / "gate" / "m.py")
    result = evaluate_rule(rule, roots=roots)
    assert result["ok"] is False and result["reason"] == "symlink_in_tree", result


def test_source_tree_rule_must_name_a_bundle_identity():
    compiled = plan()
    unpack = stage(compiled, "A6g_bundle_unpack")
    broken = deepcopy(unpack)
    broken["expected_outputs"][1]["bundle_sha256"] = "x"
    forged = resealed(compiled, stages=[broken])
    assert "invalid_rule" in {i["code"] for i in validate_plan(forged)}


def test_uv_lock_rule_fails_closed_without_tomllib(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "tomllib", None)  # import raises ImportError
    result = lock_rule(tmp_path, LOCK_PINNED)
    assert result["ok"] is False and result["reason"] == "tomllib_unavailable"


PHASE_A_RUNNER_GRAPH = ("gate/s2_bootstrap.py", "gate/s2_runtime.py", "gate/s2_identity.py",
                        "gate/s2_host_fit.py", "gate/s2_authority.py",
                        "scripts/s2_guest_bootstrap.py")


def _imports_with_function_scope(tree):
    """Every Import/ImportFrom at any depth, with whether a function body encloses it."""
    found = []

    def visit(node, in_function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                found.append((child, in_function))
            visit(child, in_function or isinstance(child, (ast.FunctionDef,
                                                           ast.AsyncFunctionDef, ast.Lambda)))

    visit(tree, False)
    return found


@pytest.mark.parametrize("relative", PHASE_A_RUNNER_GRAPH)
def test_phase_a_runner_graph_stays_python_3_10_compatible(relative):
    """The phase-A runner starts on the guest's system python3 (Ubuntu 22.04: 3.10).

    Imports at every depth count (F3-1 review: a lazy ``from datetime import UTC`` inside
    ``main`` broke phase A on 3.10); only ``tomllib`` may sit inside a function, where its
    caller fails closed without it.
    """
    source = (REPO / relative).read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 10))
    for node, in_function in _imports_with_function_scope(tree):
        where = (relative, node.lineno)
        if isinstance(node, ast.Import):
            assert in_function or all(alias.name != "tomllib" for alias in node.names), where
            modules = [alias.name for alias in node.names]
        else:
            assert in_function or node.module != "tomllib", where
            if node.module == "datetime":
                assert all(alias.name != "UTC" for alias in node.names), where
            modules = [node.module or ""]
        for module in modules:
            if module.startswith(("gate.", "scripts.")):
                imported = module.replace(".", "/") + ".py"
                assert imported in PHASE_A_RUNNER_GRAPH, (where, imported)
    assert not any(isinstance(node, ast.Attribute) and node.attr == "UTC"
                   and isinstance(node.value, ast.Name) and node.value.id == "datetime"
                   for node in ast.walk(tree)), relative


def test_runner_graph_guard_sees_imports_inside_functions():
    """Negative control for the guard: the F3-1 shape is found, the lazy tomllib is allowed."""
    tree = ast.parse("def main():\n    from datetime import UTC, datetime\n"
                     "def lock():\n    import tomllib\n", feature_version=(3, 10))
    found = [(type(node).__name__, in_function) for node, in_function in
             _imports_with_function_scope(tree)]
    assert found == [("ImportFrom", True), ("Import", True)]


def test_guest_python_prefix_is_the_frozen_environment():
    from gate.s2_bootstrap import GUEST_PYTHON

    assert list(GUEST_PYTHON) == GUEST
    programs = []
    for s in plan()["stages"]:
        assert "python3" not in (s["argv"] or []), s["id"]
        rules = s["expected_outputs"] + (s["idempotency_check"] or [])
        for argv, cwd in [(s["argv"], s["cwd"])] + [(r["argv"], r["cwd"]) for r in rules
                                                    if r["kind"] == "probe"]:
            if argv and cwd == "/opt/physloop" and "-c" in argv:
                programs.append(s["id"])
                assert argv[:7] == GUEST, s["id"]
    assert programs == ["A6h_runtime_sync", "A6i_runtime_imports"]


def test_s2_reexport_kind_is_registered_for_the_single_runner():
    """G4R item 3: the re-export kind is a literal entry (tuples), no import of its compiler."""
    from gate.s2_bootstrap import PLAN_KINDS, compiler_identity, plan_fields, resolve_compiler

    assert PLAN_KINDS["s2_reexport"] == {
        "inventory_artifact_kind": "s2_reexport_inventory",
        "plan_artifact_kind": "s2_reexport_plan",
        "function": "gate.s2_reexport.build_reexport_plan",
        "inputs": ("inventory", "action_record"),
        "checkout_root": "alpagym_checkout",
        "extra_fields": (),
    }
    assert compiler_identity("s2_reexport") == {
        "inventory_kind": "s2_reexport", "function": "gate.s2_reexport.build_reexport_plan"}
    assert plan_fields("s2_reexport") == plan_fields("s2_runtime")
    from gate.s2_reexport import PLAN_KIND_SPEC, build_reexport_plan

    assert PLAN_KINDS["s2_reexport"] == PLAN_KIND_SPEC
    assert resolve_compiler("s2_reexport") is build_reexport_plan


DCGM_GAP = "dcgm_exporter_run_long_stage_absent"


def test_runtime_plan_states_the_run_long_dcgm_exporter_gap():
    """G4R item 5: the anchors support no clean start/stop stage, so every compiled S2
    runtime plan carries an explicit plan-level gap note (info: it never refuses a plan)."""
    for compiled in (plan(), plan(record=action("BC", {"host_setup": "SYNTHETIC-ref"}),
                                  phases="CB")):
        assert compiled["compiled"] is True, compiled["issues"]
        notes = [i for i in compiled["issues"] if i["code"] == DCGM_GAP]
        assert len(notes) == 1, compiled["issues"]
        note = notes[0]
        assert note["severity"] == "info" and note["path"] == "stages"
        for anchor in ("scripts/vast_exporter_control.sh:18", ":47-49", ":114-127",
                       ":128-131", ":143-160", "scripts/s2_dcgm_scraper.py",
                       "gate.s2_telemetry_sources.parse_dcgm_scrapes",
                       "gate.s2_runtime.IMAGE_ROLES", "scripts/s2_run_units.py"):
            assert anchor in note["message"], anchor
        assert not any("dcgm" in " ".join(s["argv"] or []) for s in compiled["stages"])
    refused = build_execution_plan(None, None, phases=["A"])
    assert DCGM_GAP not in {i["code"] for i in refused["issues"]}
