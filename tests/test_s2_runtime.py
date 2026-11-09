"""Agent-authored known answers for the S2 runtime inventory (CPU-01; R2 phases D-E, S3-S11).

Every literal below is hand-derived from the R2 lane report or the hash-checked HF/registry
metadata it cites, never computed by the implementation. No tool, network or provider is used.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from gate.s2_identity import read_json_bytes
from gate.s2_runtime import (
    ALPAMAYO15_SHA,
    ALPASIM_SHA,
    DISK_REQUEST_FLOOR_BYTES,
    LOCKED_RUNTIME,
    PYTHON_REQUEST,
    TEACHER_REVISION,
    UV_ARCHIVE_SHA256,
    UV_ARCHIVE_SIZE_BYTES,
    UV_ARCHIVE_URL,
    UV_VERSION,
    source_bundle_archive_name,
    validate_runtime_inventory,
    validate_source_manifest,
)

ROOT = Path(__file__).parents[1]
NRE_REF = (
    "nvcr.io/nvidia/nre/nre-ga@sha256:"
    "97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284"
)
# Synthetic digest: choosing the public CUDA probe image is a host-time technical input.
CUDA_REF = "nvidia/cuda@sha256:" + "1" * 64
COSMOS_REV = "a9fae2cf89dc64db96b12860417f0eb403013bb9"
QWEN_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
HARMONIZER_REV = "20ca33d4612b1e98e0526b3a7ee604af5b289f58"
HARMONIZER_SHA = "ece8e2daa914e8c2a027a2da94e0eb2064491d5b3fd8514009fae9a442e06e90"
TEACHER_FILES = [
    ("model-00001-of-00005.safetensors", 4_928_204_944,
     "537259bb56815f9dfdcb028d5606e84dff5ab0c8e0e559782d63d91b6671d15c"),
    ("model-00002-of-00005.safetensors", 4_915_963_032,
     "18841e5049b7836c16c034123ae6d40b2ff0414f9f77008bc7ca72490302a47b"),
    ("model-00003-of-00005.safetensors", 4_983_071_160,
     "e8953b27fe827a60604a006b07ab1216fe766375f9926d1c3bd5ecee84ecef78"),
    ("model-00004-of-00005.safetensors", 4_980_341_192,
     "604f0c0f19986f363a23a1e0b5736eedadd231ca285beab28d5db968a1d37602"),
    ("model-00005-of-00005.safetensors", 2_349_614_196,
     "9d889c09634e5a21b4c957957ede2cdab4418c268ee1d870c1f942d04242adf6"),
    ("model.safetensors.index.json", 104_778, "unverified"),
    ("config.json", 3_058, "unverified"),
    ("README.md", 10_446, "unverified"),
    ("LICENSE", 2_619, "unverified"),
    (".gitattributes", 1_519, "unverified"),
]
COSMOS_FILES = [("chat_template.json", 5_499), ("config.json", 1_474),
                ("generation_config.json", 269), ("merges.txt", 1_671_839),
                ("preprocessor_config.json", 390), ("tokenizer.json", 7_032_403),
                ("tokenizer_config.json", 10_868), ("video_preprocessor_config.json", 385),
                ("vocab.json", 2_776_833)]
QWEN_FILES = [("chat_template.json", 5_502), ("config.json", 1_505),
              ("generation_config.json", 269), ("merges.txt", 1_671_839),
              ("preprocessor_config.json", 390), ("tokenizer.json", 7_032_403),
              ("tokenizer_config.json", 10_868), ("video_preprocessor_config.json", 385),
              ("vocab.json", 2_776_833)]
# F3 pins, transcribed from the uv 0.12.24 release metadata read on 2026-11-09 (the published
# uv-x86_64-unknown-linux-gnu.tar.gz.sha256 asset, sha256.sum and the GitHub asset digest agree).
UV_SHA = "b4dfaef47d491a7296981f8374a4595f55dbf84e8937c8ecd2983574d8bb3da6"
UV_URL = ("https://github.com/astral-sh/uv/releases/download/0.12.24/"
          "uv-x86_64-unknown-linux-gnu.tar.gz")
# Synthetic bundle identity: the real one is produced by scripts/s2_prepare_bundle.py at run time.
BUNDLE_SHA = "b" * 64
BUNDLE_ARCHIVE_SHA = "c" * 64
BUNDLE_ARCHIVE_SIZE = 4_096


def guest_runtime():
    return {
        "uv": {"version": "0.12.24", "target": "x86_64-unknown-linux-gnu", "archive_url": UV_URL,
               "archive_sha256": UV_SHA, "archive_size_bytes": 17_249_236},
        "python": {"request": "cpython-3.12.11-linux-x86_64-gnu"},
        "source_bundle": {"bundle_sha256": BUNDLE_SHA, "archive_sha256": BUNDLE_ARCHIVE_SHA,
                          "archive_size_bytes": BUNDLE_ARCHIVE_SIZE},
    }


# Hand sums: teacher 22,157,316,944 (R2 E1); Cosmos 11,499,960; Qwen 11,499,994;
# Harmonizer 1,448,843,112.  22,157,316,944 + 11,499,960 + 11,499,994 + 1,448,843,112.
KNOWN_ASSET_BYTES = 23_629_160_010


def synthetic_scenes():
    """Five obviously synthetic scene rows; real scene choice is Huy's registration."""
    rows = []
    for index in range(1, 6):
        uuid = f"00000000-0000-4000-8000-00000000000{index}"
        clip = f"10000000-0000-4000-8000-00000000000{index}"
        rows.append({
            "scene_id": f"clipgt-{clip}",
            "uuid": uuid,
            "path": f"sample_set/26.04_release/{clip}/{clip}.usdz",
            "revision": f"{index}" * 40,
            "size_bytes": "unverified",
            "sha256": "unverified",
        })
    return rows


def fixture_inventory():
    def files(rows):
        return [{"path": p, "size_bytes": n, "sha256": "unverified"} for p, n in rows]

    return {
        "artifact_kind": "s2_runtime_inventory",
        "schema_version": 1,
        "scope": "synthetic",
        "sources": {
            "alpasim": {"repository": "https://github.com/NVlabs/alpasim", "commit": ALPASIM_SHA},
            "alpamayo1_5": {"repository": "https://github.com/NVlabs/alpamayo1.5.git",
                            "commit": ALPAMAYO15_SHA},
        },
        "platform": "linux/amd64",
        "host_paths": {
            "physloop_root": "/opt/physloop",
            "checkout": "/data/alpasim",
            "data_root": "/data/physloop-data",
            "teacher_snapshot": "/data/models/alpamayo-1.5-10b",
            "hf_home": "/data/hf-home",
            "harmonizer_dir": "/data/models/harmonizer",
            "download_scratch": "/data/scratch",
            "evidence_dir": "/data/evidence/s2-bootstrap",
            "docker_data_root": "/var/lib/docker",
            "home": "/root",
        },
        "toolchain": {"rust_toolchain": "1.91.0"},
        "harmonizer_source": "pinned_mount",
        "images": [
            {"role": "renderer", "kind": "registry_digest", "reference": NRE_REF,
             "platform": "linux/amd64", "size_bytes": "unverified",
             "compressed_bytes": 14_287_662_578, "authorization_ref": "SYNTHETIC-ngc-nre-terms"},
            {"role": "alpasim_base", "kind": "local_build",
             "build": {"context": "checkout", "dockerfile": "Dockerfile",
                       "tag": "alpasim-base:0.89.0", "source_commit": ALPASIM_SHA},
             "image_id": None,
             "mutable_inputs": [
                 "FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
                 "COPY --from=ghcr.io/astral-sh/uv:latest",
                 "unpinned apt-get packages",
                 "latest rustup toolchain",
                 "uv sync --extra all --extra mtgs without uv.lock",
             ],
             "platform": "linux/amd64", "size_bytes": "unverified", "authorization_ref": None},
            {"role": "public_cuda_probe", "kind": "registry_digest", "reference": CUDA_REF,
             "platform": "linux/amd64", "size_bytes": "unverified",
             "compressed_bytes": "unverified", "authorization_ref": None},
        ],
        "assets": [
            {"role": "teacher", "repository": "nvidia/Alpamayo-1.5-10B", "repo_type": "model",
             "revision": TEACHER_REVISION, "authorization_ref": "SYNTHETIC-openmdw-terms",
             "files": [{"path": p, "size_bytes": n, "sha256": s} for p, n, s in TEACHER_FILES]},
            {"role": "vlm_config", "repository": "nvidia/Cosmos-Reason2-8B", "repo_type": "model",
             "revision": COSMOS_REV, "authorization_ref": "SYNTHETIC-cosmos-access",
             "files": files(COSMOS_FILES)},
            {"role": "processor", "repository": "Qwen/Qwen3-VL-2B-Instruct",
             "repo_type": "model", "revision": QWEN_REV, "authorization_ref": None,
             "files": files(QWEN_FILES)},
            {"role": "nurec_scenes", "repository": "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec",
             "repo_type": "dataset", "revision": None,
             "authorization_ref": "SYNTHETIC-nurec-terms", "files": synthetic_scenes()},
            {"role": "harmonizer", "repository": "nvidia/Harmonizer", "repo_type": "model",
             "revision": HARMONIZER_REV, "authorization_ref": "SYNTHETIC-harmonizer-terms",
             "files": [{"path": "harmonizer_nontemporal.pt", "size_bytes": 1_448_843_112,
                        "sha256": HARMONIZER_SHA}]},
        ],
        "disk": {
            "capacity_bytes": 400_000_000_000,
            "components": [
                {"name": "alpasim_host_python_env", "size_bytes": "unverified"},
                {"name": "per_unit_raw_outputs", "size_bytes": "unverified"},
                {"name": "export_temporary_copy", "size_bytes": "unverified"},
                {"name": "physloop_guest_runtime", "size_bytes": "unverified"},
            ],
        },
        "guest_runtime": guest_runtime(),
    }


def codes(report, severity=None):
    return sorted({i["code"] for i in report["issues"]
                   if severity is None or i["severity"] == severity})


def asset(inventory, role):
    return next(item for item in inventory["assets"] if item["role"] == role)


def image(inventory, role):
    return next(item for item in inventory["images"] if item["role"] == role)


def test_pins_are_the_r2_literals():
    assert ALPASIM_SHA == "e121e59141dc391f6c295513d6dadc840a419149"
    assert ALPAMAYO15_SHA == "36aeb4c5938cbc2eb2aed33b22434773da4ab639"
    assert TEACHER_REVISION == "7aba8293c09993f2e125c6819df05d7fa3e873ea"
    # Agent technical default (D-018), R2 sizing: request at least 300 GB.
    assert DISK_REQUEST_FLOOR_BYTES == 300_000_000_000


def test_fixture_passes_with_only_the_pending_local_build_note():
    report = validate_runtime_inventory(fixture_inventory())
    assert report["artifact_kind"] == "s2_runtime_inventory_report"
    assert report["schema_version"] == 1
    assert report["scope"] == "synthetic"
    assert report["valid"] is True
    assert report["status"] == "passed"
    assert codes(report) == ["local_build_image_id_pending"]
    assert codes(report, "info") == ["local_build_image_id_pending"]
    assert report["producer"]["module"] == "gate.s2_runtime"
    assert len(report["input_sha256"]) == 64
    assert report["units"] == {"size_bytes": "bytes"}
    assert report["level"] == "L1"


def test_teacher_bytes_match_r2_total():
    total = sum(n for _, n, _ in TEACHER_FILES)
    assert total == 22_157_316_944


def test_disk_demand_known_answer_with_unknowns_never_zero():
    disk = validate_runtime_inventory(fixture_inventory())["disk_demand"]
    assert disk["known_bytes"] == KNOWN_ASSET_BYTES
    assert disk["total_bytes"] is None
    assert disk["status"] == "unverified"
    assert disk["unknown_components"] == sorted(
        [f"asset:nurec_scenes:{row['path']}" for row in synthetic_scenes()]
        + ["component:alpasim_host_python_env", "component:export_temporary_copy",
           "component:per_unit_raw_outputs", "component:physloop_guest_runtime",
           "image:alpasim_base", "image:public_cuda_probe", "image:renderer"]
    )
    assert disk["request_floor_bytes"] == 300_000_000_000
    assert "agent technical default (D-018)" in disk["request_floor_basis"]
    assert disk["capacity_bytes"] == 400_000_000_000


def test_disk_demand_exact_total_when_everything_is_known():
    inventory = fixture_inventory()
    for item, size in zip(inventory["images"], (1_000, 2_000, 3_000)):
        item["size_bytes"] = size
    for row in asset(inventory, "nurec_scenes")["files"]:
        row["size_bytes"] = 100
    for component, size in zip(inventory["disk"]["components"], (10, 20, 30, 40)):
        component["size_bytes"] = size
    disk = validate_runtime_inventory(inventory)["disk_demand"]
    # 23,629,160,010 + 6,000 (images) + 500 (scenes) + 100 (components).
    assert disk["total_bytes"] == 23_629_166_610
    assert disk["known_bytes"] == 23_629_166_610
    assert disk["unknown_components"] == []
    assert disk["status"] == "passed"


@pytest.mark.parametrize("bad", [0, None, -5, True, 1.5])
def test_unknown_size_must_be_explicit_never_zero_or_null(bad):
    inventory = fixture_inventory()
    asset(inventory, "teacher")["files"][6]["size_bytes"] = bad
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is False
    assert "invalid_size" in codes(report, "error")
    paths = {i["path"] for i in report["issues"] if i["code"] == "invalid_size"}
    assert paths == {"assets[0].files[6].size_bytes"}


def test_disk_capacity_below_floor_and_below_known_demand_block():
    inventory = fixture_inventory()
    inventory["disk"]["capacity_bytes"] = 250_000_000_000
    assert codes(validate_runtime_inventory(inventory), "blocking") == [
        "disk_below_request_floor"]
    inventory["disk"]["capacity_bytes"] = 20_000_000_000
    assert codes(validate_runtime_inventory(inventory), "blocking") == [
        "disk_below_request_floor", "disk_insufficient_for_known_demand"]
    inventory["disk"]["capacity_bytes"] = None
    assert codes(validate_runtime_inventory(inventory), "blocking") == ["disk_capacity_unset"]


@pytest.mark.parametrize("reference", [
    "nvcr.io/nvidia/nre/nre-ga:26.04",
    "nvcr.io/nvidia/nre/nre-ga",
    "nvcr.io/nvidia/nre/nre-ga@sha256:" + "A" * 64,
    "nvcr.io/nvidia/nre/nre-ga@sha256:" + "a" * 63,
    "sha256:" + "a" * 64,
])
def test_mutable_or_malformed_registry_reference_rejected(reference):
    inventory = fixture_inventory()
    image(inventory, "renderer")["reference"] = reference
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is False
    assert "mutable_image_reference" in codes(report, "error")


def test_renderer_requires_registry_digest_not_local_build():
    inventory = fixture_inventory()
    renderer = image(inventory, "renderer")
    base = deepcopy(image(inventory, "alpasim_base"))
    renderer.clear()
    renderer.update(base)
    renderer["role"] = "renderer"
    renderer["authorization_ref"] = "SYNTHETIC-ngc-nre-terms"
    report = validate_runtime_inventory(inventory)
    assert "digest_required" in codes(report, "error")


@pytest.mark.parametrize("image_id", ["alpasim-base:0.89.0", "sha256:" + "z" * 64, "abc"])
def test_local_build_image_id_must_be_bare_digest(image_id):
    inventory = fixture_inventory()
    image(inventory, "alpasim_base")["image_id"] = image_id
    assert "invalid_local_image_id" in codes(validate_runtime_inventory(inventory), "error")


def test_local_build_with_recorded_id_has_no_pending_note():
    inventory = fixture_inventory()
    image(inventory, "alpasim_base")["image_id"] = "sha256:" + "b" * 64
    report = validate_runtime_inventory(inventory)
    assert report["issues"] == []
    assert {i["role"]: i["status"] for i in report["images"]}["alpasim_base"] == "passed"


@pytest.mark.parametrize("mutable_inputs", [[], None, [""], ["ok", 3]])
def test_local_build_must_record_mutable_inputs(mutable_inputs):
    inventory = fixture_inventory()
    image(inventory, "alpasim_base")["mutable_inputs"] = mutable_inputs
    assert "mutable_inputs_unrecorded" in codes(validate_runtime_inventory(inventory), "error")


def test_local_build_source_commit_must_match_pin():
    inventory = fixture_inventory()
    image(inventory, "alpasim_base")["build"]["source_commit"] = "f" * 40
    assert "pin_mismatch" in codes(validate_runtime_inventory(inventory), "error")


@pytest.mark.parametrize("where,value", [
    (("sources", "alpasim", "commit"), "297b4d00cfdd2c4f636b9d9bcee7a85c699174c0"),
    (("sources", "alpamayo1_5", "commit"), "11a0e01c" + "0" * 32),
])
def test_source_pin_mismatch_is_error(where, value):
    inventory = fixture_inventory()
    inventory[where[0]][where[1]][where[2]] = value
    assert "pin_mismatch" in codes(validate_runtime_inventory(inventory), "error")


def test_teacher_revision_pin_mismatch_is_error():
    inventory = fixture_inventory()
    asset(inventory, "teacher")["revision"] = "089866af" + "0" * 32
    assert "pin_mismatch" in codes(validate_runtime_inventory(inventory), "error")


@pytest.mark.parametrize("revision", ["26.04", "main", "a9fae2cf"])
def test_moving_or_short_revision_rejected(revision):
    inventory = fixture_inventory()
    asset(inventory, "vlm_config")["revision"] = revision
    assert "mutable_revision" in codes(validate_runtime_inventory(inventory), "error")
    inventory = fixture_inventory()
    asset(inventory, "nurec_scenes")["files"][0]["revision"] = revision
    assert "mutable_revision" in codes(validate_runtime_inventory(inventory), "error")


# Review A2-F8/F10: names are emitted as literal argv elements (never shell-evaluated), so
# whitespace and shell metacharacters are permitted (CPU-09); option-like components and control
# characters are rejected (a name starting with "-" would be parsed as an hf flag).
@pytest.mark.parametrize("path", ["../escape.json", "/etc/passwd", "a/../../b", "a//b", "./x",
                                  "dir/", "-h", "--force-download", "a/-b.json", "x\n.json",
                                  "tab\t.json", "nul\x00.json", ""])
def test_relative_path_escape_rejected(path):
    inventory = fixture_inventory()
    asset(inventory, "processor")["files"][0]["path"] = path
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is False
    assert "unsafe_relative_path" in codes(report, "error")


def test_duplicate_normalized_file_rejected():
    inventory = fixture_inventory()
    files = asset(inventory, "processor")["files"]
    files.append(dict(files[0]))
    assert "duplicate_file" in codes(validate_runtime_inventory(inventory), "error")


@pytest.mark.parametrize("name,value", [
    ("checkout", "relative/path"), ("data_root", "/"), ("hf_home", "/data/../etc"),
    ("evidence_dir", "/data/alpasim/evidence"), ("teacher_snapshot", "/data/with space"),
])
def test_host_paths_absolute_portable_and_outside_checkout(name, value):
    inventory = fixture_inventory()
    inventory["host_paths"][name] = value
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is False
    assert set(codes(report, "error")) & {"invalid_host_path", "host_path_overlap"}


def test_gated_items_need_authorization_refs_public_items_refuse_them():
    inventory = fixture_inventory()
    for role in ("teacher", "vlm_config", "nurec_scenes", "harmonizer"):
        asset(inventory, role)["authorization_ref"] = None
    image(inventory, "renderer")["authorization_ref"] = None
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is True
    missing = sorted(i["path"] for i in report["issues"] if i["code"] == "authorization_missing")
    assert missing == ["assets[0].authorization_ref", "assets[1].authorization_ref",
                       "assets[3].authorization_ref", "assets[4].authorization_ref",
                       "images[0].authorization_ref"]
    gated = {item["role"]: item["gated"] for item in report["assets"]}
    assert gated == {"teacher": True, "vlm_config": True, "processor": False,
                     "nurec_scenes": True, "harmonizer": True}
    assert {i["role"]: i["terms"] for i in report["assets"]}["teacher"] == "OpenMDW-1.1"

    inventory = fixture_inventory()
    asset(inventory, "processor")["authorization_ref"] = "SYNTHETIC-ref"
    assert "unexpected_authorization_ref" in codes(validate_runtime_inventory(inventory), "error")


def test_harmonizer_terms_reference_is_required_for_both_sources():
    """Review A2-F12: the upstream default still downloads nvidia/Harmonizer under its licence
    in every on-arm renderer, so the terms record is required (and permitted) for both sources."""
    for source in ("pinned_mount", "upstream_default_download"):
        inventory = fixture_inventory()
        inventory["harmonizer_source"] = source
        report = validate_runtime_inventory(inventory)
        assert "unexpected_authorization_ref" not in codes(report), source
        assert {i["role"]: i["authorization"] for i in report["assets"]}["harmonizer"] == (
            "recorded")
        asset(inventory, "harmonizer")["authorization_ref"] = None
        report = validate_runtime_inventory(inventory)
        missing = [i["path"] for i in report["issues"] if i["code"] == "authorization_missing"]
        assert missing == ["assets[4].authorization_ref"], source


@pytest.mark.parametrize("ref", ["TBD", " todo ", "Later", "____", "placeholder", "XXX", "n/a",
                                 "?", "", "hf_" + "a" * 34])
def test_placeholder_or_secret_authorization_ref_rejected(ref):
    inventory = fixture_inventory()
    asset(inventory, "teacher")["authorization_ref"] = ref
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is False
    assert set(codes(report, "error")) & {"invalid_authorization_ref", "secret_like_value"}


def test_owner_inputs_have_no_defaults():
    inventory = fixture_inventory()
    inventory["harmonizer_source"] = None
    inventory["toolchain"]["rust_toolchain"] = None
    asset(inventory, "nurec_scenes")["files"] = []
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is True
    assert report["status"] == "blocked"
    owner = sorted(i["path"] for i in report["issues"] if i["code"] == "owner_input_missing")
    assert owner == ["assets[3].files", "harmonizer_source"]
    assert "toolchain_unset" in codes(report, "blocking")


@pytest.mark.parametrize("count", [4, 6, 441])
def test_pilot_scene_list_is_exactly_five(count):
    inventory = fixture_inventory()
    rows = synthetic_scenes()
    scenes = [dict(rows[i % 5], uuid=f"00000000-0000-4000-8000-{i:012d}",
                   scene_id=f"clipgt-20000000-0000-4000-8000-{i:012d}",
                   path=f"sample_set/x/{i}.usdz") for i in range(count)]
    asset(inventory, "nurec_scenes")["files"] = scenes
    assert "pilot_scene_count" in codes(validate_runtime_inventory(inventory), "error")


def test_duplicate_scene_uuid_rejected():
    inventory = fixture_inventory()
    files = asset(inventory, "nurec_scenes")["files"]
    files[1]["uuid"] = files[0]["uuid"]
    assert "duplicate_scene" in codes(validate_runtime_inventory(inventory), "error")


def test_repository_mismatch_and_missing_roles_rejected():
    inventory = fixture_inventory()
    asset(inventory, "vlm_config")["repository"] = "nvidia/Cosmos-Reason1-7B"
    assert "repository_mismatch" in codes(validate_runtime_inventory(inventory), "error")
    inventory = fixture_inventory()
    inventory["assets"] = [a for a in inventory["assets"] if a["role"] != "vlm_config"]
    report = validate_runtime_inventory(inventory)
    assert "missing_role" in codes(report, "error")
    inventory = fixture_inventory()
    inventory["images"] = inventory["images"][:2]
    assert "missing_role" in codes(validate_runtime_inventory(inventory), "error")


def test_wrong_platform_rejected():
    inventory = fixture_inventory()
    image(inventory, "renderer")["platform"] = "linux/arm64"
    assert "invalid_platform" in codes(validate_runtime_inventory(inventory), "error")


@pytest.mark.parametrize("mutate", [
    lambda inv: inv.update(extra=1),
    lambda inv: inv.pop("disk"),
    lambda inv: inv.update(schema_version=2),
    lambda inv: inv.update(artifact_kind="other"),
    lambda inv: inv.update(scope="pretend"),
    lambda inv: inv["images"][0].update(unexpected=True),
    lambda inv: inv["assets"][0]["files"][0].update(extra="x"),
])
def test_unknown_or_missing_fields_fail_closed(mutate):
    inventory = fixture_inventory()
    mutate(inventory)
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is False
    assert report["status"] == "blocked"


@pytest.mark.parametrize("garbage", [None, [], "x", 3, {"artifact_kind": []}])
def test_garbage_input_never_raises(garbage):
    report = validate_runtime_inventory(garbage)
    assert report["valid"] is False
    assert report["input_sha256"] is None or len(report["input_sha256"]) == 64


def test_secret_like_value_anywhere_rejected():
    inventory = fixture_inventory()
    inventory["host_paths"]["teacher_snapshot"] = "/data/hf_" + "Ab1" * 12
    report = validate_runtime_inventory(inventory)
    assert "secret_like_value" in codes(report, "error")


def test_report_is_deterministic_and_input_not_mutated():
    inventory = fixture_inventory()
    before = deepcopy(inventory)
    assert validate_runtime_inventory(inventory) == validate_runtime_inventory(inventory)
    assert inventory == before


# --- experiments/S2_SETUP_DRAFT.json (R2 S1-S13) ----------------------------------------


def draft():
    return read_json_bytes((ROOT / "experiments/S2_SETUP_DRAFT.json").read_bytes())


def test_setup_draft_stays_draft_and_unauthorized():
    data = draft()
    assert data["status"] == "draft"
    assert data["cloud_launch_authorized"] is False
    assert data["scientific_admission"] is False


def test_setup_draft_camera_mismatch_unresolved_pending_huy():
    camera = draft()["runtime"]["camera_mismatch"]
    assert camera["status"] == "unresolved_pending_huy"
    assert camera["renderer_camera_count"] == 6
    assert camera["driver_camera_count"] == 4
    assert camera["decision_ref"] is None
    assert camera["selected_option"] is None


def test_setup_draft_records_cosmos_reason2_as_required():
    data = draft()
    vlm = next(a for a in data["assets"] if a["component"] == "vlm_config_processor_tokenizer")
    assert vlm["requirement"] == "required"
    assert vlm["repository"] == "https://huggingface.co/nvidia/Cosmos-Reason2-8B"
    assert vlm["evidence_ref"] == (
        "evidence/contribution-review-20261105-v1/hf-public-configs/a15_config.json")
    config = read_json_bytes((ROOT / vlm["evidence_ref"]).read_bytes())
    assert config["vlm_name_or_path"] == "nvidia/Cosmos-Reason2-8B"
    inventory_vlm = asset(data["runtime_inventory"], "vlm_config")
    assert inventory_vlm["repository"] == "nvidia/Cosmos-Reason2-8B"
    assert inventory_vlm["revision"] == COSMOS_REV


def test_setup_draft_disk_default_is_labelled_with_derivation():
    disk = draft()["runtime"]["disk_request"]
    assert disk["minimum_gb"] == 300
    assert disk["basis"].startswith("agent technical default (D-018)")
    assert "120-170 GiB" in disk["derivation"]


def test_setup_draft_runtime_inventory_is_valid_but_blocked_without_errors():
    report = validate_runtime_inventory(draft()["runtime_inventory"])
    assert report["valid"] is True, report["issues"]
    assert report["status"] == "blocked"
    assert codes(report, "error") == []
    blocking = set(codes(report, "blocking"))
    assert {"authorization_missing", "owner_input_missing", "toolchain_unset",
            "host_path_unset", "disk_capacity_unset", "image_reference_unset",
            "source_bundle_unset"} <= blocking
    assert "uv_archive_unverified" not in blocking  # the published checksum is recorded


def test_setup_draft_legacy_entries_agree_with_runtime_inventory():
    data = draft()
    inventory = data["runtime_inventory"]
    renderer = next(i for i in data["images"] if i["component"] == "renderer")
    assert renderer["reference_candidate"] == image(inventory, "renderer")["reference"] == NRE_REF
    teacher = next(a for a in data["assets"] if a["component"] == "teacher_weights_and_config")
    inventory_teacher = asset(inventory, "teacher")
    assert teacher["revision_candidate"] == inventory_teacher["revision"] == TEACHER_REVISION
    assert [(f["path"], f["size_bytes"], f["sha256"]) for f in teacher["files"]] == [
        (f["path"], f["size_bytes"], f["sha256"]) for f in inventory_teacher["files"]]
    assert sum(f["size_bytes"] for f in teacher["files"]) == 22_157_316_944
    harmonizer = next(a for a in data["assets"] if a["component"] == "harmonizer")
    assert harmonizer["revision_candidate"] == asset(inventory, "harmonizer")["revision"]
    assert harmonizer["content_sha256"] == HARMONIZER_SHA
    base = next(i for i in data["images"] if i["component"] == "alpasim_base")
    assert base["kind"] == "local_build"
    assert base["mutable_inputs"] == image(inventory, "alpasim_base")["mutable_inputs"]


# --- review A2 regressions --------------------------------------------------------------


@pytest.mark.parametrize("name", ["with space.json", "x;rm -rf $(touch pwned).json",
                                  "quote'\"`|&<>*?.json", "dir with space/a b.json",
                                  ".gitattributes"])
def test_metacharacter_filenames_are_permitted_literally(name):
    """Review A2-F10 (CPU-09 first failures): permitted, not blanket-rejected."""
    inventory = fixture_inventory()
    asset(inventory, "processor")["files"][0]["path"] = name
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is True, report["issues"]
    assert "unsafe_relative_path" not in codes(report)


@pytest.mark.parametrize("mutate", [
    lambda inv: inv["images"][0].update(role=[]),
    lambda inv: inv["images"][0].update(role={}),
    lambda inv: inv["assets"][0].update(role=[]),
    lambda inv: inv["assets"][2]["files"][0].update(path=[]),
    lambda inv: inv["assets"][3]["files"][0].update(scene_id=[], uuid={}),
    lambda inv: inv["disk"]["components"][0].update(name=[]),
    lambda inv: inv["host_paths"].update(checkout=[]),
    lambda inv: inv.update(harmonizer_source=[]),
    lambda inv: inv["sources"].update(alpasim=[]),
])
def test_unhashable_json_values_fail_closed_without_raising(mutate):
    """Review A2-F7: a JSON-valid list/object where a string belongs is an issue, not a crash."""
    inventory = fixture_inventory()
    mutate(inventory)
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is False
    assert report["status"] == "blocked"


def test_clean_absolute_allows_dash_components_but_not_traversal():
    """Absolute rule paths start with '/', so a '-x' component cannot become an option; real
    scratch roots look like /private/tmp/claude-501/-Users-.../x."""
    from gate.s2_runtime import is_clean_absolute

    assert is_clean_absolute("/private/tmp/-Users-x/a b;c.json") is True
    for bad in ("relative", "/", "/a/../b", "/a//b", "/a/./b", "/a/\nb", "/a/b\x00", 3, None):
        assert is_clean_absolute(bad) is False, bad


# --- F3 guest runtime (pinned uv, managed CPython, sealed source bundle) ----------------


def test_guest_runtime_pins_are_the_release_metadata_literals():
    assert UV_VERSION == "0.12.24"
    assert UV_ARCHIVE_URL == UV_URL
    assert UV_ARCHIVE_SHA256 == UV_SHA
    assert UV_ARCHIVE_SIZE_BYTES == 17_249_236
    assert PYTHON_REQUEST == "cpython-3.12.11-linux-x86_64-gnu"
    # R2 A6 requires uv >= 0.9.17; 0.12.24 compares above it numerically.
    assert tuple(map(int, UV_VERSION.split("."))) > (0, 9, 17)


def test_locked_runtime_versions_match_the_lock_and_python_pin():
    import tomllib

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = {p["name"]: p["version"] for p in lock["package"]}
    assert [(name, versions[name]) for name, _, _ in LOCKED_RUNTIME] == [
        ("pyyaml", "6.0.3"), ("pyarrow", "25.0.1"), ("protobuf", "7.36.2")]
    assert [version for _, _, version in LOCKED_RUNTIME] == ["6.0.3", "25.0.1", "7.36.2"]
    assert lock["requires-python"] == "==3.12.*"
    assert (ROOT / ".python-version").read_text().strip() == "3.12"
    root = next(p for p in lock["package"] if p["name"] == "physloop-gate")
    assert sorted(d["name"] for d in root["dependencies"]) == [
        "numpy", "protobuf", "pyarrow", "pyyaml", "scipy"]


@pytest.mark.parametrize("field, value, code", [
    ("version", "0.13.5", "pin_mismatch"),
    ("target", "aarch64-unknown-linux-gnu", "pin_mismatch"),
    ("archive_url", UV_URL.replace("0.12.24", "latest"), "pin_mismatch"),
    ("archive_sha256", "0" * 64, "pin_mismatch"),
    ("archive_sha256", "B" * 64, "invalid_guest_runtime"),
    ("archive_size_bytes", 17_249_235, "pin_mismatch"),
    ("archive_size_bytes", 0, "invalid_guest_runtime"),
])
def test_uv_pin_mismatches_are_errors(field, value, code):
    inventory = fixture_inventory()
    inventory["guest_runtime"]["uv"][field] = value
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is False and code in codes(report, "error")


def test_unverified_uv_checksum_blocks_but_is_not_a_structural_error():
    inventory = fixture_inventory()
    inventory["guest_runtime"]["uv"]["archive_sha256"] = "unverified"
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is True and report["status"] == "blocked"
    assert codes(report, "blocking") == ["uv_archive_unverified"]


def test_python_request_is_pinned_to_the_patch_release():
    inventory = fixture_inventory()
    inventory["guest_runtime"]["python"]["request"] = "3.12"
    assert "pin_mismatch" in codes(validate_runtime_inventory(inventory), "error")


def test_unset_bundle_blocks_and_malformed_bundle_is_an_error():
    inventory = fixture_inventory()
    inventory["guest_runtime"]["source_bundle"]["bundle_sha256"] = None
    report = validate_runtime_inventory(inventory)
    assert report["valid"] is True and codes(report, "blocking") == ["source_bundle_unset"]
    inventory["guest_runtime"]["source_bundle"]["archive_sha256"] = "xyz"
    assert "invalid_guest_runtime" in codes(validate_runtime_inventory(inventory), "error")
    inventory = fixture_inventory()
    inventory["guest_runtime"]["source_bundle"]["archive_size_bytes"] = "unverified"
    assert "invalid_guest_runtime" in codes(validate_runtime_inventory(inventory), "error")


@pytest.mark.parametrize("mutate", [
    lambda inv: inv.pop("guest_runtime"),
    lambda inv: inv["guest_runtime"].update(extra=1),
    lambda inv: inv["guest_runtime"]["uv"].pop("archive_url"),
    lambda inv: inv["guest_runtime"]["python"].update(version="3.12.11"),
    lambda inv: inv["guest_runtime"]["source_bundle"].update(path="/x"),
])
def test_guest_runtime_fields_are_exact(mutate):
    inventory = fixture_inventory()
    mutate(inventory)
    assert validate_runtime_inventory(inventory)["valid"] is False


@pytest.mark.parametrize("name", ["evidence_dir", "download_scratch", "data_root", "hf_home"])
def test_no_other_root_may_sit_inside_the_sealed_source_tree(name):
    inventory = fixture_inventory()
    inventory["host_paths"][name] = "/opt/physloop/inside"
    report = validate_runtime_inventory(inventory)
    assert "host_path_overlap" in codes(report, "error")
    assert f"host_paths.{name}" in {i["path"] for i in report["issues"]}
    inventory = fixture_inventory()
    inventory["host_paths"]["physloop_root"] = "/root/physloop"  # inside home is fine
    assert validate_runtime_inventory(inventory)["status"] == "passed"


def manifest(**overrides):
    base = {"artifact_kind": "s2_source_manifest", "schema_version": 1, "scope": "synthetic",
            "producer": {"module": "scripts.s2_prepare_bundle", "source_sha256": None},
            "inputs": {"include": ["gate"]}, "units": {"size_bytes": "bytes"},
            "files": [{"path": ".python-version", "size_bytes": 5, "sha256": "a" * 64},
                      {"path": "gate/__init__.py", "size_bytes": 0, "sha256": "e" * 64}],
            "issues": []}
    base.update(overrides)
    return base


def test_source_manifest_known_answer_and_archive_name():
    assert validate_source_manifest(manifest()) == []
    assert source_bundle_archive_name("b" * 64) == f"physloop-source-{'b' * 64}.tar"
    with pytest.raises(ValueError):
        source_bundle_archive_name("../x")


@pytest.mark.parametrize("files", [
    [],
    [{"path": "../x", "size_bytes": 1, "sha256": "a" * 64}],
    [{"path": "/etc/passwd", "size_bytes": 1, "sha256": "a" * 64}],
    [{"path": "gate/a b.py", "size_bytes": 1, "sha256": "a" * 64}],
    [{"path": "gate/-x.py", "size_bytes": 1, "sha256": "a" * 64}],
    [{"path": "physloop-source-manifest.json", "size_bytes": 1, "sha256": "a" * 64}],
    [{"path": ".venv/bin/python", "size_bytes": 1, "sha256": "a" * 64}],
    [{"path": "gate/__pycache__/x.pyc", "size_bytes": 1, "sha256": "a" * 64}],
    [{"path": "gate/b.py", "size_bytes": 1, "sha256": "a" * 64},
     {"path": "gate/a.py", "size_bytes": 1, "sha256": "a" * 64}],
    [{"path": "gate/A.py", "size_bytes": 1, "sha256": "a" * 64},
     {"path": "gate/a.py", "size_bytes": 1, "sha256": "a" * 64}],
    [{"path": "gate/a.py", "size_bytes": -1, "sha256": "a" * 64}],
    [{"path": "gate/a.py", "size_bytes": 1, "sha256": "A" * 64}],
    [{"path": "gate/a.py", "size_bytes": 1, "sha256": "a" * 64, "mode": 420}],
])
def test_source_manifest_negative_controls(files):
    assert validate_source_manifest(manifest(files=files))


@pytest.mark.parametrize("bad", [None, [], manifest(issues=[{"code": "x"}]),
                                 manifest(schema_version=2), manifest(scope="real"),
                                 dict(manifest(), extra=1)])
def test_source_manifest_header_fails_closed(bad):
    assert validate_source_manifest(bad)


def test_setup_draft_records_the_guest_runtime_pins():
    data = draft()
    assert data["status"] == "draft" and data["cloud_launch_authorized"] is False
    section = data["guest_runtime"]
    inventory = data["runtime_inventory"]["guest_runtime"]
    assert inventory["uv"] == guest_runtime()["uv"]
    assert inventory["python"] == {"request": PYTHON_REQUEST}
    assert inventory["source_bundle"] == {"bundle_sha256": None, "archive_sha256": None,
                                          "archive_size_bytes": None}
    assert section["uv"]["archive_sha256"] == UV_ARCHIVE_SHA256 == UV_SHA
    assert section["uv"]["archive_size_bytes"] == UV_ARCHIVE_SIZE_BYTES
    assert section["uv"]["version"] == UV_VERSION and section["python"]["version"] == "3.12.11"
    assert section["locked_runtime"] == {d: v for d, _, v in LOCKED_RUNTIME}
    assert section["basis"].startswith("agent technical default (D-018)")
    assert "compute_cap" in section["host_fit_open_item"]
    # F3-1/F3-2 review: the draft states the open 3.10 item and the --touch unpack.
    assert "datetime.UTC" in section["phase_a_interpreter_open_item"]
    assert "--touch" in section["bundle_unpack"] and "stale_bytecode" in section["bundle_unpack"]
