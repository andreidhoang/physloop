"""S2.4 bit-identical re-export control for policy B: plan stages and manifest verdict.

Tickets: plan ``02`` section 3 S2.4 [C] (the full focused matrix contains the BF16 teacher and
its bit-identical re-export control; G3·R1, G7) and the completeness-audit gap "S2.4
producers" (no procedure produced the re-export snapshot for policy B; only comparators and a
launch warning, ``gate/s2_launch.py:566-569``). Agent-authored evaluator engineering under
D-018/D-024. The control is policy B's admission precondition, not a scientific result.

Procedure (lane-r3 section 3 steps 4 and 7; pinned AlpaGym ``972d160``, its converters vendored
byte-identically with LICENSE and provenance in ``tests/fixtures/s2_reexport/``):

1. ``verify`` (verify-only stages): the AlpaGym checkout is at ``972d160`` with the pinned
   ``uv.lock`` and both converter scripts hash to their pinned bytes; the teacher snapshot holds
   the inventory's files with their SHA-256; the gated Cosmos-Reason2-8B config/processor files
   sit in the HF cache at the registered revision. These are produced elsewhere and only
   verified here: the AlpaGym frozen environment by ``teacher_rl.ops`` phase ``alpagym``
   (G1-G9), the teacher snapshot by ``gate.s2_bootstrap`` E1 or ``teacher_rl.ops`` M1, the
   Cosmos-Reason2-8B cache by ``gate.s2_bootstrap`` E2a or ``teacher_rl.ops`` M2.
2. ``convert``: ``C1`` runs ``convert_release_to_alpagym_checkpoint.py --input <teacher> --output
   <gym_bundle> --vlm-name-or-path nvidia/Cosmos-Reason2-8B`` (rewrites ``config.json``,
   symlinks the safetensors shards and index, builds the expanded processor and copies the
   generation config from the gated repository, validates; :38, :69-86, :89-100, :136-188).
   ``C2`` runs ``convert_alpagym_checkpoint_to_inference.py --input <gym_bundle> --output
   <reexport_dir>`` (rewrites ``config.json`` to ``alpamayo1_5``, loads each indexed shard
   through the bundle's symlinks and saves it under ``to_inference_weight_key`` names with
   ``metadata={"format": "pt"}``, rewrites the index, copies auxiliary files; :74-101, :120-170).
   ``to_inference_weight_key`` is the identity on ``vlm.``-nested release names, so the
   re-export must carry exactly the release's tensor names. The HF id, not a local path, is
   passed so ``config.json`` names the same ``vlm_name_or_path`` for A and B. Neither stage
   passes ``--overwrite``: a non-empty destination makes the converter raise
   ``FileExistsError`` (``alpamayo.utils.checkpoint_utils.prepare_output_dir``, recipes
   ``1d99fc5`` :67-92), so earlier files are never silently replaced.
3. ``control``: ``X1`` runs ``scripts/s2_reexport.py compare`` through the verified guest
   environment (``gate.s2_bootstrap.GUEST_PYTHON``): ``gate.tensor_manifest`` manifests of the
   release snapshot and the re-export (no-follow, regular files only, by the shard index) and
   ``reexport_control``. The gym bundle is never manifested: its shards are symlinks by design.
   X1's expected outputs (and idempotency check) also carry the size and SHA-256 of every
   inventory teacher file, so a release snapshot that drifted after V3, or a control-only plan
   without V3, cannot certify B against an unpinned A. That check runs after the compare; a
   change during the compare itself is out of its reach.

Verdict (``reexport_control``): ``passed`` only when ``gate.tensor_manifest.compare_manifests``
passes, the release holds at least one tensor (``empty_tensor_set`` otherwise) and every tensor
name, dtype, shape and per-tensor SHA-256 (build-spec section 2 rule 11) is identical; otherwise ``blocked`` with the differing tensors listed (``only_in_release``,
``only_in_reexport``, ``dtype_changed``, ``shape_changed``, ``bytes_changed`` and per-tensor
detail). File hashes, header bytes and ``__metadata__`` are reported separately and never enter
the verdict; byte-identical files give a ``reexport_files_identical_to_release`` warning (B
would not be its own export: the launch-side ``identical_policy_snapshots`` smell).

Plan kind. The single guest runner executes only kinds registered in
``gate.s2_bootstrap.PLAN_KINDS``. This compiler produces ``s2_reexport_plan`` artifacts with
compiler identity ``COMPILER``; ``PLAN_KIND_SPEC`` is registered there under ``s2_reexport``
(a literal entry, since that module cannot import this one), so ``validate_plan`` accepts them
and ``scripts/s2_guest_bootstrap.py`` recompiles them through
``build_reexport_plan(inventory, action_record, phases=...)``; the tests pin the two equal. Like ``teacher_rl.ops`` it
reuses ``gate.s2_bootstrap``'s stage builder ``_Builder``, rule constructors, content validator
``_plan_content_issues`` and seal ``_seal``; ``tests/test_s2_reexport.py`` pins those names.

Gates: a plan compiles all-or-nothing and refuses (``compiled: false``, no stages) on an
invalid inventory or action record, an unauthorized phase, an unset host path, overlapping
data paths, an unresolved Cosmos-Reason2-8B revision (an admission-time input, never
defaulted), a missing ``terms:teacher`` or ``terms:cosmos_reason2`` records reference (the
converter's processor dependency is the gated Cosmos-Reason2-8B repository; using it needs its
recorded terms even though nothing is downloaded here), or anything credential-like. Secrets:
none; every stage runs offline (``HF_HUB_OFFLINE=1``).

Verification level per function (build-spec section 1):

- ``validate_reexport_inventory``, ``build_reexport_plan``: L1 compile/refusal logic with literal
  known answers; every emitted command and rule is L2 (pinned converter source, vendored and
  hash-checked; argparse flags and output-directory policy checked against it). Executing any
  stage is L3 (``execution_status: qualify_on_host``): CPU tests never run a converter.
- ``reexport_control``: L1 over hand-built safetensors bytes; on an actual release and re-export
  the verdict is L3 and qualifies only those bytes.

Agent technical defaults (D-018): ``C1`` timeout ``import`` (900 s, as ``teacher_rl.ops`` C1),
``C2`` ``build`` (7,200 s: it loads and rewrites every shard), ``X1`` ``sync`` (3,600 s: it
hashes both checkpoints); the control output directory name ``s2-reexport-control`` under
``evidence_dir``; teacher file hashes must be known (``unverified`` is refused, since the
release snapshot is the control's reference), Cosmos-Reason2-8B file sizes and hashes may be
``unverified`` (they shape tokenizer files, never tensors).

This module never runs a process, opens a network connection, downloads, reads a credential,
accepts terms, imports ``teacher_rl`` or model code, chooses an owner value or decides whether
a difference is scientifically acceptable.
"""

import hashlib
import re
from pathlib import Path

from gate.s2_bootstrap import (
    GUEST_PYTHON,
    _Builder,
    _file_rule,
    _hf_cache_path,
    _issue,
    _plan_content_issues,
    _seal,
)
from gate.s2_bootstrap import UNITS as PLAN_UNITS
from gate.s2_identity import identity_hash
from gate.s2_runtime import (
    is_placeholder,
    is_portable_absolute,
    is_safe_relative,
    paths_overlap,
    secret_like,
)
from gate.tensor_manifest import compare_manifests

KIND = "s2_reexport"
INVENTORY_KIND = "s2_reexport_inventory"
PLAN_KIND = "s2_reexport_plan"
RECORD_KIND = "s2_reexport_action_record"
REPORT_KIND = "s2_reexport_inventory_report"
CONTROL_KIND = "s2_reexport_control"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
COMPILER = {"inventory_kind": KIND, "function": "gate.s2_reexport.build_reexport_plan"}
# Registered as gate.s2_bootstrap.PLAN_KINDS["s2_reexport"] (tests pin the two equal).
PLAN_KIND_SPEC = {
    "inventory_artifact_kind": INVENTORY_KIND,
    "plan_artifact_kind": PLAN_KIND,
    "function": COMPILER["function"],
    "inputs": ("inventory", "action_record"),
    "checkout_root": "alpagym_checkout",
    "extra_fields": (),
}
PHASES = ("verify", "convert", "control")
AUTHORIZATION_KEYS = ("host_setup",)
HOST_PATHS = ("physloop_root", "home", "alpagym_checkout", "teacher_snapshot", "gym_bundle",
              "reexport_dir", "hf_home", "evidence_dir")
# Pairwise disjoint; physloop_root is disjoint from all but evidence_dir; home may contain any.
DATA_PATHS = ("alpagym_checkout", "teacher_snapshot", "gym_bundle", "reexport_dir", "hf_home",
              "evidence_dir")

# Pins, duplicated as literals so the S2 guest import graph stays free of teacher_rl;
# tests/test_s2_reexport.py asserts equality with teacher_rl.pins.
ALPAGYM_REPOSITORY = "https://github.com/NVlabs/alpagym"
ALPAGYM_COMMIT = "972d160eed0e23d388497851504a3a233fec5879"
ALPAGYM_UV_LOCK_SHA256 = "5148f9b0f5f20b1cc3753dec78e165491c881b0217d095bba9a74d017fcd674e"
TEACHER_REPOSITORY = "nvidia/Alpamayo-1.5-10B"
TEACHER_REVISION = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
COSMOS_REASON2_REPOSITORY = "nvidia/Cosmos-Reason2-8B"
# AlpaGym 972d160 converter scripts: path, size, sha256, git blob (fixture provenance.json).
CONVERTERS = {
    "release_to_gym": {
        "path": "packages/policies/alpamayo_r1/scripts/convert_release_to_alpagym_checkpoint.py",
        "size_bytes": 7472,
        "sha256": "ec926d0dd345711663db2493753de492b1a7367cb640a108a87439da776e25c8",
        "git_blob_sha": "67c4b49fb10a66f0bb55c95993c7e7b483a091a0",
    },
    "gym_to_inference": {
        "path": "packages/policies/alpamayo_r1/scripts/convert_alpagym_checkpoint_to_inference.py",
        "size_bytes": 7213,
        "sha256": "936a290377955fe28ec8db4ed28e7622fe3d52450c8bce439255daf0b5f8b38f",
        "git_blob_sha": "a15800e8d7ca9f0690cf4c26eb993d3edc9951b8",
    },
}
INDEX_NAME = "model.safetensors.index.json"
# validate_checkpoint in both converters (:131-133 and :115-117).
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
GYM_MODEL_TYPE = "alpamayo_reasoning_vla_expert"  # release converter :73, :107-108
INFERENCE_MODEL_TYPE = "alpamayo1_5"  # inference converter :78, :108-109
CONTROL_DIR_NAME = "s2-reexport-control"
CONTROL_FILES = ("release.manifest.json", "reexport.manifest.json", "control.json")
CONTROL_PASSED_PATTERN = '"control":"passed"'
CRITERION = ("tensor names, dtypes, shapes and per-tensor sha256 identical (build-spec section "
             "2 rule 11); file hashes, headers and __metadata__ reported separately, never "
             "part of the verdict")
UNVERIFIED = "unverified"

_R3 = "docs/research/2026-11-08-codebase-build/lane-r3-s4-upstream.md"
_TOP = {"artifact_kind", "schema_version", "scope", "host_paths", "images", "assets"}
_ASSET_FIELDS = {"role", "repository", "repo_type", "revision", "authorization_ref", "files"}
_FILE_FIELDS = {"path", "size_bytes", "sha256"}
_RECORD_FIELDS = {"artifact_kind", "schema_version", "action_record_ref", "authorized_phases",
                  "authorizations"}
_ASSET_PINS = {"teacher": (TEACHER_REPOSITORY, TEACHER_REVISION),
               "cosmos_reason2": (COSMOS_REASON2_REPOSITORY, None)}
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


def _producer():
    return {"module": "gate.s2_reexport", "source_sha256": _source_sha256()}


def _strings(value, path):
    if type(value) is str:
        yield path, value
    elif type(value) is dict:
        for key, item in value.items():
            yield from _strings(key, f"{path}.<key>")
            yield from _strings(item, f"{path}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def _bad_ref(value):
    return type(value) is not str or not value.strip() or is_placeholder(value) or (
        secret_like(value))


# --- inventory ----------------------------------------------------------------------------


def _check_paths(paths, issues):
    if type(paths) is not dict or set(paths) != set(HOST_PATHS):
        issues.append(_issue("invalid_inventory", "inventory.host_paths",
                             f"host_paths keys are exactly {list(HOST_PATHS)}."))
        return
    for name in HOST_PATHS:
        if paths[name] is not None and not is_portable_absolute(paths[name]):
            issues.append(_issue("invalid_path", f"inventory.host_paths.{name}",
                                 "A portable absolute path or null (unset)."))
    set_paths = {n: p for n, p in paths.items() if is_portable_absolute(p)}
    for index, first in enumerate(DATA_PATHS):
        for second in DATA_PATHS[index + 1:]:
            if first in set_paths and second in set_paths and paths_overlap(
                    set_paths[first], set_paths[second]):
                issues.append(_issue("overlapping_paths", f"inventory.host_paths.{second}",
                                     f"{first} and {second} must be disjoint (a converter "
                                     "output must never land in an input or another output)."))
    for data in DATA_PATHS[:-1]:  # evidence_dir may sit under the project tree
        if "physloop_root" in set_paths and data in set_paths and paths_overlap(
                set_paths["physloop_root"], set_paths[data]):
            issues.append(_issue("overlapping_paths", f"inventory.host_paths.{data}",
                                 f"{data} must not contain or lie inside physloop_root."))


def _check_files(files, path, issues, *, hashes_required):
    if type(files) is not list or not files:
        issues.append(_issue("asset_files_unlisted", path, "List every file (nonempty)."))
        return
    seen = set()
    for index, item in enumerate(files):
        where = f"{path}[{index}]"
        if type(item) is not dict or set(item) != _FILE_FIELDS:
            issues.append(_issue("invalid_inventory", where,
                                 f"File fields are exactly {sorted(_FILE_FIELDS)}."))
            continue
        if not is_safe_relative(item["path"]) or item["path"] in seen:
            issues.append(_issue("invalid_inventory", f"{where}.path",
                                 "A unique safe relative path."))
        seen.add(item["path"])
        size, sha = item["size_bytes"], item["sha256"]
        size_ok = (type(size) is int and size >= 0) or (size == UNVERIFIED and not hashes_required)
        sha_ok = (type(sha) is str and _SHA256.fullmatch(sha) is not None) or (
            sha == UNVERIFIED and not hashes_required)
        if not (size_ok and sha_ok):
            code = ("teacher_hash_unverified" if hashes_required and UNVERIFIED in (size, sha)
                    else "invalid_inventory")
            issues.append(_issue(code, where, "size_bytes and sha256 are known values"
                                 + ("" if hashes_required else " or 'unverified'") + "."))


def _check_assets(assets, issues):
    if type(assets) is not list or len(assets) != 2 or not all(
            type(a) is dict and set(a) == _ASSET_FIELDS for a in assets) or {
            a["role"] for a in assets} != set(_ASSET_PINS):
        issues.append(_issue("invalid_inventory", "inventory.assets",
                             "Exactly the teacher and cosmos_reason2 assets, each with fields "
                             f"{sorted(_ASSET_FIELDS)}."))
        return
    for index, asset in enumerate(assets):
        where = f"inventory.assets[{index}]"
        role = asset["role"]
        repository, revision = _ASSET_PINS[role]
        if asset["repository"] != repository or asset["repo_type"] != "model":
            issues.append(_issue("unpinned_asset", f"{where}.repository",
                                 f"{role} is the model repository {repository}."))
        if revision is not None and asset["revision"] != revision:
            issues.append(_issue("unpinned_asset", f"{where}.revision",
                                 f"{role} is pinned at revision {revision}."))
        if revision is None and asset["revision"] is not None and (
                type(asset["revision"]) is not str or not _COMMIT.fullmatch(asset["revision"])):
            issues.append(_issue("invalid_inventory", f"{where}.revision",
                                 "A 40-hex commit, or null until admission."))
        ref = asset["authorization_ref"]
        if ref is not None and _bad_ref(ref):
            issues.append(_issue("invalid_inventory", f"{where}.authorization_ref",
                                 "A recorded, non-placeholder, non-secret reference or null."))
        teacher = role == "teacher"
        _check_files(asset["files"], f"{where}.files", issues, hashes_required=teacher)
        if teacher and type(asset["files"]) is list:
            names = {f.get("path") for f in asset["files"] if type(f) is dict}
            for required in ("config.json", INDEX_NAME):
                if required not in names:
                    issues.append(_issue("teacher_file_missing", f"{where}.files",
                                         f"The teacher file list includes {required}."))
            if not any(type(n) is str and n.endswith(".safetensors") for n in names):
                issues.append(_issue("teacher_file_missing", f"{where}.files",
                                     "The teacher file list includes its safetensors shards."))


def validate_reexport_inventory(inventory):
    """Validate an ``s2_reexport_inventory`` v1; deterministic report (L1; pins L2)."""
    issues = []
    try:
        input_hash = identity_hash(inventory)
    except (TypeError, ValueError):
        input_hash = None
    report = {"artifact_kind": REPORT_KIND, "schema_version": SCHEMA_VERSION, "scope": None,
              "producer": _producer(), "input_sha256": input_hash, "units": {"bytes": "bytes"},
              "level": "L1", "valid": False, "status": "blocked", "issues": issues}
    if input_hash is None or type(inventory) is not dict or set(inventory) != _TOP:
        issues.append(_issue("invalid_inventory", "inventory",
                             f"Inventory fields are exactly {sorted(_TOP)} (strict JSON)."))
        return report
    if inventory["artifact_kind"] != INVENTORY_KIND or type(
            inventory["schema_version"]) is not int or inventory["schema_version"] != 1:
        issues.append(_issue("invalid_inventory", "inventory.schema_version",
                             f"Only {INVENTORY_KIND} schema_version 1 is known."))
    if inventory["scope"] in SCOPES:
        report["scope"] = inventory["scope"]
    else:
        issues.append(_issue("invalid_scope", "inventory.scope", f"one of {list(SCOPES)}."))
    for path, value in _strings(inventory, "inventory"):
        if secret_like(value):
            issues.append(_issue("secret_like_value", path, "Never embed credentials."))
    if inventory["images"] != []:
        issues.append(_issue("invalid_inventory", "inventory.images",
                             "The re-export uses no container image (images is [])."))
    _check_paths(inventory["host_paths"], issues)
    _check_assets(inventory["assets"], issues)
    issues.sort(key=lambda i: (i["path"], i["code"], i["message"]))
    report["valid"] = not any(i["severity"] == "error" for i in issues)
    report["status"] = "passed" if report["valid"] else "blocked"
    return report


def _validate_record(record):
    if type(record) is not dict or set(record) != _RECORD_FIELDS:
        return [_issue("invalid_action_record", "action_record",
                       f"Fields are exactly {sorted(_RECORD_FIELDS)}.")]
    issues = []
    if record["artifact_kind"] != RECORD_KIND or type(record["schema_version"]) is not int or (
            record["schema_version"] != 1):
        issues.append(_issue("invalid_action_record", "action_record.schema_version",
                             f"Only {RECORD_KIND} schema_version 1 is known."))
    if _bad_ref(record["action_record_ref"]):
        issues.append(_issue("invalid_action_record", "action_record.action_record_ref",
                             "Recorded, non-placeholder, non-secret reference required."))
    phases = record["authorized_phases"]
    if type(phases) is not list or not all(type(p) is str and p in PHASES for p in phases):
        issues.append(_issue("invalid_action_record", "action_record.authorized_phases",
                             f"authorized_phases is a list drawn from {list(PHASES)}."))
    auths = record["authorizations"]
    if type(auths) is not dict:
        issues.append(_issue("invalid_action_record", "action_record.authorizations",
                             "authorizations is an object."))
        return issues
    for key in sorted(auths):
        if key not in AUTHORIZATION_KEYS:
            issues.append(_issue("invalid_action_record", f"action_record.authorizations.{key}",
                                 f"Unknown key; known {list(AUTHORIZATION_KEYS)}."))
        elif _bad_ref(auths[key]):
            issues.append(_issue("invalid_action_record", f"action_record.authorizations.{key}",
                                 "Recorded, non-placeholder, non-secret reference required."))
    return issues


# --- plan ---------------------------------------------------------------------------------


def _normalize_phases(phases):
    if type(phases) not in (list, tuple) or not phases or len(set(phases)) != len(phases) or (
            not all(type(p) is str and p in PHASES for p in phases)):
        return None
    return [p for p in PHASES if p in phases]


def _needed_inputs(b, phases):
    for name in HOST_PATHS:
        if b.paths[name] is None:
            b.refuse("host_path_unset", f"inventory.host_paths.{name}",
                     "Every host path is needed to compile a plan.")
    if ("verify" in phases or "convert" in phases) and b.assets["cosmos_reason2"][
            "revision"] is None:
        b.refuse("revision_unresolved", "inventory.assets.cosmos_reason2.revision",
                 "Cosmos-Reason2-8B revision is an admission-time input (no default).")


def _regex(root, path, pattern):
    return {"kind": "file_regex", "root": root, "path": path, "pattern": pattern}


def _config_rules(root, directory, model_type):
    path = f"{directory}/config.json"
    return [_regex(root, path, r'"model_type":\s*"' + re.escape(model_type) + '"'),
            _regex(root, path, r'"vlm_name_or_path":\s*"'
                   + re.escape(COSMOS_REASON2_REPOSITORY) + '"')]


def _exists(root, path):
    return _file_rule(root, path, None, None)


def _phase_verify(b):
    p, co = b.paths, b.paths["alpagym_checkout"]
    head = _regex("alpagym_checkout", f"{co}/.git/HEAD", r"\A" + ALPAGYM_COMMIT + r"\n\Z")
    lock = _file_rule("alpagym_checkout", f"{co}/uv.lock", None, ALPAGYM_UV_LOCK_SHA256)
    b.add("V1_alpagym_pin", "verify", "The AlpaGym checkout is detached at the pin with the "
          "pinned uv.lock (its frozen environment comes from teacher_rl.ops phase alpagym).",
          f"{_R3} section 2 (AlpaGym 972d160; uv.lock sha256); teacher_rl.ops G2/G4/G7", None,
          expected=[head, lock], timeout="verify")
    sources = [_file_rule("alpagym_checkout", f"{co}/{item['path']}", item["size_bytes"],
                          item["sha256"]) for _, item in sorted(CONVERTERS.items())]
    b.add("V2_converter_sources", "verify", "Both converter scripts are the pinned bytes before "
          "either runs.", f"{_R3} section 3 steps 4 and 7; tests/fixtures/s2_reexport/"
          "provenance.json", None, expected=sources, timeout="verify")
    teacher = b.assets["teacher"]
    rules = [_file_rule("teacher_snapshot", f"{p['teacher_snapshot']}/{f['path']}",
                        f["size_bytes"], f["sha256"]) for f in teacher["files"]]
    b.add("V3_teacher_snapshot", "verify", "The release snapshot holds the pinned teacher files "
          "(the control's reference bytes).", "R2 E1; " + f"{_R3} section 2 (teacher revision "
          "7aba829; hash shards, index, config)", None, expected=rules, timeout="verify")
    reason = b.assets["cosmos_reason2"]
    rules = [_file_rule("hf_home", _hf_cache_path(p["hf_home"], reason["repository"],
                                                  reason["revision"], f["path"]),
                        f["size_bytes"], f["sha256"]) for f in reason["files"]]
    b.add("V4_cosmos_reason2_cache", "verify", "The gated Cosmos-Reason2-8B config/processor "
          "files are cached at the registered revision (C1 builds the processor and copies the "
          "generation config from them offline).", f"{_R3} section 1 finding 8; alpagym "
          f"{CONVERTERS['release_to_gym']['path']}:38,89-100,184", None,
          requires=("host_setup", "terms:cosmos_reason2"), expected=rules, timeout="verify",
          level="L3")


def _phase_convert(b):
    p, co = b.paths, b.paths["alpagym_checkout"]
    offline = {"HF_HOME": p["hf_home"], "HF_HUB_OFFLINE": "1"}
    bundle, reexport = p["gym_bundle"], p["reexport_dir"]
    release_to_gym = CONVERTERS["release_to_gym"]["path"]
    b.add("C1_release_to_gym", "convert", "Convert the release snapshot to the AlpaGym bundle "
          "(config rewrite, symlinked shards and index, expanded processor, generation config; "
          "no --overwrite).",
          f"{_R3} section 3 step 4; alpagym {release_to_gym}:38,69-86,89-100,136-188; "
          "alpamayo-recipes 1d99fc5 src/alpamayo/utils/checkpoint_utils.py:67-120",
          ["uv", "run", "--no-sync", "python", f"{co}/{release_to_gym}", "--input",
           p["teacher_snapshot"], "--output", bundle, "--vlm-name-or-path",
           COSMOS_REASON2_REPOSITORY], cwd=co,
          requires=("host_setup", "terms:teacher", "terms:cosmos_reason2"),
          expected=[*_config_rules("gym_bundle", bundle, GYM_MODEL_TYPE),
                    *[_exists("gym_bundle", f"{bundle}/{name}") for name in TOKENIZER_FILES]],
          timeout="import", env_extra=offline, level="L3")
    gym_to_inference = CONVERTERS["gym_to_inference"]["path"]
    b.add("C2_gym_to_inference", "convert", "Re-export the bundle in inference format (VLM keys "
          "under vlm., shards re-saved, index and config rewritten; no --overwrite): policy B.",
          f"{_R3} section 3 step 7; alpagym {gym_to_inference}:74-101,120-170",
          ["uv", "run", "--no-sync", "python", f"{co}/{gym_to_inference}", "--input", bundle,
           "--output", reexport], cwd=co, requires=("host_setup", "terms:teacher"),
          expected=[*_config_rules("reexport_dir", reexport, INFERENCE_MODEL_TYPE),
                    _exists("reexport_dir", f"{reexport}/{INDEX_NAME}"),
                    *[_exists("reexport_dir", f"{reexport}/{name}") for name in TOKENIZER_FILES]],
          timeout="build", env_extra=offline, level="L3")


def control_dir(evidence_dir):
    """Directory the control stage writes (agent technical default name)."""
    return f"{evidence_dir}/{CONTROL_DIR_NAME}"


def _phase_control(b, scope):
    p = b.paths
    out = control_dir(p["evidence_dir"])
    # The release snapshot is still the inventory's pinned teacher when the verdict is taken
    # (also in a control-only plan, which has no V3): checked after X1 writes control.json.
    rules = [_file_rule("teacher_snapshot", f"{p['teacher_snapshot']}/{f['path']}",
                        f["size_bytes"], f["sha256"]) for f in b.assets["teacher"]["files"]]
    rules.extend(_exists("evidence_dir", f"{out}/{name}") for name in CONTROL_FILES)
    rules.append(_regex("evidence_dir", f"{out}/control.json", CONTROL_PASSED_PATTERN))
    b.add("X1_reexport_control", "control", "Tensor manifests of the release and the re-export "
          "and the bit-identical control verdict (policy B admission precondition).",
          "plan 02 section 3 S2.4; build-spec section 2 rule 11; gate.tensor_manifest",
          [*GUEST_PYTHON, "scripts/s2_reexport.py", "compare", "--release-root",
           p["teacher_snapshot"], "--reexport-root", p["reexport_dir"], "--index", INDEX_NAME,
           "--scope", scope, "--output-dir", out], requires=("host_setup", "terms:teacher"),
          expected=rules, idempotency=rules, timeout="sync", level="L3")


def _empty_plan(scope=None):
    return {
        "artifact_kind": PLAN_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
        "compiled": False, "producer": _producer(),
        "inputs": {"inventory_sha256": None, "action_record_sha256": None},
        "phases": [], "roots": {}, "authorizations": {}, "secrets_required": [], "stages": [],
        "issues": [], "execution_status": "qualify_on_host", "level": "L2",
        "units": dict(PLAN_UNITS), "compiler": dict(COMPILER),
    }


def build_reexport_plan(inventory, action_record, *, phases):
    """Compile the re-export stages, or refuse with issues and no stages (never raises).

    ``phases`` is a non-empty list of distinct names from ``PHASES`` (normalized to that
    order). Deterministic: the same inputs give the same plan and ``plan_sha256``.
    """
    try:
        return _build(inventory, action_record, phases)
    except (TypeError, AttributeError, KeyError, ValueError) as exc:
        plan = _empty_plan()
        plan["issues"].append(_issue("invalid_input", "inputs", "Unexpected value type in the "
                                     f"inventory or action record ({type(exc).__name__})."))
        return _seal(plan)


def _build(inventory, action_record, phases):
    plan = _empty_plan()
    for name, value in (("inventory_sha256", inventory),
                        ("action_record_sha256", action_record)):
        try:
            plan["inputs"][name] = identity_hash(value)
        except (TypeError, ValueError):
            plan["inputs"][name] = None
    report = validate_reexport_inventory(inventory)
    plan["scope"] = report["scope"]
    issues = plan["issues"]
    if not report["valid"]:
        bad = sorted({i["code"] for i in report["issues"] if i["severity"] == "error"})
        issues.append(_issue("inventory_invalid", "inventory",
                             f"Inventory has structural errors: {bad}."))
    issues.extend(_validate_record(action_record))
    normalized = _normalize_phases(phases)
    if normalized is None:
        issues.append(_issue("invalid_phases", "phases",
                             f"phases is a non-empty list of distinct names from {list(PHASES)}."))
    if issues:
        return _seal(plan)
    plan["phases"] = normalized
    unauthorized = [p for p in normalized if p not in action_record["authorized_phases"]]
    if unauthorized:
        issues.append(_issue("phase_not_authorized", "action_record.authorized_phases",
                             f"Phases {unauthorized} lack recorded authority."))
        return _seal(plan)
    builder = _Builder(inventory, action_record, [])
    _needed_inputs(builder, normalized)
    if builder.issues:
        issues.extend(builder.issues)
        return _seal(plan)
    compilers = {"verify": lambda: _phase_verify(builder),
                 "convert": lambda: _phase_convert(builder),
                 "control": lambda: _phase_control(builder, report["scope"])}
    for phase in normalized:
        compilers[phase]()
    if builder.issues:
        issues.extend(builder.issues)
        return _seal(plan)
    candidate = dict(plan, roots=dict(builder.paths), stages=builder.stages,
                     authorizations=dict(sorted(builder.used_auth.items())))
    candidate["secrets_required"] = sorted(
        {s["stdin_secret"] for s in builder.stages if s["stdin_secret"]}
        | {name for s in builder.stages for name in s["secret_env"].values()})
    problems = [i for i in _plan_content_issues(candidate) if i["severity"] == "error"]
    if problems:
        issues.extend(problems)
        return _seal(plan)
    notes = [_issue("prerequisites_produced_elsewhere", "stages.verify",
                    "This plan verifies, never builds, the AlpaGym environment (teacher_rl.ops "
                    "alpagym), the teacher snapshot (s2 E1 / ops M1) and the Cosmos-Reason2-8B "
                    "cache (s2 E2a / ops M2).", "info")]
    if "convert" in normalized:
        notes.append(_issue(
            "hf_offline_cache_resolution_unverified", "stages.C1_release_to_gym",
            "The Cosmos-Reason2-8B cache may be fetched by commit without refs/main; the "
            "offline converter load by repo name must be tested on the host (INFERRED).",
            "info"))
    candidate["issues"] = notes
    candidate["compiled"] = True
    return _seal(candidate)


# --- control verdict ----------------------------------------------------------------------


def reexport_control(release_manifest, reexport_manifest):
    """Bit-identical re-export verdict over two tensor manifests (L1; actual bytes L3).

    ``passed`` only when ``compare_manifests`` passes and the tensor sets are identical in
    name, dtype, shape and per-tensor SHA-256; every other outcome is ``blocked`` with the
    differing tensors listed. File identity is reported beside the verdict, never in it.
    """
    comparison = compare_manifests(release_manifest, reexport_manifest)
    issues = []
    compared = comparison["status"] == "passed"
    if not compared:
        for item in comparison["issues"]:
            path = {"a": "release", "b": "reexport"}.get(item["path"], item["path"])
            issues.append(_issue(item["code"], f"comparison/{path}", item["message"]))
        issues.append(_issue("comparison_blocked", "comparison", "The manifests could not be "
                             "compared; the control has no verdict and B is not admitted."))
    differences = {key: comparison[source] for key, source in (
        ("only_in_release", "only_in_a"), ("only_in_reexport", "only_in_b"),
        ("dtype_changed", "dtype_changed"), ("shape_changed", "shape_changed"),
        ("bytes_changed", "bytes_changed"))}
    differences["changed"] = [{"name": row["name"], "fields": row["fields"],
                               "release": row["a"], "reexport": row["b"]}
                              for row in comparison["changed"]]
    if compared and not comparison["tensors_identical"]:
        counts = ", ".join(f"{key}={len(value)}" for key, value in differences.items()
                           if key != "changed" and value)
        issues.append(_issue("tensors_differ", "differences", "The re-export is not "
                             f"tensor-identical to the release ({counts})."))
    empty = compared and not release_manifest["tensors"]
    if empty:
        issues.append(_issue("empty_tensor_set", "tensors", "The release manifest holds no "
                             "tensors: absence is never a bit-identical pass."))
    if compared and comparison["files_identical"]:
        issues.append(_issue("reexport_files_identical_to_release", "files", "Every file is "
                             "byte-identical to the release: B is not its own export "
                             "(identical_policy_snapshots).", "warning"))
    passed = compared and not empty and comparison["tensors_identical"] is True
    status = "passed" if passed else "blocked"

    def tensor_count(manifest):
        return len(manifest["tensors"]) if compared else None

    def set_hash(manifest):
        return manifest["tensor_set_sha256"] if compared else None

    issues.sort(key=lambda i: (i["path"], i["code"], i["message"]))
    return {
        "artifact_kind": CONTROL_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": comparison["scope"],
        "producer": _producer(),
        "input_hashes": {"release_manifest": comparison["input_hashes"]["a"],
                         "reexport_manifest": comparison["input_hashes"]["b"]},
        "units": {"tensor_counts": "tensors", "file_sha256": "sha256 of the stored file bytes"},
        "status": status,
        "control": status,
        "criterion": CRITERION,
        "role": "policy B admission precondition (S2.4); not a scientific result",
        "tensors": {
            "release_count": tensor_count(release_manifest),
            "reexport_count": tensor_count(reexport_manifest),
            "identical_count": len(comparison["identical"]) if compared else None,
            "release_tensor_set_sha256": set_hash(release_manifest),
            "reexport_tensor_set_sha256": set_hash(reexport_manifest),
        },
        "differences": differences,
        "files": {"files_identical": comparison["files_identical"],
                  "metadata_identical": comparison["metadata_identical"],
                  "relation": comparison["relation"],
                  "release": comparison["files"]["a"], "reexport": comparison["files"]["b"]},
        "host_qualification": "qualify_on_host",
        "issues": issues,
    }
