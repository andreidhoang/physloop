"""S4 two-GPU runtime inventory and bootstrap plan, compiled with the S2 bootstrap machinery.

Tickets: lane-r3 section 2 (pin/environment manifest to freeze) and section 3 steps 1-5 (the
documented two-GPU local launch sequence up to the launch itself), E15 (lock/config/weight
identity), build-spec B3 item (1). Serves S4.1-S4.4 [C] (G9, G7) under D-024. Agent-authored
evaluator engineering (D-018); an engineering plan, never an admission, rental, licence action
or download.

Interface reconciliation (recorded, not hidden). Build-spec B3 asks to compile the S4 plan
"through ``gate/s2_bootstrap.py``" and never to write a second plan compiler. Its public entry
``build_execution_plan`` cannot take an S4 inventory: ``gate.s2_runtime`` pins the inventory
sources to the S2 gate AlpaSim ``e121e59`` and phases A-E emit S2 AlpaSim stages. This module
therefore reuses the S2 compiler's own parts - the stage builder ``_Builder`` (stage record
shape, ``requires`` resolution: ``terms:<role>`` from the inventory, everything else from the
action record), the rule constructors, ``TIMEOUTS``, ``ENV_ALLOWLIST``, the content validator
``_plan_content_issues`` (argv/env credential hygiene, rule confinement, every ``requires``
key present) and the hash seal ``_seal`` - and supplies only the S4 stage list. The plan's
``artifact_kind`` is ``s4_bootstrap_plan`` and it records ``compiler`` =
``gate.s2_bootstrap.compiler_identity("s4_runtime")``: ``s4_runtime`` is a registered kind in
``gate.s2_bootstrap.PLAN_KINDS`` whose compiler is this module's ``build_s4_bootstrap_plan``,
so ``gate.s2_bootstrap.validate_plan`` accepts the plan and the single guest runner
``scripts/s2_guest_bootstrap.py`` executes it after recompiling it here (with ``--inventory``,
``--action-record`` and ``--rental-record``). There is no second compiler or runner (A2X).
``tests/test_s4_ops.py`` pins the imported private names and signatures so drift fails loudly.

Inventory (``s4_runtime_inventory`` v1): AlpaGym ``972d160`` with its ``uv.lock`` sha256, the
training AlpaSim ``46b401b`` (tree ``9b83854``, tree-identical to ``10cae1c``), the NRE renderer
by digest, the teacher at revision ``7aba829`` and the gated Cosmos-Reason2-8B snapshot at an
admission-time revision (no default; the ``a9fae2c`` observation is never used). Unknown sizes
and hashes are the string ``"unverified"``, never zero. The host data paths (AlpaGym
checkout, AlpaSim cache, teacher snapshot, converted bundle, HF home, download scratch,
evidence and Docker data root) must not overlap pairwise (review B3-10: the converter writes
``--output`` with ``--overwrite`` and would rewrite a gated snapshot in place).

AlpaSim checkout (review B3-04; agent technical default (D-018)). The run never uses a
separately cloned AlpaSim: pinned AlpaGym builds its own content-addressed checkout at
``<alpasim.checkout_cache_dir>/sha256("<repo_url>@<commit>")[:12]`` (fetch by SHA,
``compile-protos``, relocatable venv; ``alpasim_dependency.py:37-76``) and reuses an existing
entry without an integrity check (``:42-45``). ``repo_url`` is the AlpaGym ``pyproject.toml``
``alpasim-grpc`` git source ``https://github.com/NVlabs/alpasim.git`` (with ``.git``, unlike
``teacher_rl.pins``), so the key for ``46b401b`` is ``62400020a95f``. The inventory therefore
declares ``alpasim_cache_dir``, ``teacher_rl.launch`` passes it as
``alpasim.checkout_cache_dir``, the ``alpasim`` phase creates it and requires it empty before
the first launch, and the ``record`` phase verifies HEAD, tree and venv of exactly that cached
checkout. The rejected alternative, ``alpasim.repo_path``, syncs an editable venv in place
and skips ``compile-protos`` (``:25-32``), which the plan would have to replicate.

Gates: a plan compiles all-or-nothing. It refuses on an invalid inventory, an invalid bootstrap
action record, a requested phase the record does not authorize, any stage whose authorization
is absent (``host_setup`` for every stage; ``terms:<role>`` from the inventory and
``acquire:<role>`` from the record for the renderer, teacher and Cosmos-Reason2-8B), a rental
action record that ``gate.s2_execution.validate_action_record`` does not admit or whose
``resources.gpu_count`` is not 2 (the ``local_disaggregated_2gpu`` topology), a scope mismatch,
an unresolved Cosmos-Reason2-8B revision, a ``terms:<role>`` reference the rental record's
``license_attestation_refs`` does not attest (review B3-19), or anything credential-like.
Secrets never enter the plan: the HF token reaches the gated download only as ``HF_TOKEN``
from a secret file named at run time; the NGC key only through ``docker login
--password-stdin``.

Verification level per function:

- ``validate_s4_inventory``, ``build_s4_bootstrap_plan``: **L1** compile/refusal logic with
  literal known answers; every emitted command is **L2** (transcribed from lane-r3 sections
  2-3 and the pinned AlpaGym/AlpaSim sources cited in each ``source_anchor``). Stages whose
  outcome depends on the delivered host (GPU inventory, NCCL P2P, AlpaGym preflight, the
  AlpaSim cache after AlpaGym's first build, the offline converter run) carry ``level:
  "L3"``. C1 converts with ``HF_HUB_OFFLINE=1`` by repo name after M2 downloaded by commit,
  which may write no ``refs/main`` (review B3-18): the compiled plan carries the info issue
  ``hf_offline_cache_resolution_unverified`` (as ``gate.s2_bootstrap`` does for its E2).
  Executing any stage is **L3**: the plan reports ``execution_status: "qualify_on_host"``,
  never ``passed``.

Agent technical defaults (D-018): stage timeouts reuse ``gate.s2_bootstrap.TIMEOUTS`` (the
AlpaGym frozen sync uses the 7,200 s ``build`` budget because grouped-gemm may build from
source, lane-r3 section 2); the ``record`` phase is compiled separately, after AlpaGym's first
launch has built the training AlpaSim venv (lane-r3 finding 12); H6 applies the S2
``HOST_FILESYSTEM_FLOOR_BYTES`` (285 GB) to the teacher-snapshot and Docker filesystems and
lists the S4 consumers whose size is unknown (``S4_UNKNOWN_CONSUMERS``: AlpaGym venv and uv
cache, the cached AlpaSim checkout and venv, locally built AlpaSim images, checkpoint exports
and resume bundles, run logs and capture streams, converted-bundle files) - never as zero
(review B3-15).

Never: runs a process, opens a network connection, reads a credential, provisions, funds,
accepts terms, downloads or chooses an owner value.
"""

import hashlib
import re
from pathlib import Path

from gate.s2_bootstrap import (
    HOST_FILESYSTEM_FLOOR_BYTES,
    _Builder,
    _file_rule,
    _hf_cache_path,
    _inspect_probe,
    _issue,
    _min_version,
    _plan_content_issues,
    _probe,
    _seal,
    _stdout,
    compiler_identity,
)
from gate.s2_bootstrap import UNITS as PLAN_UNITS
from gate.s2_execution import validate_action_record
from gate.s2_identity import identity_hash
from gate.s2_runtime import (
    is_placeholder,
    is_portable_absolute,
    is_safe_relative,
    paths_overlap,
    secret_like,
)
from teacher_rl import pins

INVENTORY_KIND = "s4_runtime_inventory"
PLAN_KIND = "s4_bootstrap_plan"
RECORD_KIND = "s4_bootstrap_action_record"
REPORT_KIND = "s4_runtime_inventory_report"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
PLATFORM = "linux/amd64"
REQUIRED_GPU_COUNT = 2  # lane-r3 finding 13: topology local_disaggregated_2gpu
PHASES = ("host", "alpagym", "alpasim", "images", "assets", "convert", "nccl", "record")
AUTHORIZATION_KEYS = ("host_setup", "acquire:renderer", "acquire:teacher",
                      "acquire:cosmos_reason2")
HOST_PATHS = (
    "physloop_root", "alpagym_checkout", "alpasim_cache_dir", "teacher_snapshot",
    "converted_bundle", "hf_home", "download_scratch", "evidence_dir", "docker_data_root",
    "home",
)
CHECKOUTS = ("alpagym_checkout",)
# Pairwise disjoint (review B3-10); home and physloop_root may contain them.
DISJOINT_PATHS = ("alpagym_checkout", "alpasim_cache_dir", "teacher_snapshot",
                  "converted_bundle", "hf_home", "download_scratch", "evidence_dir",
                  "docker_data_root")
# alpagym pyproject.toml [tool.uv.sources] alpasim-grpc git (pinned 972d160).
ALPAGYM_ALPASIM_REPO_URL = "https://github.com/NVlabs/alpasim.git"
S4_UNKNOWN_CONSUMERS = (  # agent technical default (D-018): sizes unknown until measured
    "alpagym_venv_and_uv_cache", "alpasim_cached_checkout_and_venv",
    "checkpoint_exports_and_resume_bundles", "converted_bundle_files",
    "locally_built_alpasim_images", "run_directory_logs_and_capture_streams",
)
ALPAGYM = pins.UPSTREAM_PINS["alpagym"]
TRAINING_ALPASIM = pins.UPSTREAM_PINS["training_alpasim"]
ASSET_ROLES = {
    "teacher": (pins.TEACHER["repository"], pins.TEACHER["repo_type"]),
    "cosmos_reason2": (pins.COSMOS_REASON2["repository"], pins.COSMOS_REASON2["repo_type"]),
}
CONVERTER = "packages/policies/alpamayo_r1/scripts/convert_release_to_alpagym_checkpoint.py"
NCCL_CHECK = "install/check_nccl_p2p.py"
CHECK_ENV = "install/check_env.sh"
UV_MINIMUM = "0.10.0"  # lane-r3 section 3 step 1 (uv >= 0.10)
UNVERIFIED = "unverified"
_TOP = {"artifact_kind", "schema_version", "scope", "sources", "alpagym_uv_lock_sha256",
        "platform", "host_paths", "images", "assets"}
_IMAGE_FIELDS = {"role", "kind", "reference", "platform", "size_bytes", "authorization_ref"}
_ASSET_FIELDS = {"role", "repository", "repo_type", "revision", "authorization_ref", "files"}
_FILE_FIELDS = {"path", "size_bytes", "sha256"}
_RECORD_FIELDS = {"artifact_kind", "schema_version", "action_record_ref", "authorized_phases",
                  "authorizations"}
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIGEST_REFERENCE = re.compile(r"[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}")
_R3 = "docs/research/2026-11-08-codebase-build/lane-r3-s4-upstream.md"


def alpasim_cached_checkout(cache_dir):
    """AlpaGym's content-addressed checkout path for the training AlpaSim pin (L2)."""
    key = hashlib.sha256(f"{ALPAGYM_ALPASIM_REPO_URL}@{TRAINING_ALPASIM['commit']}".encode())
    return f"{cache_dir}/{key.hexdigest()[:12]}"


def _producer():
    return {"module": "teacher_rl.ops",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _strings(value, path):
    if type(value) is str:
        yield path, value
    elif type(value) is dict:
        for key in sorted(value, key=str):
            yield from _strings(value[key], f"{path}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def _bad_ref(value):
    return (type(value) is not str or not value.strip() or is_placeholder(value)
            or secret_like(value))


# --- inventory ----------------------------------------------------------------------


def _check_sources(inventory, issues):
    sources = inventory["sources"]
    expected = {"alpagym": {"repository": ALPAGYM["repository"], "commit": ALPAGYM["commit"]},
                "training_alpasim": {"repository": TRAINING_ALPASIM["repository"],
                                     "commit": TRAINING_ALPASIM["commit"],
                                     "tree": TRAINING_ALPASIM["tree"]}}
    if type(sources) is not dict or set(sources) != set(expected):
        issues.append(_issue("invalid_inventory", "inventory.sources",
                             "sources are exactly alpagym and training_alpasim."))
        return
    for name, pin in expected.items():
        if sources[name] != pin:
            issues.append(_issue("pin_mismatch", f"inventory.sources.{name}",
                                 f"{name} is pinned to {pin} (teacher_rl.pins)."))
    if inventory["alpagym_uv_lock_sha256"] != pins.ALPAGYM_LOCK["uv_lock_sha256"]:
        issues.append(_issue("pin_mismatch", "inventory.alpagym_uv_lock_sha256",
                             "uv.lock sha256 must equal the AlpaGym lock pin."))


def _check_paths(paths, issues):
    if type(paths) is not dict or set(paths) != set(HOST_PATHS):
        issues.append(_issue("invalid_inventory", "inventory.host_paths",
                             f"host_paths are exactly {list(HOST_PATHS)}."))
        return
    for name in HOST_PATHS:
        if paths[name] is None:
            issues.append(_issue("host_path_unset", f"inventory.host_paths.{name}",
                                 "Host path is chosen with the host; not defaulted.", "blocking"))
        elif not is_portable_absolute(paths[name]):
            issues.append(_issue("invalid_host_path", f"inventory.host_paths.{name}",
                                 "Absolute portable ASCII path without dot components."))
    pairs = set()
    for checkout in CHECKOUTS:
        pairs.update((checkout, name) for name in HOST_PATHS if name not in (checkout, "home"))
    pairs.update((a, b) for i, a in enumerate(DISJOINT_PATHS) for b in DISJOINT_PATHS[i + 1:])
    for first, second in sorted(pairs, key=lambda pair: (HOST_PATHS.index(pair[0]),
                                                         HOST_PATHS.index(pair[1]))):
        if not (is_portable_absolute(paths[first]) and is_portable_absolute(paths[second])):
            continue
        if paths_overlap(paths[first], paths[second]):
            issues.append(_issue("host_path_overlap", f"inventory.host_paths.{second}",
                                 f"Keep {second} and {first} disjoint (clean worktree; gated "
                                 "snapshots, outputs and evidence never share a directory)."))


def _check_files(files, path, issues, demand, role):
    if type(files) is not list:
        issues.append(_issue("invalid_asset", f"{path}.files", "files is a list."))
        return
    if not files:
        issues.append(_issue("asset_files_unlisted", f"{path}.files",
                             "List every file to acquire (pinned snapshot).", "blocking"))
        return
    seen = set()
    for index, row in enumerate(files):
        rpath = f"{path}.files[{index}]"
        if type(row) is not dict or set(row) != _FILE_FIELDS:
            issues.append(_issue("invalid_file", rpath, f"File rows are {sorted(_FILE_FIELDS)}."))
            continue
        if not is_safe_relative(row["path"]) or row["path"] in seen:
            issues.append(_issue("unsafe_relative_path", f"{rpath}.path",
                                 "Unique safe relative path (no traversal, no -leading part)."))
        seen.add(row["path"])
        size = row["size_bytes"]
        if size != UNVERIFIED and not (type(size) is int and size > 0):
            issues.append(_issue("invalid_size", f"{rpath}.size_bytes",
                                 "Size is a positive integer or 'unverified', never 0/null."))
        else:
            demand.append((role, f"asset:{role}:{row['path']}", size))
        sha = row["sha256"]
        if sha != UNVERIFIED and (type(sha) is not str or _SHA256.fullmatch(sha) is None):
            issues.append(_issue("invalid_sha256", f"{rpath}.sha256",
                                 "sha256 is 64 lowercase hex or 'unverified'."))


def _check_assets(assets, issues, demand):
    if type(assets) is not list or sorted(
            a.get("role") for a in assets if type(a) is dict) != sorted(ASSET_ROLES) or (
            len(assets) != len(ASSET_ROLES)):
        issues.append(_issue("invalid_inventory", "inventory.assets",
                             f"assets are exactly the roles {sorted(ASSET_ROLES)}."))
        return
    for index, asset in enumerate(assets):
        path = f"inventory.assets[{index}]"
        if set(asset) != _ASSET_FIELDS:
            issues.append(_issue("invalid_asset", path, f"Fields are {sorted(_ASSET_FIELDS)}."))
            continue
        role = asset["role"]
        if (asset["repository"], asset["repo_type"]) != ASSET_ROLES[role]:
            issues.append(_issue("pin_mismatch", f"{path}.repository",
                                 f"{role} is {ASSET_ROLES[role]}."))
        revision = asset["revision"]
        if role == "teacher" and revision != pins.TEACHER["revision"]:
            issues.append(_issue("pin_mismatch", f"{path}.revision",
                                 f"teacher revision is pinned to {pins.TEACHER['revision']}."))
        elif role == "cosmos_reason2" and revision is None:
            issues.append(_issue("revision_unresolved", f"{path}.revision",
                                 "Cosmos-Reason2-8B revision is an admission-time input "
                                 "(no default).", "blocking"))
        elif type(revision) is not str or _COMMIT.fullmatch(revision) is None:
            issues.append(_issue("invalid_revision", f"{path}.revision", "40-hex revision."))
        ref = asset["authorization_ref"]
        if ref is not None and _bad_ref(ref):
            issues.append(_issue("invalid_authorization_ref", f"{path}.authorization_ref",
                                 "A recorded, non-placeholder, non-secret reference."))
        _check_files(asset["files"], path, issues, demand, role)


def _check_images(images, issues, demand):
    if type(images) is not list or len(images) != 1 or type(images[0]) is not dict or (
            set(images[0]) != _IMAGE_FIELDS) or images[0]["role"] != "renderer":
        issues.append(_issue("invalid_inventory", "inventory.images",
                             f"images is exactly one renderer row {sorted(_IMAGE_FIELDS)}."))
        return
    image, path = images[0], "inventory.images[0]"
    if image["kind"] != "registry_digest" or image["platform"] != PLATFORM:
        issues.append(_issue("invalid_image", path, f"registry_digest image for {PLATFORM}."))
    reference = image["reference"]
    if reference is None:
        issues.append(_issue("image_reference_unset", f"{path}.reference",
                             "The renderer is pulled only as repository@sha256.", "blocking"))
    elif type(reference) is not str or _DIGEST_REFERENCE.fullmatch(reference) is None:
        issues.append(_issue("mutable_reference", f"{path}.reference",
                             "A tag is not an identity; use repository@sha256:<64 hex>."))
    size = image["size_bytes"]
    if size != UNVERIFIED and not (type(size) is int and size > 0):
        issues.append(_issue("invalid_size", f"{path}.size_bytes", "Positive int or "
                                                                   "'unverified'."))
    else:
        demand.append(("renderer", "image:renderer", size))
    ref = image["authorization_ref"]
    if ref is not None and _bad_ref(ref):
        issues.append(_issue("invalid_authorization_ref", f"{path}.authorization_ref",
                             "A recorded, non-placeholder, non-secret reference."))


def validate_s4_inventory(inventory):
    """Validate an S4 runtime inventory; deterministic report (L1; pinned facts L2)."""
    issues, demand = [], []
    try:
        input_hash = identity_hash(inventory)
    except (TypeError, ValueError):
        input_hash = None
    report = {"artifact_kind": REPORT_KIND, "schema_version": SCHEMA_VERSION, "scope": None,
              "producer": _producer(), "input_sha256": input_hash,
              "units": {"bytes": "bytes"}, "level": "L1",
              "execution_status": "qualify_on_host", "valid": False, "status": "blocked",
              "issues": issues, "disk_demand": None}
    if input_hash is None or type(inventory) is not dict or set(inventory) != _TOP:
        issues.append(_issue("invalid_inventory", "inventory",
                             f"Inventory fields are exactly {sorted(_TOP)} (strict JSON)."))
        return report
    try:
        if inventory["artifact_kind"] != INVENTORY_KIND or inventory["schema_version"] != 1 or (
                type(inventory["schema_version"]) is not int):
            issues.append(_issue("invalid_inventory", "inventory.schema_version",
                                 f"Only {INVENTORY_KIND} schema_version 1 is known."))
        if inventory["scope"] in SCOPES:
            report["scope"] = inventory["scope"]
        else:
            issues.append(_issue("invalid_scope", "inventory.scope", f"one of {SCOPES}."))
        if inventory["platform"] != PLATFORM:
            issues.append(_issue("invalid_platform", "inventory.platform", PLATFORM))
        for path, value in _strings(inventory, "inventory"):
            if secret_like(value):
                issues.append(_issue("secret_like_value", path, "Never embed credentials."))
        _check_sources(inventory, issues)
        _check_paths(inventory["host_paths"], issues)
        _check_images(inventory["images"], issues, demand)
        _check_assets(inventory["assets"], issues, demand)
    except (TypeError, AttributeError, KeyError, ValueError) as exc:
        issues.append(_issue("invalid_inventory", "inventory",
                             f"Unexpected value type ({type(exc).__name__})."))
    known = {}
    for root, _, size in demand:
        if type(size) is int:
            known[root] = known.get(root, 0) + size
    report["disk_demand"] = {
        "known_bytes_by_role": dict(sorted(known.items())),
        "known_bytes": sum(known.values()),
        "unknown_components": sorted(name for _, name, size in demand if size == UNVERIFIED),
    }
    issues.sort(key=lambda i: (i["path"], i["code"]))
    severities = {i["severity"] for i in issues}
    report["valid"] = "error" not in severities
    report["status"] = "blocked" if severities & {"error", "blocking"} else "passed"
    return report


# --- plan ---------------------------------------------------------------------------


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
    if type(phases) is not list or not all(p in PHASES for p in phases):
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


def _rental_gpu_count(rental_record, scope, issues):
    admission = validate_action_record(rental_record)
    if not admission["admitted"]:
        issues.append(_issue("rental_record_not_admitted", "rental_record",
                             f"{sorted({i['code'] for i in admission['issues']})}"))
        return None, None
    if rental_record["scope"] != scope:
        issues.append(_issue("scope_mismatch", "rental_record.scope",
                             "inventory and rental record come from different scopes"))
    gpus = rental_record["resources"]["gpu_count"]
    if gpus != REQUIRED_GPU_COUNT:
        issues.append(_issue("gpu_count_mismatch", "rental_record.resources.gpu_count",
                             f"the two-GPU AlpaGym topology needs exactly {REQUIRED_GPU_COUNT} "
                             f"GPUs; the record rents {gpus}"))
    return gpus, admission["input_hashes"]["record"]


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
    if "images" in phases and b.images["renderer"]["reference"] is None:
        b.refuse("image_reference_unset", "inventory.images.renderer",
                 "The images phase pulls only repository@sha256.")
    if ("assets" in phases or "convert" in phases) and (
            b.assets["cosmos_reason2"]["revision"] is None):
        b.refuse("revision_unresolved", "inventory.assets.cosmos_reason2.revision",
                 "Cosmos-Reason2-8B revision is an admission-time input (no default).")
    if "assets" in phases:
        for role in ASSET_ROLES:
            if not b.assets[role]["files"]:
                b.refuse("asset_files_unlisted", f"inventory.assets.{role}.files",
                         "List every file to acquire.")


def _known(rows):
    return sum(r["size_bytes"] for r in rows if type(r["size_bytes"]) is int)


def _git_rules(root, checkout, commit):
    head_any = {"kind": "file_regex", "root": root, "path": f"{checkout}/.git/HEAD",
                "pattern": r"\A(ref: refs/heads/[A-Za-z0-9._/-]+|[0-9a-f]{40})\n\Z"}
    head_pin = {"kind": "file_regex", "root": root, "path": f"{checkout}/.git/HEAD",
                "pattern": r"\A" + commit + r"\n\Z"}
    return head_any, head_pin


def _phase_host(b, gpus, unknown):
    p = b.paths
    b.add("H0_directories", "host", "Create the evidence and download directories.",
          f"{_R3} section 3 (run outputs stay outside checkouts)",
          ["install", "-d", p["evidence_dir"], p["download_scratch"]])
    b.add("H1_uv_version", "host", "uv >= 0.10 on the host.",
          f"{_R3} section 3 step 1; alpagym docs/ONBOARDING.md:41-89", ["uv", "--version"],
          expected=[_min_version(r"(?m)^uv (\d+\.\d+\.\d+)", UV_MINIMUM)])
    b.add("H2_compose_version", "host", "Docker with Compose v2.",
          f"{_R3} section 3 step 1", ["docker", "compose", "version"],
          expected=[_stdout(r"Docker Compose version v?2\.\d+\.\d+")])
    b.add("H3_gpu_inventory", "host", f"Exactly {gpus} GPUs with edition, driver and compute "
          "capability recorded (Blackwell sm_120 is outside the AlpaGym Dockerfile arch list).",
          f"{_R3} sections 2 and 3 (record GPU edition, driver, CUDA, compute capability)",
          ["nvidia-smi", "--query-gpu=index,name,driver_version,compute_cap",
           "--format=csv,noheader"],
          expected=[_stdout(r"\A(?:\d+, [^\n]+\n){" + str(gpus) + r"}\Z")], level="L3")
    b.add("H4_redis_version", "host", "redis-server present (Cosmos controller).",
          f"{_R3} section 3 step 1", ["redis-server", "--version"],
          expected=[_stdout(r"Redis server v=\d+\.\d+")])
    b.add("H5_git_lfs_version", "host", "git-lfs present.", f"{_R3} section 3 step 1",
          ["git", "lfs", "version"], expected=[_stdout(r"git-lfs/\d+\.\d+")])
    teacher, reason = b.assets["teacher"]["files"], b.assets["cosmos_reason2"]["files"]
    renderer = b.images["renderer"]["size_bytes"]
    floor = HOST_FILESYSTEM_FLOOR_BYTES
    wanted = [("teacher_snapshot", _known(teacher), floor), ("hf_home", _known(reason), 0),
              ("docker_data_root", renderer if type(renderer) is int else 0, floor)]
    b.add("H6_disk_free", "host", "Known byte demand fits the free space per filesystem; the "
          "model and Docker filesystems meet the 285 GB floor; unknown sizes (inventory and "
          "S4 consumers) are recorded as unverified, never as zero.",
          f"{_R3} section 2 (sizes); gate.s2_bootstrap free_space rule and floor", None,
          expected=[{"kind": "free_space", "requirements": [
              {"root": name, "path": b.paths[name], "bytes": size,
               "filesystem_floor_bytes": minimum}
              for name, size, minimum in wanted if size > 0 or minimum > 0],
              "unknown_components": sorted({*unknown, *S4_UNKNOWN_CONSUMERS})}],
          timeout="verify")


def _phase_alpagym(b):
    co = b.paths["alpagym_checkout"]
    head_any, head_pin = _git_rules("alpagym_checkout", co, ALPAGYM["commit"])
    lock = {"kind": "file", "root": "alpagym_checkout", "path": f"{co}/uv.lock",
            "size_bytes": None, "sha256": pins.ALPAGYM_LOCK["uv_lock_sha256"]}
    b.add("G1_clone", "alpagym", "Clone AlpaGym.", f"{_R3} section 2 (AlpaGym commit 972d160)",
          ["git", "clone", ALPAGYM["repository"], co], idempotency=[head_any], timeout="clone")
    b.add("G2_checkout", "alpagym", "Detach at the AlpaGym pin.", f"{_R3} section 2",
          ["git", "-C", co, "checkout", "--detach", ALPAGYM["commit"]], expected=[head_pin],
          idempotency=[head_pin])
    b.add("G3_status_before", "alpagym", "Clean worktree before sync.",
          f"{_R3} section 2 (working-tree diff must be empty)",
          ["git", "-C", co, "status", "--porcelain"], expected=[_stdout(r"\A\Z")])
    b.add("G4_lock_before", "alpagym", "uv.lock bytes equal the pinned sha256 before sync.",
          f"{_R3} section 2 (record uv.lock sha256 before/after; fail on change)", None,
          expected=[lock], timeout="verify")
    b.add("G5_uv_version", "alpagym", "Record uv --version beside the lock.",
          f"{_R3} section 2", ["uv", "--version"], expected=[_stdout(r"(?m)^uv \d+\.\d+\.\d+")])
    b.add("G6_uv_sync_frozen", "alpagym", "Frozen environment from the pinned lock "
          "(never re-resolved; Linux x86_64 only).",
          f"{_R3} section 3 step 2 (uv sync --all-packages; harness adds --frozen); "
          "alpagym pyproject.toml:30", ["uv", "sync", "--frozen", "--all-packages"], cwd=co,
          timeout="build")
    b.add("G7_lock_after", "alpagym", "uv.lock unchanged after sync.", f"{_R3} section 2",
          None, expected=[lock], timeout="verify")
    b.add("G8_status_after", "alpagym", "Working tree still clean after sync.",
          f"{_R3} section 2", ["git", "-C", co, "status", "--porcelain"],
          expected=[_stdout(r"\A\Z")])
    freeze = [_stdout(r"(?m)^" + re.escape(name) + "==" + re.escape(version) + "$")
              for name, version in sorted(pins.ALPAGYM_LOCK["packages"].items())]
    b.add("G9_pip_freeze", "alpagym", "Installed set carries the locked versions.",
          f"{_R3} section 2 (uv.lock 217 packages)", ["uv", "pip", "freeze"], cwd=co,
          expected=freeze)
    b.add("G10_check_env", "alpagym", "AlpaGym preflight passes on the delivered host.",
          f"{_R3} section 3 step 1; alpagym {CHECK_ENV}:36-105",
          ["bash", f"{co}/{CHECK_ENV}"], cwd=co, level="L3")


def _phase_alpasim(b):
    cache = b.paths["alpasim_cache_dir"]
    b.add("S1_cache_dir", "alpasim", "Create the AlpaSim checkout cache that "
          "teacher_rl.launch passes as alpasim.checkout_cache_dir.",
          "alpagym alpasim_dependency.py:119-144,248-253 (checkout_cache_dir)",
          ["install", "-d", cache])
    b.add("S2_cache_fresh", "alpasim", "The cache holds no entry before the first launch: "
          "AlpaGym reuses a cached checkout without an integrity check, so the pin must be "
          "built by AlpaGym itself (fetch by SHA, compile-protos, relocatable venv).",
          f"alpagym alpasim_dependency.py:37-76; {_R3} section 2 (cache reused without "
          "integrity check)", ["find", cache, "-mindepth", "1", "-maxdepth", "1"],
          expected=[_stdout(r"\A\Z")])


def _phase_images(b):
    reference = b.images["renderer"]["reference"]
    gated = ("host_setup", "terms:renderer", "acquire:renderer")
    pattern, inspect = _inspect_probe(reference)
    present = [_probe(inspect, pattern)]
    b.add("I1_ngc_login", "images", "GATE: log in to nvcr.io with Huy's NGC key from stdin.",
          f"{_R3} section 2 (renderer nre-ga:26.04; record RepoDigests)",
          ["docker", "login", "nvcr.io", "-u", "$oauthtoken", "--password-stdin"],
          requires=gated, stdin_secret="ngc_api_key", idempotency=present,
          expected=[_stdout("Login Succeeded")], cleanup=["docker", "logout", "nvcr.io"])
    b.add("I2_renderer_pull", "images", "Pull the renderer by digest.", f"{_R3} section 2",
          ["docker", "pull", reference], requires=gated, idempotency=present, timeout="pull")
    b.add("I3_renderer_inspect", "images", "Record Id, RepoDigests and os/arch.",
          f"{_R3} section 2", inspect, requires=gated, expected=[_stdout(pattern)])
    b.add("I4_ngc_logout", "images", "Remove the stored NGC credential.", f"{_R3} section 2",
          ["docker", "logout", "nvcr.io"])


def _phase_assets(b):
    p, co = b.paths, b.paths["alpagym_checkout"]
    download = ["uv", "run", "--no-sync", "hf", "download"]
    teacher = b.assets["teacher"]
    rules = [_file_rule("teacher_snapshot", f"{p['teacher_snapshot']}/{f['path']}",
                        f["size_bytes"], f["sha256"]) for f in teacher["files"]]
    b.add("M1_teacher_download", "assets", "GATE: the exact teacher file list at the pinned "
          "revision into a local snapshot (ONBOARDING downloads without a revision).",
          f"{_R3} section 2 (download with revision=7aba829; hash shards, index, config)",
          [*download, teacher["repository"], *[f["path"] for f in teacher["files"]],
           "--revision", teacher["revision"], "--local-dir", p["teacher_snapshot"]],
          cwd=co, requires=("host_setup", "terms:teacher", "acquire:teacher"), expected=rules,
          idempotency=rules, timeout="teacher", env_extra={"HF_HOME": p["hf_home"]},
          first_delivery=True)
    reason = b.assets["cosmos_reason2"]
    rules = [_file_rule("hf_home", _hf_cache_path(p["hf_home"], reason["repository"],
                                                  reason["revision"], f["path"]),
                        f["size_bytes"], f["sha256"]) for f in reason["files"]]
    b.add("M2_cosmos_reason2_cache", "assets", "GATE: pre-cache the gated Cosmos-Reason2-8B "
          "snapshot at the admission-time revision (converter and every model construction "
          "load it).", f"{_R3} section 1 finding 8 and section 2 (pin via offline HF cache)",
          [*download, reason["repository"], *[f["path"] for f in reason["files"]],
           "--revision", reason["revision"]], cwd=co,
          requires=("host_setup", "terms:cosmos_reason2", "acquire:cosmos_reason2"),
          secret_env={"HF_TOKEN": "hf_token"}, expected=rules, idempotency=rules,
          timeout="small_download", env_extra={"HF_HOME": p["hf_home"]}, level="L3")


def _phase_convert(b):
    p, co = b.paths, b.paths["alpagym_checkout"]
    config = {"kind": "file_regex", "root": "converted_bundle",
              "path": f"{p['converted_bundle']}/config.json",
              "pattern": r'"vlm_name_or_path":\s*"' + re.escape(
                  pins.COSMOS_REASON2["repository"]) + '"'}
    b.add("C1_convert_release", "convert", "Convert the release checkpoint to the AlpaGym "
          "bundle (rewrites config, symlinks weights, builds the processor offline).",
          f"{_R3} section 3 step 4; alpagym {CONVERTER}:38,89-100,136-188",
          ["uv", "run", "--no-sync", "python", f"{co}/{CONVERTER}", "--input",
           p["teacher_snapshot"], "--output", p["converted_bundle"], "--vlm-name-or-path",
           pins.COSMOS_REASON2["repository"], "--overwrite"], cwd=co, expected=[config],
          timeout="import", env_extra={"HF_HOME": p["hf_home"], "HF_HUB_OFFLINE": "1"},
          level="L3")


def _phase_nccl(b, gpus):
    co = b.paths["alpagym_checkout"]
    b.add("N1_nccl_p2p", "nccl", "NCCL peer-to-peer check across both GPUs (a failure is "
          "recorded; the NCCL_P2P_DISABLE=1 fallback is an operator decision, never automatic).",
          f"{_R3} section 3 step 5", ["uv", "run", "--no-sync", "torchrun",
                                      f"--nproc-per-node={gpus}", NCCL_CHECK],
          cwd=co, timeout="import", level="L3")


def _phase_record(b):
    co = alpasim_cached_checkout(b.paths["alpasim_cache_dir"])
    anchor = "alpagym alpasim_dependency.py:37-41 (content-addressed cached checkout)"
    b.add("R1_alpasim_head", "record", "AlpaGym's cached AlpaSim checkout is at the pin.",
          f"{anchor}; {_R3} section 2 (cache reused without integrity check)",
          ["git", "-C", co, "rev-parse", "HEAD"],
          expected=[_stdout(r"\A" + TRAINING_ALPASIM["commit"] + r"\n\Z")], level="L3")
    b.add("R2_alpasim_tree", "record", "Its tree equals 10cae1c's tree (content identity).",
          f"{anchor}; {_R3} section 2 (tree hash, 10cae1c equivalence)",
          ["git", "-C", co, "rev-parse", "HEAD^{tree}"],
          expected=[_stdout(r"\A" + TRAINING_ALPASIM["tree"] + r"\n\Z")], level="L3")
    b.add("R3_alpasim_venv_freeze", "record", "Record the venv AlpaGym built in that checkout "
          "(no upstream lock; resolved at AlpaGym's first build).",
          f"{anchor}; {_R3} section 2 and finding 12 (record uv pip freeze of <checkout>/.venv)",
          ["uv", "pip", "freeze", "--python", f"{co}/.venv/bin/python"], cwd=co,
          expected=[_stdout(r"\S")], timeout="freeze", level="L3")


def _empty_plan(scope=None):
    return {
        "artifact_kind": PLAN_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
        "compiled": False, "producer": _producer(),
        "inputs": {"inventory_sha256": None, "action_record_sha256": None,
                   "rental_record_sha256": None},
        "phases": [], "gpu_count": None, "roots": {}, "authorizations": {},
        "secrets_required": [], "stages": [], "issues": [],
        "execution_status": "qualify_on_host", "level": "L2", "units": dict(PLAN_UNITS),
        "executable_by": None, "compiler": compiler_identity("s4_runtime"),
    }


def build_s4_bootstrap_plan(inventory, action_record, rental_record, *, phases):
    """Compile the S4 two-GPU stages or refuse with issues and no stages (never raises)."""
    try:
        return _build(inventory, action_record, rental_record, phases)
    except (TypeError, AttributeError, KeyError, ValueError) as exc:
        plan = _empty_plan()
        plan["issues"].append(_issue("invalid_input", "inputs", "Unexpected value type in the "
                                     f"inputs ({type(exc).__name__})."))
        return _seal(plan)


def _build(inventory, action_record, rental_record, phases):
    plan = _empty_plan()
    for name, value in (("inventory_sha256", inventory),
                        ("action_record_sha256", action_record)):
        try:
            plan["inputs"][name] = identity_hash(value)
        except (TypeError, ValueError):
            plan["inputs"][name] = None
    report = validate_s4_inventory(inventory)
    plan["scope"] = report["scope"]
    issues = plan["issues"]
    if not report["valid"]:
        bad = sorted({i["code"] for i in report["issues"] if i["severity"] == "error"})
        issues.append(_issue("inventory_invalid", "inventory",
                             f"Inventory has structural errors: {bad}."))
    issues.extend(_validate_record(action_record))
    gpus, rental_hash = _rental_gpu_count(rental_record, report["scope"], issues)
    plan["inputs"]["rental_record_sha256"] = rental_hash
    normalized = _normalize_phases(phases)
    if normalized is None:
        issues.append(_issue("invalid_phases", "phases",
                             f"phases is a non-empty list of distinct names from {list(PHASES)}."))
    if issues:
        return _seal(plan)
    plan["phases"], plan["gpu_count"] = normalized, gpus
    unauthorized = [p for p in normalized if p not in action_record["authorized_phases"]]
    if unauthorized:
        issues.append(_issue("phase_not_authorized", "action_record.authorized_phases",
                             f"Phases {unauthorized} lack recorded authority."))
        return _seal(plan)
    builder = _Builder(inventory, action_record, report["disk_demand"]["unknown_components"])
    _needed_inputs(builder, normalized)
    if builder.issues:
        issues.extend(builder.issues)
        return _seal(plan)
    compilers = {
        "host": lambda: _phase_host(builder, gpus, report["disk_demand"]["unknown_components"]),
        "alpagym": lambda: _phase_alpagym(builder), "alpasim": lambda: _phase_alpasim(builder),
        "images": lambda: _phase_images(builder), "assets": lambda: _phase_assets(builder),
        "convert": lambda: _phase_convert(builder), "nccl": lambda: _phase_nccl(builder, gpus),
        "record": lambda: _phase_record(builder),
    }
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
    problems.extend(_attestation_issues(builder.used_auth, rental_record))
    if problems:
        issues.extend(problems)
        return _seal(plan)
    if "convert" in normalized:
        candidate["issues"] = [_issue(
            "hf_offline_cache_resolution_unverified", "stages.C1_convert_release",
            "M2 fetches Cosmos-Reason2-8B by commit, which may not create refs/main; the "
            "offline converter load by repo name must be tested on the host (INFERRED).",
            "info")]
    candidate["compiled"] = True
    candidate["executable_by"] = "scripts/s2_guest_bootstrap.py"
    return _seal(candidate)


def _attestation_issues(used_auth, rental_record):
    """Every licence the plan consumes is attested by the rental action record (B3-19)."""
    attested = set(rental_record["license_attestation_refs"])
    missing = sorted(key for key, ref in used_auth.items()
                     if key.startswith("terms:") and ref not in attested)
    if not missing:
        return []
    return [_issue("license_attestation_missing", "rental_record.license_attestation_refs",
                   f"The rental record does not attest the terms the plan consumes: {missing}.")]


_PLAN_FIELDS = frozenset(_empty_plan()) | {"plan_sha256"}


def plan_content_issues(plan):
    """Re-run the S2 content validator and the seal check on an S4 plan (L1; never raises)."""
    if type(plan) is not dict or plan.get("artifact_kind") != PLAN_KIND:
        return [_issue("invalid_plan", "plan", f"Not an {PLAN_KIND}.")]
    if set(plan) != _PLAN_FIELDS:
        return [_issue("invalid_plan", "plan", f"Plan fields are exactly {sorted(_PLAN_FIELDS)}.")]
    issues = []
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    try:
        if identity_hash(body) != plan.get("plan_sha256"):
            issues.append(_issue("plan_hash_mismatch", "plan_sha256", "Content differs from "
                                                                      "its seal."))
    except (TypeError, ValueError):
        return [_issue("invalid_plan", "plan", "Plan is not strict JSON.")]
    if plan.get("compiled") is not True:
        issues.append(_issue("plan_not_compiled", "compiled", "Refused plans never execute."))
        return issues
    try:
        issues.extend(i for i in _plan_content_issues(plan) if i["severity"] == "error")
    except (TypeError, AttributeError, KeyError, ValueError) as exc:
        issues.append(_issue("invalid_plan", "plan",
                             f"Plan content is malformed ({type(exc).__name__})."))
    return issues
