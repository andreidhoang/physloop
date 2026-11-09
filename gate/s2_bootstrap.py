"""S2 guest bootstrap plan compiler and output-verification rules (R2 phases A-E).

Tickets: CPU-09 (reproducible setup plan and staged bootstrap), runbook section 4 steps 4-6,
R2 section (1) phases A-E (``docs/research/2026-11-08-codebase-build/lane-r2-alpasim-runtime.md``).
Agent-authored under D-018/D-024; engineering preparation for S2.1/S2.2/S2.7.

Verification ceiling:
- ``build_execution_plan`` and ``validate_plan``: **L1** compile/refusal logic with literal known
  answers; every emitted command is **L2** (transcribed from R2 phases A-E and the pinned AlpaSim
  ``e121e59`` source cited in each stage's ``source_anchor``; a flag absent from R2 cites the
  live-run probe line it comes from, and the one deliberate change - A7a writes the rustup
  installer to a file with ``-o`` instead of R2's pipe to ``sh`` so its bytes are archived - is
  labelled in the anchor). Stages whose
  expectation R2 marks INFERRED carry ``level: "L3"``. Executing any stage is **L3**: the plan
  reports ``execution_status: "qualify_on_host"``, never ``passed``.
- ``evaluate_rule``: **L1** for file, lock, archive, tree, zip, scene-catalog, regex, version,
  ``source_tree`` (F3 sealed bundle) and ``host_fit`` (F3 delivered host; the probe text it
  reads is L3) rules on local bytes; ``probe`` rules are L1 logic around an injected runner whose
  real behaviour is L3. It never raises: any exception becomes a failed result.
- F3 guest-runtime stages A6a-A6i (``_guest_runtime``): **L2** argv checked against the uv
  0.12.24 CLI source (``crates/uv-cli/src/lib.rs``: ``--frozen``, ``--no-dev``, ``--no-sync``,
  ``--no-install-project``, ``--no-build``, global ``--managed-python``/``--no-python-downloads``
  /``--no-config``, ``python install --no-bin``) and its release ``uv-installer.sh`` (``tar xf
  --no-same-owner --strip-components 1``); A6e's install path is INFERRED (``L3``).

Gates: a plan compiles all-or-nothing. It refuses (``compiled: false``, no stages) on an invalid
inventory, an invalid action record, a requested phase the record does not authorize, any gated
stage without both its inventory terms reference (``terms:<role>``) and the record's acquisition
authority (``acquire:<role>``), any owner input the requested phases need (Harmonizer source,
the five scene rows), a missing toolchain/path/image reference, known component sizes above the
recorded disk capacity, or anything credential-like in argv/env. Phase E under the upstream
Harmonizer default still requires ``terms:harmonizer`` (each on-arm renderer downloads it). Secrets never enter the plan: the HF token reaches a download only as ``HF_TOKEN``
injected by the guest runner from a file named at run time (alpasim ONBOARDING.md:17), and the
NGC key only through ``docker login --password-stdin`` from a file (R2 D1).

Agent technical defaults (D-018), labelled so Huy sees what he inherits: stage timeouts in
``TIMEOUTS``; the guest ``PATH`` (``<home>/.local/bin`` and ``<home>/.cargo/bin`` first, then
the Ubuntu system directories); the Cosmos-Reason2-8B / Qwen3-VL-2B file lists, which are every
non-weight, non-documentation file the inventory lists; the probe image (an inventory input);
``HOST_FILESYSTEM_FLOOR_BYTES`` = 285 GB, the filesystem-size floor A9b checks on the data_root
and docker_data_root filesystems (95% of the 300 GB request floor, leaving room for filesystem
metadata and reserved blocks). A9b verifies known demand against free space and records every
unknown component as ``demand_status: "unverified"``; it never treats an unknown size as zero.

Attempt-scoped stages (``attempt_scoped: true``, only A1) carry ``ATTEMPT_TOKEN`` in argv and rule
paths; the guest runner substitutes the attempt number, because the host probe refuses an existing
output directory (exit 73) and a retried probe needs a fresh one.

Registered plan kinds (A2X): ``PLAN_KINDS`` maps each inventory kind to its one compiler -
``s2_runtime`` -> ``build_execution_plan`` here, ``s2_reexport`` -> ``gate.s2_reexport.
build_reexport_plan`` (the literal ``gate.s2_reexport.PLAN_KIND_SPEC``), ``s4_runtime`` ->
``teacher_rl.ops.build_s4_bootstrap_plan`` (both build their stages with this module's
``_Builder``). Every plan
records ``compiler`` = ``{"inventory_kind", "function"}``; ``validate_plan`` refuses an unknown
kind (``unknown_plan_kind``) or a plan whose compiler identity or ``artifact_kind`` is not the
registered one (``compiler_mismatch``) and applies that kind's field set and checkout root. The
single guest runner recompiles through ``resolve_compiler(kind)``. Compilers are dotted import
paths resolved lazily because ``teacher_rl.ops`` imports this module.

Guest runtime (F3; completeness-audit CPU-09 "guest runtime"). Phase A also provisions the
PhysLoop runner itself, because the guest's system ``python3`` (Ubuntu 22.04: 3.10) cannot run
the >= 3.11 modules and has no PyYAML/protobuf. After the host prerequisites, A6a-A6c fetch the
pinned uv archive over HTTPS (``--proto =https --proto-redir =https``), check its published
SHA-256 and size, extract it and install ``uv`` into ``<home>/.local/bin``; A6d checks the
exact version; A6e installs the uv-managed ``cpython-3.12.11-linux-x86_64-gnu`` (uv verifies
the download against its own embedded hash); A6f/A6g verify the transferred sealed source
bundle (``scripts/s2_prepare_bundle.py``) and unpack it over ``<physloop_root>`` with
``--touch`` (files get the unpack time, not the archive's mtime 0), then the ``source_tree``
rule requires the tree to equal the bundle manifest exactly (manifest identity = the inventory
``bundle_sha256``, every listed file's size and sha256, no symlink, no unlisted file apart from
``.venv`` and ``__pycache__``, and no ``__pycache__`` bytecode CPython would load in place of a
sealed source, ``stale_bytecode``); A6h runs ``uv sync --frozen --no-dev
--no-install-project --no-build --managed-python --no-python-downloads --no-config --python
cpython-3.12.11-linux-x86_64-gnu`` (main group only: the ``dev`` and ``s4`` groups stay off the
S2 guest; wheels only; the project itself is not built, so no unpinned build backend is
fetched - the form of the 2026-11-08 clean frozen rebuild); A6i imports ``gate.s2_executor``,
``gate.s2_capture``, ``yaml``, ``pyarrow`` and ``google.protobuf`` through ``GUEST_PYTHON`` and
requires the pinned interpreter, the locked versions, the venv under ``<physloop_root>``, the
tree's own ``gate`` package and an absent ``torch``.

Chicken and egg, stated: the runner lives in the bundle, so the operator first extracts the
same archive into ``<physloop_root>`` (``tar -x --touch``, so bytecode left from an earlier tree
cannot validate) and starts ``python3 scripts/s2_guest_bootstrap.py`` on the system interpreter
for phase A. That graph (this module, ``gate.s2_runtime``, ``gate.s2_identity``,
``gate.s2_host_fit`` and the runner) must import on Python 3.10 at every depth, lazy imports
included; ``tests/test_s2_bootstrap.py`` checks every import node, and ``tomllib`` is the one
function-level import allowed (the lock rule fails closed with ``tomllib_unavailable`` without
it). Open item, stated: the 2026-11-09 review (F3-1) found ``scripts/s2_guest_bootstrap.py``
``main`` lazily importing ``datetime.UTC`` (3.11+), so phase A does not yet start on 3.10; the
fix is a pending shared change to that file and the guard stays red until it lands. Residual,
stated: the phase-A runner process itself imports before A6g, so if the operator's own
extraction omits ``--touch`` it may run stale bytecode that A6g's later check cannot undo.
A6g re-extracts the verified archive over that tree (identical bytes) and proves the tree is
exactly the sealed one. Every
later guest-side PhysLoop command - phases B-E of this runner included - runs as
``GUEST_PYTHON + [script, ...]`` = ``uv run --frozen --no-dev --no-sync --no-config python ...``
with ``cwd`` = ``<physloop_root>``. ``--no-sync`` is a labelled deviation from a bare
``uv run --frozen --no-dev``: without it uv would sync and install the editable project, which
fetches an unpinned ``hatchling`` at run time; the environment is the one A6h synced and A6i
verified.

Delivered-host fit (F3). A1's expected outputs (and idempotency check) carry a ``host_fit`` rule
that reads the probe's own attempt directory and compares its facts with the action record's
``required_resources`` through ``gate.s2_host_fit.assess_host_fit``; any shortfall or unverified
fact fails A1 and the run stops. Nothing selects or rents a replacement host. The rule lives on
A1 because only the attempt-scoped stage knows its attempt directory.

The module never runs a process, opens a network connection, reads a credential, provisions or
funds anything, accepts terms or chooses an owner value. ``evaluate_rule`` reads local files only
under declared roots and runs a ``probe`` argv only through the runner it is given.
"""

import csv
import hashlib
import importlib
import io
import os
import re
import stat
import zipfile
from collections import Counter, namedtuple
from pathlib import Path, PurePosixPath

from gate.s2_host_fit import PROBE_FILES, assess_host_fit, validate_requirements
from gate.s2_identity import identity_hash, read_json_bytes
from gate.s2_runtime import (
    ALPAMAYO15_REPOSITORY,
    ALPAMAYO15_SHA,
    ALPASIM_REPOSITORY,
    ALPASIM_SHA,
    HOST_PATHS,
    LOCKED_RUNTIME,
    PYTHON_REQUEST,
    PYTHON_VERSION,
    SOURCE_MANIFEST_NAME,
    SOURCE_UNTRACKED_DIRS,
    SOURCE_UNTRACKED_TOP_DIRS,
    UV_ARCHIVE_SHA256,
    UV_ARCHIVE_SIZE_BYTES,
    UV_ARCHIVE_URL,
    UV_TARGET,
    UV_VERSION,
    is_clean_absolute,
    is_placeholder,
    is_portable_absolute,
    is_safe_relative,
    paths_overlap,
    secret_like,
    source_bundle_archive_name,
    validate_runtime_inventory,
    validate_source_manifest,
)

RunResult = namedtuple("RunResult", "returncode stdout stderr timed_out")

PHASES = ("A", "B", "C", "D", "E")
AUTHORIZATION_KEYS = (
    "host_setup", "acquire:renderer", "acquire:teacher", "acquire:vlm_config",
    "acquire:nurec_scenes", "acquire:harmonizer",
)
SECRET_NAMES = ("hf_token", "ngc_api_key")
ENV_ALLOWLIST = ["LANG", "LC_ALL"]
# Agent technical defaults (D-018), seconds.
TIMEOUTS = {
    "check": 120, "apt": 1_200, "probe_script": 1_980, "fetch": 600, "rustup": 1_200,
    "clone": 1_200, "lock": 1_800, "sync": 3_600, "protos": 600, "import": 900,
    "pull": 3_600, "build": 7_200, "freeze": 600, "teacher": 14_400, "small_download": 1_200,
    "scene": 3_600, "harmonizer": 3_600, "verify": 600,
}
HOOD_FILES = (
    ("hyperion_8_1/camera_front_wide_120fov.png", 425_624,
     "b7c36c0d179e82cc0d820a202fb9b0017e4006aa815cc850fbd3ea1848d9a8cd"),
    ("hyperion_8/camera_front_wide_120fov.png", 808_775,
     "04fa3f354940d03984a37eb755ebf845be4b4bdae39c178b8de990d0b045392d"),
)
IMPORT_CHECK = (
    "import torch,transformers,alpamayo1_5,alpasim_driver; print(torch.__version__,"
    "torch.version.cuda,torch.cuda.get_arch_list(),transformers.__version__)"
)
INSPECT_FORMAT = "{{.Id}} {{json .RepoDigests}} {{.Os}}/{{.Architecture}}"
LOCAL_INSPECT_FORMAT = "{{.Id}} {{.Os}}/{{.Architecture}}"
RULE_KINDS = (
    "file", "file_regex", "uv_lock_git_rev", "usdz_metadata_uuid", "archive_file", "tree",
    "free_space", "scene_catalog", "stdout_regex", "stdout_min_version", "probe", "host_fit",
    "source_tree",
)
# F3: every guest-side PhysLoop command after A6 runs through the frozen, verified environment
# (cwd = <physloop_root>); --no-sync keeps uv from building the editable project at run time.
GUEST_PYTHON = ("uv", "run", "--frozen", "--no-dev", "--no-sync", "--no-config", "python")
UV_SYNC = ("uv", "sync", "--frozen", "--no-dev", "--no-install-project", "--no-build",
           "--managed-python", "--no-python-downloads", "--no-config", "--python", PYTHON_REQUEST)
RUNTIME_IMPORT_CHECK = (
    "import importlib.util,os,sys\n"
    "import gate.s2_capture,gate.s2_executor,google.protobuf,pyarrow,yaml\n"
    "print(sys.version.split()[0],yaml.__version__,pyarrow.__version__,"
    "google.protobuf.__version__,sys.prefix,os.path.dirname(gate.s2_executor.__file__),"
    "'torch='+('absent' if importlib.util.find_spec('torch') is None else 'present'))"
)
ATTEMPT_TOKEN = "__ATTEMPT__"
# Agent technical default (D-018): 95% of the 300 GB request floor (filesystem overhead).
HOST_FILESYSTEM_FLOOR_BYTES = 285_000_000_000
UNITS = {"timeout_s": "seconds", "size_bytes": "bytes", "bytes": "bytes",
         "filesystem_floor_bytes": "bytes"}
# alpasim@e121e59 src/wizard/configs/nurec_scenes/curated_val.yaml:9-11 (the launch preset,
# gate/s2_launch.py::_command_group); sha256 of the git blobs 5cc40ad1 and 0dabd6be at the pin.
SCENE_CATALOGS = (
    ("data/scenes/sim_scenes.csv",
     "b1cc2d2fd0235fcfbaac7258a2943e1076376f7f1d51c64553e1a73b5797a7e0"),
    ("data/scenes/sim_scenes_2604.csv",
     "cf0df2b74d2129a3a43d0a4fe5d120b73bf4f63ee70fa7d534db5635eedbf872"),
)
# alpasim@e121e59 src/wizard/alpasim_wizard/scenes/csv_utils.py:33-41.
SCENES_COLUMNS = [
    "uuid", "scene_id", "nre_version_string", "path", "last_modified", "artifact_repository",
    "hf_revision",
]
_STAGE_FIELDS = (
    "id", "phase", "purpose", "source_anchor", "argv", "cwd", "env_allowlist", "env_set",
    "secret_env", "stdin_secret", "requires", "expected_outputs", "idempotency_check",
    "cleanup_argv", "timeout_s", "level", "records_first_delivery", "reviewed_script",
    "attempt_scoped",
)
_PLAN_FIELDS = {
    "artifact_kind", "schema_version", "scope", "compiled", "producer", "inputs", "phases",
    "roots", "authorizations", "secrets_required", "stages", "issues", "execution_status",
    "level", "units", "plan_sha256", "compiler",
}
# A2X: one registered compiler per inventory kind; the guest runner recompiles only through it.
PLAN_KINDS = {
    "s2_runtime": {
        "inventory_artifact_kind": "s2_runtime_inventory",
        "plan_artifact_kind": "s2_bootstrap_plan",
        "function": "gate.s2_bootstrap.build_execution_plan",
        "inputs": ("inventory", "action_record"),
        "checkout_root": "checkout",
        "extra_fields": (),
    },
    # Literal copy of gate.s2_reexport.PLAN_KIND_SPEC (that module imports this one).
    "s2_reexport": {
        "inventory_artifact_kind": "s2_reexport_inventory",
        "plan_artifact_kind": "s2_reexport_plan",
        "function": "gate.s2_reexport.build_reexport_plan",
        "inputs": ("inventory", "action_record"),
        "checkout_root": "alpagym_checkout",
        "extra_fields": (),
    },
    "s4_runtime": {
        "inventory_artifact_kind": "s4_runtime_inventory",
        "plan_artifact_kind": "s4_bootstrap_plan",
        "function": "teacher_rl.ops.build_s4_bootstrap_plan",
        "inputs": ("inventory", "action_record", "rental_record"),
        "checkout_root": "alpagym_checkout",
        "extra_fields": ("executable_by", "gpu_count"),
    },
}
_ACTION_FIELDS = {
    "artifact_kind", "schema_version", "action_record_ref", "authorized_phases", "authorizations",
    "required_resources",
}
_FORBIDDEN_FLAGS = ("--token", "--password", "--hf-token")
_ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
_STAGE_ID = re.compile(r"[A-Za-z0-9_]{1,64}")
_MAX_TEXT_READ = 64 * 1024
_MAX_LOCK_READ = 16 * 1024 * 1024
_MAX_METADATA_READ = 1024 * 1024
_MAX_CATALOG_READ = 16 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_UUID_LINE = re.compile(r"""uuid:[ \t]*(['"]?)([^'"\s#]+)\1[ \t]*(?:#.*)?""")


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


def compiler_identity(kind):
    """The ``compiler`` record a plan of a registered kind carries (L1)."""
    return {"inventory_kind": kind, "function": PLAN_KINDS[kind]["function"]}


def plan_fields(kind):
    """Exact top-level field set of a plan of a registered kind (L1)."""
    return frozenset(_PLAN_FIELDS | set(PLAN_KINDS[kind]["extra_fields"]))


def plan_kind(plan):
    """``(kind, issues)``: the registered kind a plan names, or a refusal (L1; never raises)."""
    compiler = plan.get("compiler") if type(plan) is dict else None
    kind = compiler.get("inventory_kind") if type(compiler) is dict else None
    if type(kind) is not str or kind not in PLAN_KINDS:
        return None, [_issue("unknown_plan_kind", "compiler.inventory_kind",
                             f"Registered plan kinds are {sorted(PLAN_KINDS)}.")]
    if compiler != compiler_identity(kind) or (
            plan.get("artifact_kind") != PLAN_KINDS[kind]["plan_artifact_kind"]):
        return None, [_issue("compiler_mismatch", "compiler",
                             f"A {kind} plan is a {PLAN_KINDS[kind]['plan_artifact_kind']} "
                             f"compiled by {PLAN_KINDS[kind]['function']}.")]
    return kind, []


def resolve_compiler(kind):
    """Import the registered compiler of ``kind`` (never a name taken from a plan)."""
    module, _, name = PLAN_KINDS[kind]["function"].rpartition(".")
    return getattr(importlib.import_module(module), name)


def version_tuple(text):
    """Dotted numeric version to a 3+-tuple padded with zeros ("580" -> (580, 0, 0))."""
    parts = tuple(int(part) for part in text.split("."))
    return parts + (0,) * max(0, 3 - len(parts))


def _issue(code, path, message, severity="error"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


# --- plan compilation -------------------------------------------------------------------


class _Builder:
    def __init__(self, inventory, record, unknown_components):
        self.inv = inventory
        self.unknown_components = sorted(unknown_components)
        self.paths = inventory["host_paths"]
        self.record = record
        self.issues = []
        self.stages = []
        self.used_auth = {}
        self.images = {i["role"]: i for i in inventory["images"]}
        self.assets = {a["role"]: a for a in inventory["assets"]}
        home = self.paths["home"]
        self.env = {
            "HOME": home,
            "PATH": f"{home}/.local/bin:{home}/.cargo/bin:/usr/local/sbin:/usr/local/bin:"
                    "/usr/sbin:/usr/bin:/sbin:/bin",
        }

    def refuse(self, code, path, message):
        self.issues.append(_issue(code, path, message))

    def require(self, key, stage_id):
        """Resolve an authorization key from the inventory (terms) or action record (acquire)."""
        if key.startswith("terms:"):
            role = key.split(":", 1)[1]
            item = self.images.get(role) or self.assets.get(role)
            ref = item["authorization_ref"]
            index = (self.inv["images"] if role in self.images else self.inv["assets"]).index(item)
            where = (f"inventory.{'images' if role in self.images else 'assets'}[{index}]"
                     ".authorization_ref")
        else:
            ref = self.record["authorizations"].get(key)
            where = f"action_record.authorizations.{key}"
        if ref is None:
            self.refuse("authorization_missing", where,
                        f"Stage {stage_id} needs recorded authority {key}; refusing to compile.")
            return
        self.used_auth[key] = ref

    def add(self, stage_id, phase, purpose, anchor, argv, *, cwd=None, requires=("host_setup",),
            expected=(), idempotency=None, timeout="check", level="L2", env_extra=None,
            secret_env=None, stdin_secret=None, cleanup=None, first_delivery=False,
            reviewed=False, attempt_scoped=False):
        requires = sorted(set(requires))
        for key in requires:
            self.require(key, stage_id)
        env = dict(self.env)
        env.update(env_extra or {})
        self.stages.append({
            "id": stage_id, "phase": phase, "purpose": purpose, "source_anchor": anchor,
            "argv": list(argv) if argv is not None else None,
            "cwd": (cwd or self.paths["physloop_root"]) if argv is not None else None,
            "env_allowlist": list(ENV_ALLOWLIST) if argv is not None else [],
            "env_set": env if argv is not None else {},
            "secret_env": dict(secret_env or {}), "stdin_secret": stdin_secret,
            "requires": requires, "expected_outputs": list(expected),
            "idempotency_check": list(idempotency) if idempotency is not None else None,
            "cleanup_argv": list(cleanup) if cleanup else None,
            "timeout_s": TIMEOUTS[timeout], "level": level,
            "records_first_delivery": first_delivery, "reviewed_script": reviewed,
            "attempt_scoped": attempt_scoped,
        })


def _file_rule(root, path, size, sha):
    return {"kind": "file", "root": root, "path": path,
            "size_bytes": size if type(size) is int else None,
            "sha256": sha if sha not in (None, "unverified") else None}


def _stdout(pattern, required=True):
    return {"kind": "stdout_regex", "pattern": pattern, "required": required}


def _min_version(pattern, minimum):
    return {"kind": "stdout_min_version", "pattern": pattern, "minimum": minimum}


def _probe(argv, pattern, cwd="/"):
    return {"kind": "probe", "argv": list(argv), "cwd": cwd, "pattern": pattern,
            "timeout_s": TIMEOUTS["check"]}


def _inspect_probe(reference):
    pattern = (r"(?m)^sha256:[0-9a-f]{64} \[[^\]]*\"" + re.escape(reference)
               + r"\"[^\]]*\] linux/amd64$")
    return pattern, ["docker", "image", "inspect", reference, "--format", INSPECT_FORMAT]


def _phase_a(b):
    p = b.paths
    probe_ref = b.images["public_cuda_probe"]["reference"]
    toolchain = b.inv["toolchain"]["rust_toolchain"]
    scratch, evidence, home = p["download_scratch"], p["evidence_dir"], p["home"]
    b.add("A0_directories", "A", "Create the evidence, download, uv extraction and user-binary "
          "directories later steps use.",
          "R2 A1/A7/B2: output directories (install -d as in R2 B2); R2 A6 uv install "
          "directories (<home>/.local/bin is the first PATH entry)",
          ["install", "-d", evidence, scratch, p["data_root"], _uv_dir(b), f"{home}/.local/bin"])
    probe_dir = f"{evidence}/host-probe-attempt-{ATTEMPT_TOKEN}"
    probe_pass = {"kind": "file_regex", "root": "evidence_dir", "path": f"{probe_dir}/result.txt",
                  "pattern": r"\ADEPLOYMENT_PASS_MONITORING_REVIEW_REQUIRED\n\Z"}
    fit = {"kind": "host_fit", "root": "evidence_dir", "path": probe_dir,
           "requirements": b.record["required_resources"]}
    b.add("A1_host_probe", "A", "Record the host inventory and public synthetic GPU checks "
          "(one new output directory per attempt; the probe refuses an existing one), then "
          "require the delivered host to fit the action record's required resources.",
          "R2 A1 + H1-H3: physloop:scripts/vast_s2_host_probe.sh::public_image_pull (digest-only "
          "pull); physloop:scripts/vast_s2_host_probe.sh::out (mkdir of the output directory "
          "exits 73 if it exists); host fit physloop:gate/s2_host_fit.py::assess_host_fit "
          "(no automatic replacement host)",
          ["bash", f"{p['physloop_root']}/scripts/vast_s2_host_probe.sh", probe_dir, probe_ref,
           p["data_root"]],
          expected=[probe_pass, fit], idempotency=[probe_pass, fit], timeout="probe_script",
          level="L3", reviewed=True, attempt_scoped=True)
    b.add("A2a_apt_update", "A", "Refresh apt indexes before installing host prerequisites.",
          "R2 A2; alpasim@e121e59:Dockerfile:18 (apt-get update && apt-get install -y)",
          ["apt-get", "update"], timeout="apt")
    b.add("A2b_apt_install", "A", "Install git/build tools for wizard import git, uv git deps "
          "and maturin.", "R2 A2; src/wizard/alpasim_wizard/wizard.py:13; "
          "src/driver/pyproject.toml:41-44; src/utils_rs/pyproject.toml:1-3",
          ["apt-get", "install", "-y", "git", "build-essential", "ca-certificates", "curl"],
          timeout="apt")
    b.add("A3a_docker_version", "A", "Docker Engine >= 23.0.1 (NRE minimum).",
          "R2 A3; docs/ONBOARDING.md:24-29; --format {{.Server.Version}} as in the live-run "
          "probe physloop:scripts/vast_s2_host_probe.sh::docker_engine", ["docker", "version", "--format",
                                               "{{.Server.Version}}"],
          expected=[_min_version(r"(?m)^(\d+\.\d+\.\d+)", "23.0.1")])
    b.add("A3b_compose_version", "A", "Docker Compose plugin present.",
          "R2 A3; docs/ONBOARDING.md:24-29", ["docker", "compose", "version"],
          expected=[_stdout(r"Docker Compose version v?\d+\.\d+\.\d+")])
    b.add("A3c_buildx_version", "A", "Docker buildx plugin present.",
          "R2 A3; docs/ONBOARDING.md:24-29", ["docker", "buildx", "version"],
          expected=[_stdout(r"v?\d+\.\d+\.\d+")])
    b.add("A4a_nvidia_ctk_version", "A", "NVIDIA Container Toolkit >= 1.13.5.",
          "R2 A4 (NRE requirement)", ["nvidia-ctk", "--version"],
          expected=[_min_version(r"version (\d+\.\d+\.\d+)", "1.13.5")])
    b.add("A4b_gpu_container_check", "A", "GPU visible in a container (public image by digest).",
          "R2 A4: docker run --rm --gpus all <cuda-ref> nvidia-smi",
          ["docker", "run", "--rm", "--gpus", "all", probe_ref, "nvidia-smi"],
          expected=[_stdout("NVIDIA-SMI")], timeout="pull", level="L3")
    b.add("A5_driver_version", "A", "Driver >= R580 (NRE on Blackwell; CUDA 12.8 needs >= 570).",
          "R2 A5; docs/ONBOARDING.md:30-32; NRE hardware page 26.04; --query-gpu=... "
          "--format=csv query form as in physloop:scripts/vast_s2_host_probe.sh::gpu_identity",
          ["nvidia-smi", "--query-gpu=driver_version", "--format=csv"],
          expected=[_min_version(r"(?m)^(\d+\.\d+(?:\.\d+)?)$", "580")])
    _guest_runtime(b)
    rustc = _probe([f"{home}/.cargo/bin/rustc", "-V"],
                   r"(?m)^rustc " + re.escape(toolchain) + r"( |$)")
    b.add("A7a_rustup_fetch", "A", "Fetch rustup-init over TLS and archive its bytes.",
          "R2 A7; setup_local_env.sh:16-41 (curl --proto '=https' --tlsv1.2 -sSf "
          "https://sh.rustup.rs | sh); agent change: -o <file> replaces the pipe to sh so the "
          "installer bytes are archived before A7b runs them",
          ["curl", "--proto", "=https", "--tlsv1.2", "-sSf", "-o",
           f"{scratch}/rustup-init.sh", "https://sh.rustup.rs"],
          expected=[{"kind": "archive_file", "root": "download_scratch",
                     "path": f"{scratch}/rustup-init.sh", "name": "rustup-init.sh"}],
          idempotency=[rustc], timeout="fetch")
    b.add("A7b_rustup_install", "A", "Install the pinned Rust toolchain for utils_rs (maturin).",
          "R2 A7 (sh -s -- -y with pinned --default-toolchain); setup_local_env.sh:22; "
          "src/utils/pyproject.toml:26-29",
          ["sh", f"{scratch}/rustup-init.sh", "-y", "--default-toolchain", toolchain],
          expected=[rustc], idempotency=[rustc], timeout="rustup")
    b.add("A9a_data_root", "A", "Create <DATA_ROOT> on the large volume.",
          "R2 A9 (Docker data root and <DATA_ROOT> on >= 300 GB volume; INFERRED)",
          ["install", "-d", p["data_root"]])
    b.add("A9b_disk_free", "A", "Known byte demand fits the free space per filesystem; the "
          "data and Docker filesystems meet the size floor; unknown sizes are recorded as "
          "unverified, never as zero.",
          "R2 A9 (Docker data root and <DATA_ROOT> on >= 300 GB volume) and sizing", None,
          expected=[{"kind": "free_space", "requirements": _space_requirements(b),
                     "unknown_components": list(b.unknown_components)}],
          timeout="verify")


def _uv_dir(b):
    return f"{b.paths['download_scratch']}/uv-{UV_VERSION}-{UV_TARGET}"


def _guest_runtime(b):
    """F3 stages A6a-A6i: pinned uv, managed CPython, sealed bundle, frozen env, import check."""
    p, runtime = b.paths, b.inv["guest_runtime"]
    scratch, home, root = p["download_scratch"], p["home"], p["physloop_root"]
    archive = f"{scratch}/uv-{UV_VERSION}-{UV_TARGET}.tar.gz"
    extracted = f"{_uv_dir(b)}/uv"
    uv_bin = f"{home}/.local/bin/uv"
    uv_archive = _file_rule("download_scratch", archive, UV_ARCHIVE_SIZE_BYTES, UV_ARCHIVE_SHA256)
    version = r"(?m)^uv " + re.escape(UV_VERSION) + r"( |$)"
    installed = _probe([uv_bin, "--version"], version)
    b.add("A6a_uv_fetch", "A", f"Fetch the pinned uv {UV_VERSION} archive over HTTPS only; its "
          "published SHA-256 and size must match.",
          f"R2 A6 (uv >= 0.9.17); F3 guest runtime: {UV_ARCHIVE_URL} with the sha256 of its "
          "published .sha256 asset (gate.s2_runtime UV_ARCHIVE_SHA256); curl --proto/--tlsv1.2 "
          "as in A7a, plus -L and --proto-redir =https for the release CDN redirect",
          ["curl", "--proto", "=https", "--proto-redir", "=https", "--tlsv1.2", "-sSfL", "-o",
           archive, UV_ARCHIVE_URL],
          expected=[uv_archive], idempotency=[uv_archive], timeout="fetch")
    b.add("A6b_uv_extract", "A", "Extract the verified archive (one top-level directory).",
          "R2 A6; F3: tar xf --no-same-owner --strip-components 1 as the release's own "
          "uv-installer.sh does for the tar.gz",
          ["tar", "-x", "-z", "-f", archive, "-C", _uv_dir(b), "--no-same-owner",
           "--strip-components", "1"],
          expected=[uv_archive, _file_rule("download_scratch", extracted, None, None)],
          timeout="check")
    b.add("A6c_uv_install", "A", "Install the extracted uv binary first on PATH.",
          "R2 A6; F3: install -m 0755 into <home>/.local/bin (the builder's first PATH entry)",
          ["install", "-m", "0755", extracted, uv_bin],
          expected=[installed], idempotency=[installed])
    b.add("A6d_uv_version", "A", f"uv on PATH is exactly {UV_VERSION} (>= 0.9.17).",
          "R2 A6; pyproject.toml:79-80; docs/ONBOARDING.md:18-20", ["uv", "--version"],
          expected=[_stdout(version)])
    python_path = (r"(?m)^" + re.escape(f"{home}/.local/share/uv/python/{PYTHON_REQUEST}/bin/")
                   + r"python3(\.12)?$")
    found = _probe(["uv", "python", "find", "--no-config", "--managed-python",
                    "--no-python-downloads", PYTHON_REQUEST], python_path)
    b.add("A6e_python_install", "A", f"Install the uv-managed CPython {PYTHON_VERSION} "
          "(uv checks the download against its embedded sha256).",
          "R2 A6/C4 (uv-managed CPython 3.12, .python-version); F3: uv python install "
          "--no-bin (no interpreter shims on PATH); install directory INFERRED",
          ["uv", "python", "install", "--no-config", "--no-bin", PYTHON_REQUEST],
          expected=[found], idempotency=[found], timeout="sync", level="L3")
    bundle = runtime["source_bundle"]
    bundle_archive = _file_rule(
        "download_scratch", f"{scratch}/{source_bundle_archive_name(bundle['bundle_sha256'])}",
        bundle["archive_size_bytes"], bundle["archive_sha256"])
    tree = {"kind": "source_tree", "root": "physloop_root", "path": root,
            "bundle_sha256": bundle["bundle_sha256"]}
    b.add("A6f_bundle_archive", "A", "Verify the transferred sealed PhysLoop source bundle "
          "bytes before unpacking.",
          "R2 A6; F3 sealed bundle physloop:scripts/s2_prepare_bundle.py::build_bundle",
          None, expected=[bundle_archive], timeout="verify")
    b.add("A6g_bundle_unpack", "A", "Unpack the verified bundle over <physloop_root> and "
          "require the tree to equal its manifest exactly.",
          "R2 A6; F3 physloop:scripts/s2_prepare_bundle.py::build_bundle (deterministic tar, "
          "manifest physloop-source-manifest.json); --touch so a __pycache__ entry compiled "
          "from an earlier unpack never validates against the new bytes (F3-2 review)",
          ["tar", "-x", "-f", bundle_archive["path"], "-C", root, "--no-same-owner", "--touch"],
          expected=[bundle_archive, tree], idempotency=[tree], timeout="check")
    versions = " ".join(re.escape(version) for _, _, version in LOCKED_RUNTIME)
    imports = (r"\A" + re.escape(PYTHON_VERSION) + " " + versions + " "
               + re.escape(f"{root}/.venv") + " " + re.escape(f"{root}/gate")
               + r" torch=absent\n\Z")
    import_argv = [*GUEST_PYTHON, "-c", RUNTIME_IMPORT_CHECK]
    b.add("A6h_runtime_sync", "A", "Frozen PhysLoop guest environment: main group only (no dev, "
          "no s4), wheels only, project not built, managed interpreter.",
          "R2 A6/C1 (frozen sync); F3; flags as the 2026-11-08 clean frozen rebuild "
          "(evidence/s2-clean-rebuild-20261108-v1) with --no-dev instead of --all-groups",
          list(UV_SYNC), cwd=root, expected=[tree],
          idempotency=[_probe(import_argv, imports, cwd=root)], timeout="sync")
    b.add("A6i_runtime_imports", "A", "The runner's modules and locked dependencies import "
          "under the pinned interpreter from <physloop_root>/.venv; torch is absent.",
          "R2 A6/C3 (import check); F3: gate.s2_executor, gate.s2_capture, yaml, pyarrow, "
          "google.protobuf at their uv.lock versions", import_argv, cwd=root,
          expected=[_stdout(imports)], timeout="import", reviewed=True)


def _space_requirements(b):
    def known(rows):
        return sum(r["size_bytes"] for r in rows if type(r["size_bytes"]) is int)

    p, assets = b.paths, b.assets
    floor = HOST_FILESYSTEM_FLOOR_BYTES
    wanted = [
        ("teacher_snapshot", known(assets["teacher"]["files"]), 0),
        ("hf_home", known(assets["vlm_config"]["files"]) + known(assets["processor"]["files"]),
         0),
        ("data_root", known(assets["nurec_scenes"]["files"]), floor),
        ("docker_data_root", known(b.inv["images"]), floor),
    ]
    if b.inv["harmonizer_source"] == "pinned_mount":
        wanted.append(("harmonizer_dir", known(assets["harmonizer"]["files"]), 0))
    return [{"root": name, "path": p[name], "bytes": size, "filesystem_floor_bytes": minimum}
            for name, size, minimum in wanted if size > 0 or minimum > 0]


def _phase_b(b):
    p = b.paths
    checkout, data = p["checkout"], p["data_root"]
    head_any = {"kind": "file_regex", "root": "checkout", "path": f"{checkout}/.git/HEAD",
                "pattern": r"\A(ref: refs/heads/[A-Za-z0-9._/-]+|[0-9a-f]{40})\n\Z"}
    head_pin = {"kind": "file_regex", "root": "checkout", "path": f"{checkout}/.git/HEAD",
                "pattern": r"\A" + ALPASIM_SHA + r"\n\Z"}
    b.add("B1a_clone", "B", "Clone AlpaSim without LFS smudge.",
          "R2 B1; .gitattributes:1-11 (LFS only tests/*.asl/*.usdz/*.rclog)",
          ["git", "clone", ALPASIM_REPOSITORY, checkout], idempotency=[head_any],
          env_extra={"GIT_LFS_SKIP_SMUDGE": "1"}, timeout="clone")
    b.add("B1b_checkout", "B", "Detach at the reviewed AlpaSim pin.", "R2 B1",
          ["git", "-C", checkout, "checkout", "--detach", ALPASIM_SHA], expected=[head_pin],
          idempotency=[head_pin])
    b.add("B1c_rev_parse", "B", "Record HEAD.", "R2 B1 (record rev-parse HEAD)",
          ["git", "-C", checkout, "rev-parse", "HEAD"],
          expected=[_stdout(r"\A" + ALPASIM_SHA + r"\n\Z")])
    b.add("B1d_status", "B", "Record a clean worktree (uv.lock, .venv and stubs are ignored).",
          "R2 B1 (record status --porcelain); .gitignore:1,39; src/grpc/.gitignore:3-5",
          ["git", "-C", checkout, "status", "--porcelain"], expected=[_stdout(r"\A\Z")])
    hoods = [_file_rule("data_root", f"{data}/nre-artifacts/ego-hoods/{name}", size, sha)
             for name, size, sha in HOOD_FILES]
    b.add("B2a_hood_dir", "B", "Create the NRE artifact directory under <DATA_ROOT>.",
          "R2 B2", ["install", "-d", f"{data}/nre-artifacts"])
    b.add("B2b_hood_copy", "B", "Stage ego hoods; verify the pinned git blob bytes.",
          "R2 B2; base_config.yaml:30-31,125-126,139,287; sensorsim_service.py:205-227",
          ["cp", "-a", f"{checkout}/data/nre-artifacts/ego-hoods", f"{data}/nre-artifacts/"],
          expected=hoods, idempotency=hoods)
    gate = {"kind": "uv_lock_git_rev", "root": "checkout", "path": f"{checkout}/uv.lock",
            "package": "alpamayo1-5", "repository": ALPAMAYO15_REPOSITORY,
            "commit": ALPAMAYO15_SHA}
    b.add("B3a_uv_lock", "B", "Resolve the runtime lock first; GATE alpamayo1_5 == 36aeb4c.",
          "R2 B3; pyproject.toml:45-60,62-110; src/driver/pyproject.toml:12-30,41-47",
          ["uv", "lock"], cwd=checkout, expected=[gate], idempotency=[gate], timeout="lock")
    b.add("B3b_lock_archive", "B", "Archive uv.lock bytes and sha256.", "R2 B3 (archive uv.lock "
          "+ sha256)", None,
          expected=[{"kind": "archive_file", "root": "checkout", "path": f"{checkout}/uv.lock",
                     "name": "alpasim-uv.lock"}, gate], timeout="verify")


def _phase_c(b):
    checkout = b.paths["checkout"]
    toolchain = b.inv["toolchain"]["rust_toolchain"]
    uv_run = ["uv", "run", "--no-sync"]
    b.add("C1_uv_sync", "C", "Frozen host environment (minimum extras wizard + driver).",
          "R2 C1; setup_local_env.sh:76-104", ["uv", "sync", "--frozen", "--extra", "wizard",
                                               "--extra", "driver"],
          cwd=checkout, timeout="sync")
    b.add("C2_compile_protos", "C", "Generate gRPC stubs once (never between units).",
          "R2 C2; src/grpc/scripts/compile_protos.py:35-45; base_config.yaml:154,168,191,214",
          [*uv_run, "compile-protos"], cwd=f"{checkout}/src/grpc",
          expected=[_stdout("Proto compilation completed successfully!")], timeout="protos")
    b.add("C3_import_check", "C", "Import check: torch 2.8.0 cu12.8, transformers 4.57.1, "
          "sm_120 (INFERRED).", "R2 C3", [*uv_run, "python", "-c", IMPORT_CHECK], cwd=checkout,
          expected=[_stdout(r"(?m)^2\.8\.0(\+\S+)? 12\.8 \[[^\]]*'sm_120'[^\]]*\] 4\.57\.1$")],
          timeout="import", level="L3", reviewed=True)
    b.add("C4a_uv_version", "C", "Archive uv --version.", "R2 C4", ["uv", "--version"],
          expected=[_stdout(r"(?m)^uv \d+\.\d+\.\d+")])
    b.add("C4b_python_version", "C", "Archive the managed Python version.",
          "R2 C4; .python-version (3.12)", [*uv_run, "python", "-V"], cwd=checkout,
          expected=[_stdout(r"(?m)^Python 3\.12\.\d+$")])
    b.add("C4c_rustc_version", "C", "Archive rustc -V.", "R2 C4", ["rustc", "-V"],
          expected=[_stdout(r"(?m)^rustc " + re.escape(toolchain) + r"( |$)")])
    b.add("C4d_uv_export", "C", "Archive the frozen requirement export.", "R2 C4",
          ["uv", "export", "--frozen"], cwd=checkout, expected=[_stdout(r"\S")])
    b.add("C4e_uv_pip_freeze", "C", "Archive the installed package set.", "R2 C4",
          ["uv", "pip", "freeze"], cwd=checkout,
          expected=[_stdout(r"(?m)^torch==2\.8\.0(\+\S+)?$"),
                    _stdout(r"(?m)^transformers==4\.57\.1$"),
                    _stdout(r"(?m)^alpamayo1[-_]5 @ git\+" + re.escape(ALPAMAYO15_REPOSITORY)
                            + "@" + ALPAMAYO15_SHA + "$", required=False)])
    b.add("C4f_stub_hashes", "C", "Record generated stub hashes.", "R2 C4 (stub hashes); "
          "src/grpc/scripts/compile_protos.py:24-45", None,
          expected=[{"kind": "tree", "root": "checkout",
                     "path": f"{checkout}/src/grpc/alpasim_grpc/v0",
                     "suffixes": ["_pb2.py", "_pb2_grpc.py", ".pyi"]}], timeout="verify")


def _phase_d(b):
    checkout = b.paths["checkout"]
    renderer = b.images["renderer"]["reference"]
    gated = ("host_setup", "terms:renderer", "acquire:renderer")
    pattern, inspect = _inspect_probe(renderer)
    present = [_probe(inspect, pattern)]
    b.add("D1_ngc_login", "D", "GATE: log in to nvcr.io with Huy's NGC key from stdin.",
          "R2 D1", ["docker", "login", "nvcr.io", "-u", "$oauthtoken", "--password-stdin"],
          requires=gated, stdin_secret="ngc_api_key", idempotency=present,
          expected=[_stdout("Login Succeeded")], cleanup=["docker", "logout", "nvcr.io"])
    b.add("D2a_renderer_pull", "D", "Pull the NRE image by digest.", "R2 D2",
          ["docker", "pull", renderer], requires=gated, idempotency=present, timeout="pull")
    b.add("D2b_renderer_inspect", "D", "Record Id, RepoDigests and os/arch.", "R2 D2 (docker "
          "image inspect); format as physloop:scripts/vast_s2_host_probe.sh::image_identity",
          inspect, requires=gated, expected=[_stdout(pattern)])
    b.add("D2c_ngc_logout", "D", "Remove the stored NGC credential.", "R2 D2 (docker logout)",
          ["docker", "logout", "nvcr.io"])
    base = b.images["alpasim_base"]
    if base["kind"] == "local_build":
        tag = base["build"]["tag"]
        image_id = base["image_id"]
        local_pattern = (r"(?m)^" + (re.escape(image_id) if image_id else r"sha256:[0-9a-f]{64}")
                         + r" linux/amd64$")
        idem = ([_probe(["docker", "image", "inspect", image_id, "--format",
                         LOCAL_INSPECT_FORMAT], local_pattern)] if image_id else None)
        b.add("D3a_base_build", "D", "Build alpasim-base locally (mutable inputs recorded).",
              "R2 D3; Dockerfile:11-58; .dockerignore:1-25; base_config.yaml:25-26; "
              "pyproject.toml:3", ["docker", "build", "-t", tag, checkout], cwd=checkout,
              idempotency=idem, timeout="build")
        b.add("D3b_base_inspect", "D", "Record the built image ID (bare sha256 for launch).",
              "R2 D3 and (4) L2", ["docker", "image", "inspect", tag, "--format",
                                   LOCAL_INSPECT_FORMAT], expected=[_stdout(local_pattern)])
        freeze_ref = tag
    else:
        reference = base["reference"]
        base_pattern, base_inspect = _inspect_probe(reference)
        b.add("D3a_base_pull", "D", "Pull the prebuilt base image by digest.", "R2 D3/D5 "
              "(immutable base reference)", ["docker", "pull", reference],
              idempotency=[_probe(base_inspect, base_pattern)], timeout="pull")
        b.add("D3b_base_inspect", "D", "Record Id, RepoDigests and os/arch.", "R2 D3 (inspect)",
              base_inspect, expected=[_stdout(base_pattern)])
        freeze_ref = reference
    b.add("D4_base_freeze", "D", "In-image package freeze (runtime, evaluator, physics, "
          "controller env).", "R2 D4 (path /repo/.venv/bin/python INFERRED)",
          ["docker", "run", "--rm", "--entrypoint", "uv", freeze_ref, "pip", "freeze",
           "--python", "/repo/.venv/bin/python"], expected=[_stdout(r"\S")], timeout="freeze",
          level="L3")
    b.add("D5_compose_pull_flag", "D", "Record whether compose up supports --pull (never).",
          "R2 D5; deployment/docker_compose.py:115-116; schema.py:150-157",
          ["docker", "compose", "up", "--help"], expected=[_stdout("--pull", required=False)],
          level="L3")


def _hf_cache_path(hf_home, repository, revision, name):
    return f"{hf_home}/hub/models--{repository.replace('/', '--')}/snapshots/{revision}/{name}"


def _phase_e(b):
    p, assets = b.paths, b.assets
    checkout, hf_home = p["checkout"], p["hf_home"]
    download = ["uv", "run", "--no-sync", "hf", "download"]
    hf_env = {"HF_HOME": hf_home}
    teacher = assets["teacher"]
    rules = [_file_rule("teacher_snapshot", f"{p['teacher_snapshot']}/{f['path']}",
                        f["size_bytes"], f["sha256"]) for f in teacher["files"]]
    b.add("E1_teacher_download", "E", "GATE: download the exact teacher file list at the pinned "
          "revision into a local snapshot.",
          "R2 E1; src/driver/src/alpasim_driver/models/alpamayo1_5_model.py:19,73-81",
          [*download, teacher["repository"], *[f["path"] for f in teacher["files"]],
           "--revision", teacher["revision"], "--local-dir", p["teacher_snapshot"]],
          cwd=checkout, requires=("host_setup", "terms:teacher", "acquire:teacher"),
          expected=rules, idempotency=rules, timeout="teacher", env_extra=hf_env)
    for stage_id, role, gated in (("E2a_vlm_config_cache", "vlm_config", True),
                                  ("E2b_processor_cache", "processor", False)):
        item = assets[role]
        rules = [_file_rule("hf_home", _hf_cache_path(hf_home, item["repository"],
                                                      item["revision"], f["path"]),
                            f["size_bytes"], f["sha256"]) for f in item["files"]]
        b.add(stage_id, "E", f"Pre-cache {item['repository']} config/processor/tokenizer files "
              "at the pinned revision (no weights).",
              "R2 E2; alpamayo1.5@36aeb4c:src/alpamayo1_5/models/base_model.py:211,248-266; "
              "helper.py:25,190-199",
              [*download, item["repository"], *[f["path"] for f in item["files"]],
               "--revision", item["revision"]], cwd=checkout,
              requires=(("host_setup", f"terms:{role}", f"acquire:{role}") if gated
                        else ("host_setup",)),
              secret_env={"HF_TOKEN": "hf_token"} if gated else None,
              expected=rules, idempotency=rules, timeout="small_download", level="L3",
              env_extra=hf_env)
    scenes = sorted(assets["nurec_scenes"]["files"], key=lambda row: row["uuid"])
    destination = f"{p['data_root']}/nre-artifacts/all-usdzs"
    scratch = f"{p['download_scratch']}/nurec"
    catalogs = [{"root": "checkout", "path": f"{checkout}/{relative}", "sha256": digest}
                for relative, digest in SCENE_CATALOGS]
    b.add("E3_scene_catalog", "E", "Each registered row is the scene the wizard itself selects "
          "from the pinned curated_val catalogs (scene_id -> newest uuid, path, huggingface).",
          "R2 E3; alpasim@e121e59:src/wizard/configs/nurec_scenes/curated_val.yaml:9-11; "
          "src/wizard/alpasim_wizard/scenes/sceneset.py:34-66,91-95,272-292; "
          "scenes/csv_utils.py:33-41; physloop:gate/s2_launch.py::_command_group argv "
          "+nurec_scenes=curated_val",
          None, expected=[{"kind": "scene_catalog", "catalogs": catalogs,
                           "scene_id": row["scene_id"], "uuid": row["uuid"],
                           "scene_path": row["path"]} for row in scenes], timeout="verify")
    b.add("E3a_scene_dir", "E", "Create the wizard's all-usdzs directory.",
          "R2 E3; sceneset.py:444-481", ["install", "-d", destination])
    gated = ("host_setup", "terms:nurec_scenes", "acquire:nurec_scenes")
    for k, row in enumerate(scenes, start=1):
        fetched = f"{scratch}/{row['path']}"
        placed = f"{destination}/{row['uuid']}.usdz"
        uuid_rule = {"kind": "usdz_metadata_uuid", "root": "data_root", "path": placed,
                     "uuid": row["uuid"]}
        final = [_file_rule("data_root", placed, row["size_bytes"], row["sha256"]), uuid_rule]
        b.add(f"E3_scene{k}_download", "E", f"GATE: download registered scene {row['scene_id']} "
              "at its resolved commit and check metadata.yaml uuid.",
              "R2 E3; sceneset.py:444-481 (metadata.yaml uuid check); csv_utils.py:18,33-41",
              [*download, assets["nurec_scenes"]["repository"], row["path"], "--repo-type",
               "dataset", "--revision", row["revision"], "--local-dir", scratch], cwd=checkout,
              requires=gated, secret_env={"HF_TOKEN": "hf_token"},
              expected=[_file_rule("download_scratch", fetched, row["size_bytes"], row["sha256"]),
                        dict(uuid_rule, root="download_scratch", path=fetched)],
              idempotency=final, timeout="scene", first_delivery=True, env_extra=hf_env)
        b.add(f"E3_scene{k}_place", "E", "Place the artifact as <uuid>.usdz for the wizard.",
              "R2 E3; sceneset.py:494-501,524-560", ["mv", fetched, placed], expected=final,
              idempotency=final, requires=gated)
    if b.inv["harmonizer_source"] == "pinned_mount":
        item = assets["harmonizer"]
        rules = [_file_rule("harmonizer_dir", f"{p['harmonizer_dir']}/{f['path']}",
                            f["size_bytes"], f["sha256"]) for f in item["files"]]
        b.add("E4_harmonizer_download", "E", "GATE: pinned Harmonizer checkpoint for the "
              "--harmonizer-cache mount in both arms (Huy decision L8).", "R2 E4",
              [*download, item["repository"], *[f["path"] for f in item["files"]],
               "--revision", item["revision"], "--local-dir", p["harmonizer_dir"]],
              cwd=checkout, requires=("host_setup", "terms:harmonizer", "acquire:harmonizer"),
              expected=rules, idempotency=rules, timeout="harmonizer", env_extra=hf_env)
    else:
        # Review A2-F12: the per-unit download is licensed use; its terms record lives in the
        # plan (no acquire:harmonizer, since bootstrap acquires nothing).
        b.require("terms:harmonizer", "E4_harmonizer_upstream_default")
        b.issues.append(_issue(
            "harmonizer_downloaded_per_on_arm_unit", "inventory.harmonizer_source",
            "Upstream default: each on-arm renderer container downloads 1.35 GiB from HF main "
            "(INFERRED, R2 E4); no pinned checkpoint is staged.", "info"))
    b.issues.append(_issue(
        "hf_offline_cache_resolution_unverified", "stages.E2",
        "Files fetched by commit may not create refs/main; offline load by repo name must be "
        "tested on the host (R2 E2, INFERRED).", "info"))


def _validate_action_record(record):
    issues = []
    if type(record) is not dict or set(record) != _ACTION_FIELDS:
        return [_issue("invalid_action_record", "action_record",
                       f"Action record fields are exactly {sorted(_ACTION_FIELDS)}.")]
    if (record["artifact_kind"] != "s2_bootstrap_action_record"
            or type(record["schema_version"]) is not int or record["schema_version"] != 1):
        issues.append(_issue("invalid_action_record", "action_record.schema_version",
                             "Only s2_bootstrap_action_record schema_version 1 is known."))

    def bad_ref(value):
        return (type(value) is not str or not value.strip() or is_placeholder(value)
                or secret_like(value))

    if bad_ref(record["action_record_ref"]):
        issues.append(_issue("invalid_action_record", "action_record.action_record_ref",
                             "Recorded, non-placeholder, non-secret reference required."))
    phases = record["authorized_phases"]
    if type(phases) is not list or not all(p in PHASES for p in phases):
        issues.append(_issue("invalid_action_record", "action_record.authorized_phases",
                             f"authorized_phases is a list drawn from {list(PHASES)}."))
    resources = record["required_resources"]
    if resources is not None:
        for item in validate_requirements(resources):
            issues.append(_issue("invalid_action_record", f"action_record.{item['path']}",
                                 item["message"]))
    auths = record["authorizations"]
    if type(auths) is not dict:
        issues.append(_issue("invalid_action_record", "action_record.authorizations",
                             "authorizations is an object."))
    else:
        for key in sorted(auths):
            if key not in AUTHORIZATION_KEYS:
                issues.append(_issue("invalid_action_record", f"action_record.authorizations.{key}",
                                     f"Unknown authorization key; known {list(AUTHORIZATION_KEYS)}."))
            elif bad_ref(auths[key]):
                issues.append(_issue("invalid_action_record", f"action_record.authorizations.{key}",
                                     "Recorded, non-placeholder, non-secret reference required."))
    return issues


def _normalize_phases(phases):
    if type(phases) not in (list, tuple) or not phases:
        return None
    if not all(type(p) is str and p in PHASES for p in phases) or len(set(phases)) != len(phases):
        return None
    return [p for p in PHASES if p in phases]


def _needed_inputs(b, phases):
    inv = b.inv
    for name in HOST_PATHS:
        if b.paths[name] is None:
            b.refuse("host_path_unset", f"inventory.host_paths.{name}",
                     "Every host path is needed to compile a plan.")
    if ("A" in phases or "C" in phases) and inv["toolchain"]["rust_toolchain"] is None:
        b.refuse("toolchain_unset", "inventory.toolchain.rust_toolchain",
                 "R2 A7/C4 need the pinned Rust toolchain.")
    if "A" in phases and b.images["public_cuda_probe"]["reference"] is None:
        b.refuse("image_reference_unset", "inventory.images.public_cuda_probe",
                 "The host probe pulls only a repository@sha256 reference.")
    if "A" in phases:
        if b.record["required_resources"] is None:
            b.refuse("host_requirements_unset", "action_record.required_resources",
                     "A1's host-fit check needs the rental's recorded required resources.")
        uv = inv["guest_runtime"]["uv"]
        if uv["archive_sha256"] != UV_ARCHIVE_SHA256 or uv["archive_size_bytes"] != (
                UV_ARCHIVE_SIZE_BYTES) or UV_ARCHIVE_SHA256 is None:
            b.refuse("uv_archive_unverified", "inventory.guest_runtime.uv",
                     "The uv archive's published sha256 and size are required to compile.")
        if any(v is None for v in inv["guest_runtime"]["source_bundle"].values()):
            b.refuse("source_bundle_unset", "inventory.guest_runtime.source_bundle",
                     "Seal the source bundle (scripts/s2_prepare_bundle.py) and record it.")
    if "D" in phases:
        for role in ("renderer", "alpasim_base"):
            image = b.images[role]
            if image["kind"] == "registry_digest" and image["reference"] is None:
                b.refuse("image_reference_unset", f"inventory.images.{role}",
                         "Image phase needs every immutable reference.")
    if "E" in phases:
        if inv["harmonizer_source"] is None:
            b.refuse("owner_input_missing", "inventory.harmonizer_source",
                     "Harmonizer source is Huy's decision (R2 L8); not chosen here.")
        if not b.assets["nurec_scenes"]["files"]:
            b.refuse("owner_input_missing", "inventory.assets.nurec_scenes.files",
                     "Five registered scene rows (Huy) are required.")
        for role in ("vlm_config", "processor", "harmonizer"):
            if b.assets[role]["revision"] is None and (
                    role != "harmonizer" or inv["harmonizer_source"] == "pinned_mount"):
                b.refuse("revision_unresolved", f"inventory.assets.{role}.revision",
                         "Immutable revision required before acquisition.")
        for row in b.assets["nurec_scenes"]["files"]:
            if row["revision"] is None:
                b.refuse("revision_unresolved", f"inventory.assets.nurec_scenes.{row['uuid']}",
                         "Resolve each scene's moving ref to a commit before acquisition.")


# G4R item 5: an explicit plan-level gap instead of an exporter start/stop stage, because the
# anchors support no clean stage. Each clause names the anchor it rests on.
_DCGM_EXPORTER_GAP = _issue(
    "dcgm_exporter_run_long_stage_absent", "stages",
    "No run-long dcgm-exporter start/stop stage and no scripts/s2_dcgm_scraper.py invocation "
    "is compiled, so a run has no in-run producer for "
    "gate.s2_telemetry_sources.parse_dcgm_scrapes. Why no clean stage: the exporter digest "
    "exists only in scripts/vast_exporter_control.sh:18 (no exporter role in "
    "gate.s2_runtime.IMAGE_ROLES, so the inventory cannot pin or pre-stage it); its start "
    "depends on per-process safety state a static argv cannot carry (host-engine and :9400 "
    "listener refusal :35-46, mktemp owner token and label :47-49, $$-suffixed name and "
    "--cidfile :16, :128-131); its only stop is the same process's EXIT trap with "
    "owner-verified removal (:55-73, :114-127), so no standalone stop stage exists; the script "
    "is a fixed 14-sample probe (:143-160), not run-long; the scraper is a foreground loop "
    "whose interval, duration and timeout are required inputs with no default; and this "
    "bootstrap plan holds no unit runs to wrap (units run from gate.s2_executor's plan via "
    "scripts/s2_run_units.py, and stages here run one at a time). Needed: an exporter image "
    "role, a start/stop split of the control script and executor-plan wiring.", "info")


def _empty_plan(scope=None):
    return {
        "artifact_kind": "s2_bootstrap_plan", "schema_version": 1, "scope": scope,
        "compiled": False,
        "producer": {"module": "gate.s2_bootstrap", "source_sha256": _source_sha256()},
        "inputs": {"inventory_sha256": None, "action_record_sha256": None},
        "phases": [], "roots": {}, "authorizations": {}, "secrets_required": [], "stages": [],
        "issues": [], "execution_status": "qualify_on_host", "level": "L2",
        "units": dict(UNITS), "compiler": compiler_identity("s2_runtime"),
    }


def _seal(plan):
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = identity_hash(plan)
    return plan


def build_execution_plan(inventory, action_record, *, phases):
    """Compile ordered R2 A-E stages, or refuse with issues and no stages (L1 logic, L2 argv).

    ``phases`` is a non-empty list of distinct letters from A-E (normalized to A-E order).
    Deterministic: the same inputs give the same plan and ``plan_sha256``. Never raises.
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
    for name, value in (("inventory_sha256", inventory), ("action_record_sha256", action_record)):
        try:
            plan["inputs"][name] = identity_hash(value)
        except (TypeError, ValueError, UnicodeError):
            plan["inputs"][name] = None
    report = validate_runtime_inventory(inventory)
    plan["scope"] = report["scope"]
    issues = plan["issues"]
    if not report["valid"]:
        bad = sorted({i["code"] for i in report["issues"] if i["severity"] == "error"})
        issues.append(_issue("inventory_invalid", "inventory",
                             f"Inventory has structural errors: {bad}."))
    elif any(i["code"] == "disk_insufficient_for_known_demand" for i in report["issues"]):
        issues.append(_issue("disk_insufficient_for_known_demand", "inventory.disk.capacity_bytes",
                             "Known component sizes already exceed the recorded disk capacity."))
    issues.extend(_validate_action_record(action_record))
    normalized = _normalize_phases(phases)
    if normalized is None:
        issues.append(_issue("invalid_phases", "phases",
                             "phases is a non-empty list of distinct letters from A-E."))
    if issues:
        return _seal(plan)
    plan["phases"] = normalized
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
    for phase, compile_phase in zip(PHASES, (_phase_a, _phase_b, _phase_c, _phase_d, _phase_e)):
        if phase in normalized:
            compile_phase(builder)
    builder.issues.append(dict(_DCGM_EXPORTER_GAP, evidence_refs=[]))  # S2 only, not _Builder
    candidate = dict(plan, roots=dict(builder.paths), authorizations=dict(
        sorted(builder.used_auth.items())), stages=builder.stages)
    candidate["secrets_required"] = sorted(
        {s["stdin_secret"] for s in builder.stages if s["stdin_secret"]}
        | {name for s in builder.stages for name in s["secret_env"].values()})
    candidate["issues"] = [i for i in builder.issues]
    if any(i["severity"] == "error" for i in builder.issues):
        issues.extend(i for i in builder.issues)
        return _seal(plan)
    secret_issues = [i for i in _plan_content_issues(candidate) if i["severity"] == "error"]
    if secret_issues:
        issues.extend(secret_issues)
        return _seal(plan)
    candidate["compiled"] = True
    return _seal(candidate)


# --- plan validation (also used by the guest runner) -------------------------------------


def _argv_issues(argv, where):
    issues = []
    for element in argv:
        if type(element) is not str or not element or "\x00" in element:
            issues.append(_issue("invalid_argv", where, "argv elements are non-empty strings."))
            continue
        if secret_like(element) or element in _FORBIDDEN_FLAGS or any(
                element.startswith(flag + "=") for flag in _FORBIDDEN_FLAGS):
            issues.append(_issue("secret_in_argv", where,
                                 "Credentials never appear in argv; use stdin or a secret file."))
    return issues


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _rule_issues(rule, roots, where, stage_has_argv):
    if type(rule) is not dict or type(rule.get("kind")) is not str or (
            rule["kind"] not in RULE_KINDS):
        return [_issue("invalid_rule", where, f"Rule kind is one of {list(RULE_KINDS)}.")]
    kind = rule["kind"]
    issues = []
    if kind in ("stdout_regex", "stdout_min_version") and not stage_has_argv:
        issues.append(_issue("invalid_rule", where, "Verify-only stages have no stdout."))
    located = []
    if kind == "free_space":
        located = rule.get("requirements") if type(rule.get("requirements")) is list else None
        unknown = rule.get("unknown_components")
        if located is None or type(unknown) is not list or not all(
                type(u) is str for u in unknown) or not all(
                type(i) is dict and _nonnegative_int(i.get("bytes"))
                and _nonnegative_int(i.get("filesystem_floor_bytes")) for i in located):
            return [_issue("invalid_rule", where, "free_space needs requirements (bytes and "
                           "filesystem_floor_bytes) and an unknown_components list.")]
    elif kind == "scene_catalog":
        located = rule.get("catalogs") if type(rule.get("catalogs")) is list else None
        if not located or not all(type(rule.get(k)) is str and rule[k] for k in (
                "scene_id", "uuid", "scene_path")) or not all(
                type(i) is dict and (i.get("sha256") is None or (
                    type(i.get("sha256")) is str and _SHA256.fullmatch(i["sha256"])))
                for i in located):
            return [_issue("invalid_rule", where, "scene_catalog needs catalogs (root, path, "
                           "sha256) and scene_id, uuid, scene_path strings.")]
    elif kind not in ("stdout_regex", "stdout_min_version", "probe"):
        located = [rule]
    for item in located:
        root = item.get("root") if type(item) is dict else None
        path = item.get("path") if type(item) is dict else None
        if type(path) is str:
            path = path.replace(ATTEMPT_TOKEN, "1")
        if (type(root) is not str or root not in roots or not is_clean_absolute(path)
                or not is_portable_absolute(roots[root])
                or not (path == roots[root]
                        or PurePosixPath(roots[root]) in PurePosixPath(path).parents)):
            issues.append(_issue("rule_path_outside_root", where,
                                 "Rule paths are clean absolute paths under a declared root."))
    if kind == "probe":
        if type(rule.get("argv")) is not list or not rule["argv"]:
            issues.append(_issue("invalid_rule", where, "probe needs argv."))
        else:
            issues.extend(_argv_issues(rule["argv"], where))
        timeout = rule.get("timeout_s")
        if (rule.get("cwd") != "/" and not is_portable_absolute(rule.get("cwd"))) or (
                type(timeout) is not int or not 0 < timeout <= 3_600):
            issues.append(_issue("invalid_rule", where, "probe needs an absolute cwd and a "
                                 "1..3600 s timeout."))
    if kind == "host_fit" and validate_requirements(rule.get("requirements")):
        issues.append(_issue("invalid_rule", where, "host_fit needs valid required resources "
                             "(gate.s2_host_fit.REQUIREMENT_FIELDS)."))
    if kind == "source_tree" and (type(rule.get("bundle_sha256")) is not str
                                  or _SHA256.fullmatch(rule["bundle_sha256"]) is None):
        issues.append(_issue("invalid_rule", where, "source_tree needs a bundle_sha256."))
    if kind == "archive_file" and (type(rule.get("name")) is not str
                                   or not is_safe_relative(rule["name"]) or "/" in rule["name"]):
        issues.append(_issue("invalid_rule", where, "Archive names are single safe components."))
    return issues


def _stage_strings(stage):
    """Argv, cwd and rule-path strings of a stage (where an attempt token could appear)."""
    values = [stage.get("cwd")]
    for key in ("argv", "cleanup_argv"):
        if type(stage.get(key)) is list:
            values.extend(stage[key])
    for key in ("expected_outputs", "idempotency_check"):
        for rule in stage.get(key) or []:
            if type(rule) is not dict:
                continue
            values.append(rule.get("path"))
            for nested in ("requirements", "catalogs"):
                items = rule.get(nested) if type(rule.get(nested)) is list else []
                values.extend(item.get("path") for item in items if type(item) is dict)
    return [value for value in values if type(value) is str]


def _attempt_issues(stage, where):
    scoped = stage["attempt_scoped"]
    tokened = any(ATTEMPT_TOKEN in value for value in _stage_strings(stage))
    if type(scoped) is not bool:
        return [_issue("invalid_stage", f"{where}.attempt_scoped", "attempt_scoped is boolean.")]
    if tokened and not scoped:
        return [_issue("invalid_stage", f"{where}.attempt_scoped",
                       f"Only attempt-scoped stages may carry {ATTEMPT_TOKEN}.")]
    if scoped and not (type(stage["argv"]) is list and any(
            type(a) is str and ATTEMPT_TOKEN in a for a in stage["argv"])):
        return [_issue("invalid_stage", f"{where}.attempt_scoped",
                       f"An attempt-scoped stage carries {ATTEMPT_TOKEN} in its argv.")]
    return []


def substitute_attempt(stage, attempt):
    """Copy of an attempt-scoped stage with ``ATTEMPT_TOKEN`` replaced in argv and rule paths."""
    if not stage.get("attempt_scoped"):
        return stage
    text = str(int(attempt))

    def rule_copy(rule):
        rule = dict(rule)
        if type(rule.get("path")) is str:
            rule["path"] = rule["path"].replace(ATTEMPT_TOKEN, text)
        return rule

    copy = dict(stage)
    copy["argv"] = [a.replace(ATTEMPT_TOKEN, text) for a in stage["argv"]]
    copy["expected_outputs"] = [rule_copy(r) for r in stage["expected_outputs"]]
    if stage["idempotency_check"] is not None:
        copy["idempotency_check"] = [rule_copy(r) for r in stage["idempotency_check"]]
    return copy


def _plan_content_issues(plan):
    issues = []
    roots = plan["roots"]
    secrets = {name for name in plan["secrets_required"] if type(name) is str}
    seen = set()
    for index, stage in enumerate(plan["stages"]):
        where = f"stages[{index}]"
        if type(stage) is not dict or set(stage) != set(_STAGE_FIELDS):
            issues.append(_issue("invalid_stage", where, "Stage fields are fixed."))
            continue
        if (type(stage["id"]) is not str or _STAGE_ID.fullmatch(stage["id"]) is None
                or stage["id"] in seen):
            issues.append(_issue("invalid_stage", f"{where}.id",
                                 "Stage IDs are unique [A-Za-z0-9_] names."))
        else:
            seen.add(stage["id"])
        issues.extend(_attempt_issues(stage, where))
        argv = stage["argv"]
        if argv is None:
            if not stage["expected_outputs"] or stage["secret_env"] or stage["stdin_secret"]:
                issues.append(_issue("invalid_stage", where,
                                     "Verify-only stages carry rules and no secrets."))
        elif type(argv) is not list or not argv:
            issues.append(_issue("invalid_argv", f"{where}.argv", "argv is a list or null."))
        else:
            issues.extend(_argv_issues(argv, f"{where}.argv"))
            if stage["cwd"] != "/" and not is_portable_absolute(stage["cwd"]):
                issues.append(_issue("invalid_stage", f"{where}.cwd", "cwd is absolute."))
        env_set = stage["env_set"]
        if (type(env_set) is not dict or type(stage["env_allowlist"]) is not list
                or type(stage["secret_env"]) is not dict):
            issues.append(_issue("invalid_stage", f"{where}.env", "Environment is declared."))
            continue
        for name, value in env_set.items():
            if _ENV_NAME.fullmatch(name) is None or type(value) is not str:
                issues.append(_issue("invalid_stage", f"{where}.env_set", "NAME=value strings."))
            elif secret_like(value) or name in stage["secret_env"] or any(
                    word in name for word in ("TOKEN", "PASSWORD", "SECRET", "API_KEY")):
                issues.append(_issue("secret_in_argv", f"{where}.env_set.{name}",
                                     "Secrets reach a stage only through secret_env files."))
        for name in stage["env_allowlist"]:
            if type(name) is not str or _ENV_NAME.fullmatch(name) is None or name in env_set or (
                    any(word in name for word in ("TOKEN", "PASSWORD", "SECRET", "API_KEY"))):
                issues.append(_issue("invalid_stage", f"{where}.env_allowlist",
                                     "Allowlisted names are plain, non-secret variable names."))
        if not all(type(k) is str and _ENV_NAME.fullmatch(k) and type(v) is str and v in secrets
                   for k, v in stage["secret_env"].items()):
            issues.append(_issue("invalid_stage", f"{where}.secret_env",
                                 "secret_env maps a variable to a declared secret file name."))
        if stage["stdin_secret"] is not None and (type(stage["stdin_secret"]) is not str
                                                  or stage["stdin_secret"] not in secrets):
            issues.append(_issue("invalid_stage", f"{where}.stdin_secret", "Undeclared secret."))
        if type(stage["requires"]) is not list:
            issues.append(_issue("invalid_stage", f"{where}.requires", "requires is a list."))
        else:
            for key in stage["requires"]:
                value = (plan["authorizations"].get(key) if type(key) is str
                         and type(plan["authorizations"]) is dict else None)
                if type(value) is not str or not value.strip() or is_placeholder(value):
                    issues.append(_issue("authorization_missing", f"{where}.requires.{key}",
                                         "Stage authority is absent from the plan."))
        if stage["cleanup_argv"] is not None:
            if type(stage["cleanup_argv"]) is not list or not stage["cleanup_argv"]:
                issues.append(_issue("invalid_stage", f"{where}.cleanup_argv", "argv or null."))
            else:
                issues.extend(_argv_issues(stage["cleanup_argv"], f"{where}.cleanup_argv"))
        if type(stage["timeout_s"]) is not int or not 0 < stage["timeout_s"] <= 86_400:
            issues.append(_issue("invalid_stage", f"{where}.timeout_s", "1..86400 seconds."))
        rules = stage["expected_outputs"]
        idem = stage["idempotency_check"]
        if type(rules) is not list or (idem is not None and type(idem) is not list):
            issues.append(_issue("invalid_stage", f"{where}.rules", "Rule lists."))
            continue
        for n, rule in enumerate(rules):
            issues.extend(_rule_issues(rule, roots, f"{where}.expected_outputs[{n}]",
                                       argv is not None))
        for n, rule in enumerate(idem or []):
            if type(rule) is dict and rule.get("kind") in ("stdout_regex", "stdout_min_version"):
                issues.append(_issue("invalid_rule", f"{where}.idempotency_check[{n}]",
                                     "Idempotency re-verifies current state, not old stdout."))
            issues.extend(_rule_issues(rule, roots, f"{where}.idempotency_check[{n}]", False))
    return issues


def validate_plan(plan):
    """Return issues for a plan the guest runner would execute; empty means acceptable (L1).

    Checks the registered kind and compiler identity, the hash seal, schema, roots, declared
    secrets, authorizations for every ``requires`` key, argv/env credential hygiene and rule
    confinement. It authenticates nothing.
    """
    if type(plan) is not dict:
        return [_issue("invalid_plan", "plan", "A plan is a JSON object.")]
    kind, issues = plan_kind(plan)
    if issues:
        return issues
    fields = plan_fields(kind)
    if set(plan) != fields:
        return [_issue("invalid_plan", "plan", f"Plan fields are exactly {sorted(fields)}.")]
    try:
        return _validate_plan(plan, PLAN_KINDS[kind])
    except (TypeError, AttributeError, KeyError, ValueError) as exc:
        return [_issue("invalid_plan", "plan",
                       f"Unexpected value type in the plan ({type(exc).__name__}).")]


def _validate_plan(plan, spec):
    issues = []
    try:
        body = {k: v for k, v in plan.items() if k != "plan_sha256"}
        if identity_hash(body) != plan["plan_sha256"]:
            issues.append(_issue("plan_hash_mismatch", "plan_sha256",
                                 "Plan content does not match its seal."))
    except (TypeError, ValueError, UnicodeError):
        return [_issue("invalid_plan", "plan", "Plan is not strict JSON.")]
    if plan["artifact_kind"] != spec["plan_artifact_kind"] or plan["schema_version"] != 1:
        issues.append(_issue("invalid_plan", "schema_version", "Unknown plan kind or version."))
    if plan["compiled"] is not True:
        issues.append(_issue("plan_not_compiled", "compiled", "Refused plans never execute."))
    if plan["units"] != UNITS:
        issues.append(_issue("invalid_plan", "units", f"Units are exactly {UNITS}."))
    roots = plan["roots"]
    if type(roots) is not dict or not all(is_portable_absolute(v) for v in roots.values()):
        issues.append(_issue("invalid_plan", "roots", "Roots are portable absolute paths."))
        return issues
    checkout = spec["checkout_root"]
    for name, value in roots.items():
        if name != checkout and name != "home" and paths_overlap(value, roots.get(
                checkout, "/nonexistent-checkout")):
            issues.append(_issue("invalid_plan", f"roots.{name}", "Roots stay outside checkout."))
    if type(plan["secrets_required"]) is not list or not all(
            type(name) is str and name in SECRET_NAMES for name in plan["secrets_required"]):
        issues.append(_issue("invalid_plan", "secrets_required", f"Known: {list(SECRET_NAMES)}."))
        return issues
    if type(plan["authorizations"]) is not dict or type(plan["stages"]) is not list:
        issues.append(_issue("invalid_plan", "plan", "authorizations/stages are typed."))
        return issues
    for key, value in plan["authorizations"].items():
        if type(value) is not str or secret_like(value):
            issues.append(_issue("secret_in_argv", f"authorizations.{key}", "Credential-like."))
    issues.extend(_plan_content_issues(plan))
    return issues


# --- rule evaluation (used by the guest runner; never by plan compilation) --------------


def _fail(rule, reason, detail=None):
    return {"kind": rule.get("kind") if type(rule) is dict else None, "ok": False,
            "required": True, "reason": reason, "detail": detail, "recorded": None}


def _ok(rule, recorded=None, required=True):
    return {"kind": rule["kind"], "ok": True, "required": required, "reason": None,
            "detail": None, "recorded": recorded}


def _confined(roots, root_name, path):
    """Resolve path (following symlinks) and require it to stay under the resolved root."""
    root = roots.get(root_name)
    if type(root) is not str or type(path) is not str or not os.path.isabs(path):
        return None
    lexical_root = os.path.normpath(root)
    lexical = os.path.normpath(path)
    if lexical != lexical_root and not lexical.startswith(lexical_root.rstrip("/") + "/"):
        return None
    real_root = os.path.realpath(root)
    real = os.path.realpath(path)
    if real != real_root and not real.startswith(real_root.rstrip("/") + "/"):
        return None
    return real


def _open_regular(real):
    descriptor = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise IsADirectoryError(real)
    return os.fdopen(descriptor, "rb")


def _hash_file(real):
    digest, size = hashlib.sha256(), 0
    with _open_regular(real) as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _read_bounded(real, limit):
    with _open_regular(real) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("file exceeds the read bound")
    return data


def _locate(rule, roots):
    real = _confined(roots, rule.get("root"), rule.get("path"))
    if real is None:
        return None, _fail(rule, "outside_root")
    if not os.path.lexists(real):
        return None, _fail(rule, "missing")
    return real, None


def _rule_file(rule, roots):
    real, failure = _locate(rule, roots)
    if failure:
        return failure
    try:
        size, sha = _hash_file(real)
    except (OSError, ValueError) as exc:
        return _fail(rule, "unreadable", type(exc).__name__)
    if rule.get("size_bytes") is not None and size != rule["size_bytes"]:
        return _fail(rule, "size_mismatch", {"expected": rule["size_bytes"], "actual": size})
    if rule.get("sha256") is not None and sha != rule["sha256"]:
        return _fail(rule, "sha256_mismatch", {"expected": rule["sha256"], "actual": sha})
    return _ok(rule, {"size_bytes": size, "sha256": sha})


def _rule_file_regex(rule, roots):
    real, failure = _locate(rule, roots)
    if failure:
        return failure
    try:
        text = _read_bounded(real, _MAX_TEXT_READ).decode("utf-8", errors="replace")
    except (OSError, ValueError) as exc:
        return _fail(rule, "unreadable", type(exc).__name__)
    if re.search(rule["pattern"], text) is None:
        return _fail(rule, "pattern_mismatch")
    return _ok(rule)


def _rule_uv_lock(rule, roots):
    try:
        import tomllib  # Python >= 3.11; lazy so the phase-A runner imports on system 3.10
    except ImportError:
        return _fail(rule, "tomllib_unavailable")
    real, failure = _locate(rule, roots)
    if failure:
        return failure
    try:
        data = tomllib.loads(_read_bounded(real, _MAX_LOCK_READ).decode("utf-8"))
    except (OSError, ValueError, UnicodeError):
        return _fail(rule, "unparseable")
    packages = data.get("package") if type(data.get("package")) is list else []
    matches = [p for p in packages if type(p) is dict and p.get("name") == rule["package"]]
    if len(matches) != 1:
        return _fail(rule, "package_count", len(matches))
    source = matches[0].get("source")
    git = source.get("git") if type(source) is dict else None
    if type(git) is not str:
        return _fail(rule, "not_git_source")
    location, _, fragment = git.partition("#")
    repository = location.split("?", 1)[0]
    if repository != rule["repository"]:
        return _fail(rule, "repository_mismatch", repository)
    if fragment != rule["commit"]:
        return _fail(rule, "commit_mismatch", fragment or None)
    return _ok(rule, {"package": rule["package"], "repository": repository, "commit": fragment})


def _rule_usdz(rule, roots):
    real, failure = _locate(rule, roots)
    if failure:
        return failure
    try:
        with _open_regular(real) as stream, zipfile.ZipFile(stream) as archive:
            info = archive.getinfo("metadata.yaml")
            if info.file_size > _MAX_METADATA_READ:
                return _fail(rule, "metadata_too_large")
            text = archive.read(info).decode("utf-8")
    except (OSError, KeyError, ValueError, UnicodeError, zipfile.BadZipFile, EOFError,
            RuntimeError, NotImplementedError):
        # RuntimeError: encrypted member; NotImplementedError: unsupported compression.
        return _fail(rule, "unreadable")
    found = [m.group(2) for line in text.splitlines()
             if (m := _UUID_LINE.fullmatch(line)) is not None]
    if found != [rule["uuid"]]:
        return _fail(rule, "uuid_mismatch", found)
    return _ok(rule, {"uuid": found[0]})


def _rule_archive(rule, roots, archive_dir):
    if type(archive_dir) is not str:
        return _fail(rule, "no_archive_dir")
    real, failure = _locate(rule, roots)
    if failure:
        return failure
    try:
        size, sha = _hash_file(real)
        target = os.path.join(archive_dir, rule["name"])
        if os.path.lexists(target):
            if _hash_file(target) != (size, sha):
                return _fail(rule, "archive_conflict", rule["name"])
            return _ok(rule, {"name": rule["name"], "size_bytes": size, "sha256": sha})
        with _open_regular(real) as source:
            data = source.read()
        if hashlib.sha256(data).hexdigest() != sha:
            return _fail(rule, "changed_while_reading")
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError) as exc:
        return _fail(rule, "unreadable", type(exc).__name__)
    return _ok(rule, {"name": rule["name"], "size_bytes": size, "sha256": sha})


def _rule_tree(rule, roots):
    real, failure = _locate(rule, roots)
    if failure:
        return failure
    files = []
    try:
        for directory, dirnames, filenames in os.walk(real, followlinks=False):
            dirnames.sort()
            for name in sorted(filenames):
                if not any(name.endswith(suffix) for suffix in rule["suffixes"]):
                    continue
                full = os.path.join(directory, name)
                if os.path.islink(full):
                    return _fail(rule, "symlink_in_tree", os.path.relpath(full, real))
                size, sha = _hash_file(full)
                files.append({"path": os.path.relpath(full, real), "size_bytes": size,
                              "sha256": sha})
    except (OSError, ValueError) as exc:
        return _fail(rule, "unreadable", type(exc).__name__)
    if not files:
        return _fail(rule, "no_files")
    files.sort(key=lambda item: item["path"])
    return _ok(rule, {"files": files})


def _real_directory(rule, roots):
    real, failure = _locate(rule, roots)
    if failure:
        return None, failure
    if os.path.islink(rule["path"]) or not stat.S_ISDIR(os.lstat(real).st_mode):
        return None, _fail(rule, "not_a_directory")
    return real, None


def _rule_host_fit(rule, roots):
    """Probe-directory facts against the recorded required resources (gate.s2_host_fit)."""
    real, failure = _real_directory(rule, roots)
    if failure:
        return failure
    texts = {}
    for name in PROBE_FILES:
        path = os.path.join(real, name)
        if not os.path.lexists(path):
            continue
        try:
            texts[name] = _read_bounded(path, _MAX_TEXT_READ).decode("utf-8")
        except (OSError, ValueError, UnicodeError):
            continue  # unreadable, symlinked or oversized: the fact stays unverified
    result = assess_host_fit(texts, rule["requirements"])
    if result["status"] != "passed":
        return dict(_fail(rule, "host_fit_blocked",
                          sorted({i["code"] for i in result["issues"]})), recorded=result)
    return _ok(rule, result)


def _rule_source_tree(rule, roots):
    """The unpacked tree equals the sealed manifest: identity, bytes, no symlink, nothing extra.

    Skipped: the manifest file itself, top-level ``SOURCE_UNTRACKED_TOP_DIRS`` and any
    ``SOURCE_UNTRACKED_DIRS`` directory (real directories only; a symlinked one fails). Bytecode
    in a ``__pycache__`` beside a manifest ``.py`` file is checked, not trusted: an entry CPython
    would load instead of the sealed source fails ``stale_bytecode`` (``_shadowing_bytecode``).
    """
    real, failure = _real_directory(rule, roots)
    if failure:
        return failure
    try:
        manifest = read_json_bytes(_read_bounded(os.path.join(real, SOURCE_MANIFEST_NAME),
                                                 _MAX_LOCK_READ))
    except (OSError, ValueError) as exc:
        return _fail(rule, "manifest_unreadable", type(exc).__name__)
    problems = validate_source_manifest(manifest)
    if problems:
        return _fail(rule, "manifest_invalid", sorted({i["code"] for i in problems}))
    if identity_hash(manifest) != rule["bundle_sha256"]:
        return _fail(rule, "bundle_sha256_mismatch")
    expected = {row["path"]: row for row in manifest["files"]}
    seen, extra, changed = set(), [], []
    try:
        found = _walk_source_tree(real)
    except OSError as exc:
        return _fail(rule, "unreadable", type(exc).__name__)
    if type(found) is dict:
        return _fail(rule, "symlink_in_tree", found["symlink"])
    found, caches = found
    for relative, full in found:
        if relative not in expected:
            extra.append(relative)
            continue
        seen.add(relative)
        try:
            size, sha = _hash_file(full)
        except (OSError, ValueError) as exc:
            return _fail(rule, "unreadable", type(exc).__name__)
        row = expected[relative]
        if (size, sha) != (row["size_bytes"], row["sha256"]):
            changed.append(relative)
    missing = sorted(set(expected) - seen)
    if extra or changed or missing:
        return _fail(rule, "tree_mismatch", {"extra": sorted(extra)[:20],
                                             "changed": sorted(changed)[:20],
                                             "missing": missing[:20]})
    try:
        stale = _shadowing_bytecode(real, caches, expected)
    except (OSError, ValueError) as exc:
        return _fail(rule, "unreadable", type(exc).__name__)
    if type(stale) is dict:
        return _fail(rule, "symlink_in_tree", stale["symlink"])
    if stale:
        return _fail(rule, "stale_bytecode", stale[:20])
    return _ok(rule, {"bundle_sha256": rule["bundle_sha256"], "files_checked": len(seen)})


def _raise(error):
    raise error  # an unlisted directory is never treated as empty


def _walk_source_tree(real):
    """``(files, caches)`` or ``{"symlink": relative}``.

    ``files``: sorted ``(relative, full)`` regular-file candidates; ``caches``: the relative
    ``__pycache__`` directories (real directories) whose contents ``_shadowing_bytecode`` checks.
    """
    found, caches = [], []
    for directory, dirnames, filenames in os.walk(real, followlinks=False, onerror=_raise):
        relative_dir = os.path.relpath(directory, real)
        top = relative_dir == "."
        kept = []
        for name in sorted(dirnames):
            full = os.path.join(directory, name)
            if os.path.islink(full):
                return {"symlink": os.path.relpath(full, real).replace(os.sep, "/")}
            if name in SOURCE_UNTRACKED_DIRS or (top and name in SOURCE_UNTRACKED_TOP_DIRS):
                if name == "__pycache__":
                    caches.append(os.path.relpath(full, real).replace(os.sep, "/"))
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            full = os.path.join(directory, name)
            relative = os.path.relpath(full, real).replace(os.sep, "/")
            if os.path.islink(full):
                return {"symlink": relative}
            if top and name == SOURCE_MANIFEST_NAME:
                continue
            found.append((relative, full))
    return sorted(found), sorted(caches)


_PYC_HEADER_BYTES = 16


def _shadowing_bytecode(real, caches, expected):
    """Bytecode CPython could load in place of a sealed ``.py`` file (F3-2 review), sorted.

    The archive stores mtime 0, so without ``tar --touch`` CPython's default timestamp check
    reduces to the source size and an entry compiled from an earlier, same-size unpack keeps
    running. For each ``__pycache__/<stem>.<tag>[.opt-N].pyc`` beside a manifest
    ``<stem>.py`` (any interpreter tag) the 16-byte header decides:

    - flags 0 (timestamp): CPython loads it when the recorded mtime and size equal the source's
      ``int(st_mtime)`` and ``st_size`` (low 32 bits). Such an entry is accepted only when the
      source mtime is non-zero (the unpack set a real time) and the ``.pyc`` file was written
      at or after it (``st_mtime_ns``); otherwise it may predate the sealed bytes;
    - flags 0b01 (unchecked hash): loaded without any check, so always refused;
    - flags 0b11 (checked hash) or an unparseable header: CPython checks or ignores it.

    A symlink inside a ``__pycache__`` returns ``{"symlink": relative}``. Residual, stated: a
    ``.pyc`` written in the same filesystem timestamp tick as the unpack cannot be ordered.
    """
    stale = []
    for cache in caches:
        directory = os.path.join(real, *cache.split("/"))
        parent = cache.rsplit("/", 1)[0] if "/" in cache else ""
        for name in sorted(os.listdir(directory)):
            full = os.path.join(directory, name)
            relative = f"{cache}/{name}"
            if os.path.islink(full):
                return {"symlink": relative}
            if not name.endswith(".pyc") or "." not in name[: -len(".pyc")]:
                continue
            source = (f"{parent}/" if parent else "") + name.split(".", 1)[0] + ".py"
            if source not in expected:
                continue  # CPython only reads __pycache__ for a source that exists
            info = os.lstat(full)
            if not stat.S_ISREG(info.st_mode):
                continue
            with open(full, "rb") as stream:
                header = stream.read(_PYC_HEADER_BYTES)
            if len(header) < _PYC_HEADER_BYTES:
                continue
            flags = int.from_bytes(header[4:8], "little")
            if flags not in (0, 0b01):  # 0b11 checked hash: CPython verifies it
                continue
            if flags == 0b01:
                stale.append(relative)
                continue
            source_info = os.stat(os.path.join(real, *source.split("/")))
            recorded = (int.from_bytes(header[8:12], "little"),
                        int.from_bytes(header[12:16], "little"))
            if recorded != (int(source_info.st_mtime) & 0xFFFFFFFF,
                            source_info.st_size & 0xFFFFFFFF):
                continue  # CPython recompiles from the sealed source
            if source_info.st_mtime_ns == 0 or info.st_mtime_ns < source_info.st_mtime_ns:
                stale.append(relative)
    return sorted(stale)


def _rule_free_space(rule, roots):
    """Known demand vs free space per filesystem, plus a filesystem-size floor per device.

    Unknown sizes are never zero: they are recorded and ``demand_status`` is ``unverified``.
    """
    unknown = rule["unknown_components"]
    if type(unknown) is not list or not all(type(name) is str for name in unknown):
        return _fail(rule, "invalid_rule")
    devices = {}
    for item in rule["requirements"]:
        if (type(item) is not dict or not _nonnegative_int(item.get("bytes"))
                or not _nonnegative_int(item.get("filesystem_floor_bytes"))):
            return _fail(rule, "invalid_rule")
        real = _confined(roots, item.get("root"), item.get("path"))
        if real is None:
            return _fail(rule, "outside_root")
        existing = real
        while not os.path.exists(existing) and existing != "/":
            existing = os.path.dirname(existing)
        try:
            device = os.stat(existing).st_dev
            vfs = os.statvfs(existing)
        except OSError as exc:
            return _fail(rule, "unreadable", type(exc).__name__)
        entry = devices.setdefault(device, {
            "paths": [], "required_bytes": 0, "available_bytes": vfs.f_bavail * vfs.f_frsize,
            "filesystem_bytes": vfs.f_blocks * vfs.f_frsize, "floor_bytes": 0})
        entry["paths"].append(item["path"])
        entry["required_bytes"] += item["bytes"]
        entry["floor_bytes"] = max(entry["floor_bytes"], item["filesystem_floor_bytes"])
    recorded = {"devices": [devices[key] for key in sorted(devices)],
                "unknown_components": sorted(unknown),
                "demand_status": "unverified" if unknown else "passed"}
    if any(d["required_bytes"] > d["available_bytes"] for d in recorded["devices"]):
        return dict(_fail(rule, "insufficient_free_space"), recorded=recorded)
    if any(d["filesystem_bytes"] < d["floor_bytes"] for d in recorded["devices"]):
        return dict(_fail(rule, "filesystem_below_floor"), recorded=recorded)
    return _ok(rule, recorded)


def _catalog_rows(item, rule, roots):
    """Rows of one hash-checked scene catalog; (rows, failure)."""
    real, failure = _locate({"kind": rule["kind"], "root": item.get("root"),
                             "path": item.get("path")}, roots)
    if failure:
        return None, dict(failure, detail=item.get("path"))
    try:
        data = _read_bounded(real, _MAX_CATALOG_READ)
        text = data.decode("utf-8")
    except (OSError, ValueError, UnicodeError) as exc:
        return None, _fail(rule, "unreadable", type(exc).__name__)
    if item["sha256"] is not None and hashlib.sha256(data).hexdigest() != item["sha256"]:
        return None, _fail(rule, "catalog_sha256_mismatch", item["path"])
    reader = csv.reader(io.StringIO(text, newline=""))
    if next(reader, None) != SCENES_COLUMNS:
        return None, _fail(rule, "catalog_header_mismatch", item["path"])
    rows = []
    for values in reader:
        if not values:
            continue
        if len(values) != len(SCENES_COLUMNS):
            return None, _fail(rule, "catalog_row_malformed", item["path"])
        rows.append(dict(zip(SCENES_COLUMNS, values)))
    return rows, None


def _rule_scene_catalog(rule, roots):
    """Mirror alpasim@e121e59 sceneset.py: concatenate the catalogs (uuid unique across files,
    :34-66), filter by scene_id and keep the newest ``last_modified`` row (:91-95, :272-292,
    string order as polars reads every column as String); that row must be the registered one."""
    rows = []
    for item in rule["catalogs"]:
        found, failure = _catalog_rows(item, rule, roots)
        if failure:
            return failure
        rows.extend(found)
    repeated = sorted(uuid for uuid, n in Counter(r["uuid"] for r in rows).items() if n > 1)
    if repeated:
        return _fail(rule, "duplicate_uuid", repeated[:5])
    matches = [r for r in rows if r["scene_id"] == rule["scene_id"]]
    if not matches:
        return _fail(rule, "scene_not_in_catalog", rule["scene_id"])
    newest = max(r["last_modified"] for r in matches)
    top = [r for r in matches if r["last_modified"] == newest]
    if len(top) != 1:
        return _fail(rule, "ambiguous_scene", sorted(r["uuid"] for r in top))
    chosen = top[0]
    if chosen["uuid"] != rule["uuid"]:
        return _fail(rule, "uuid_mismatch", {"expected": rule["uuid"], "catalog": chosen["uuid"]})
    if chosen["path"] != rule["scene_path"]:
        return _fail(rule, "path_mismatch", {"expected": rule["scene_path"],
                                             "catalog": chosen["path"]})
    if chosen["artifact_repository"].strip().lower() != "huggingface":
        return _fail(rule, "not_huggingface", chosen["artifact_repository"])
    return _ok(rule, {key: chosen[key] for key in (
        "scene_id", "uuid", "path", "nre_version_string", "last_modified", "hf_revision")})


def evaluate_rule(rule, *, roots, stdout=b"", runner=None, env=None, archive_dir=None):
    """Evaluate one verification rule; never raises for bad input (fails closed).

    Returns ``{kind, ok, required, reason, detail, recorded}``. A rule with ``required: false``
    records its observation and always reports ``ok``.
    """
    if type(rule) is not dict or rule.get("kind") not in RULE_KINDS:
        return _fail(rule if type(rule) is dict else {}, "unknown_rule")
    kind = rule["kind"]
    try:
        if kind == "file":
            return _rule_file(rule, roots)
        if kind == "file_regex":
            return _rule_file_regex(rule, roots)
        if kind == "uv_lock_git_rev":
            return _rule_uv_lock(rule, roots)
        if kind == "usdz_metadata_uuid":
            return _rule_usdz(rule, roots)
        if kind == "archive_file":
            return _rule_archive(rule, roots, archive_dir)
        if kind == "tree":
            return _rule_tree(rule, roots)
        if kind == "free_space":
            return _rule_free_space(rule, roots)
        if kind == "scene_catalog":
            return _rule_scene_catalog(rule, roots)
        if kind == "host_fit":
            return _rule_host_fit(rule, roots)
        if kind == "source_tree":
            return _rule_source_tree(rule, roots)
        text = stdout.decode("utf-8", errors="replace") if type(stdout) is bytes else ""
        if kind == "stdout_regex":
            matched = re.search(rule["pattern"], text) is not None
            if rule.get("required", True) is False:
                return _ok(rule, {"matched": matched}, required=False)
            return _ok(rule) if matched else _fail(rule, "pattern_mismatch")
        if kind == "stdout_min_version":
            match = re.search(rule["pattern"], text)
            if match is None:
                return _fail(rule, "version_not_found")
            found = match.group(1)
            if version_tuple(found) < version_tuple(rule["minimum"]):
                return _fail(rule, "version_below_minimum", found)
            return _ok(rule, {"version": found})
        if runner is None:
            return _fail(rule, "no_runner")
        result = runner(list(rule["argv"]), cwd=rule["cwd"], env=dict(env or {}), stdin=None,
                        timeout_s=rule["timeout_s"])
        output = result.stdout.decode("utf-8", errors="replace")
        if result.timed_out or result.returncode != 0 or re.search(rule["pattern"],
                                                                   output) is None:
            return _fail(rule, "probe_mismatch", result.returncode)
        return _ok(rule)
    except (KeyError, TypeError, ValueError, re.error) as exc:
        return _fail(rule, "invalid_rule", type(exc).__name__)
    except Exception as exc:  # noqa: BLE001 - "never raises": any other failure fails closed
        return _fail(rule, "rule_error", type(exc).__name__)
