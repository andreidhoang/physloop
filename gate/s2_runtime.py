"""S2 host runtime inventory: pinned sources, images, gated assets and disk demand.

Tickets: CPU-01 (source/runtime/asset inventory), R2 phases D-E and setup-draft facts S3-S11
(``docs/research/2026-11-08-codebase-build/lane-r2-alpasim-runtime.md``). Agent-authored under
D-018/D-024; engineering preparation for S2.1/S2.2/S2.7, not a scientific registration.

Verification ceiling:
- ``validate_runtime_inventory``, ``validate_source_manifest``, ``is_source_path`` and
  ``source_bundle_archive_name``: **L1** (pure structure, pin, path and arithmetic checks with
  literal known answers in ``tests/test_s2_runtime.py``).
- The pinned facts it compares against (AlpaSim/alpamayo1.5 commits, teacher revision, gated
  repository identities, upstream ``alpasim-base`` build route) are **L2**: read from the pinned
  sources and HF/registry metadata cited by R2, untested on a host.
- Whether images resolve, files download, bytes match or the disk fits is **L3**
  (``qualify_on_host``); a passing report establishes none of it.

Agent technical default (D-018): ``DISK_REQUEST_FLOOR_BYTES`` = 300 GB. Derivation (R2 sizing):
teacher 20.64 GiB + NRE image ~40-50 GiB + alpasim-base ~20-30 GiB + host env ~10-15 GiB +
Harmonizer 1.35 GiB and caches total about 120-170 GiB before scenes (the R2 estimate; the image
and env sizes are INFERRED). 170 GiB is 182.5 GB; the remaining ~117 GB covers unverified scene
bytes, at least eleven units of raw outputs (ASL 0.3-0.8 GB each, INFERRED, plus videos) and one
temporary export copy. It is a rental request floor, not a fit proof; replace it with measured
sizes after the first admitted host.

Owner inputs stay blank: scene rows (Huy's five-scene registration), ``harmonizer_source``
(R2 L8), every ``authorization_ref``. The Harmonizer terms reference is required for both sources:
the upstream default still downloads ``nvidia/Harmonizer`` under its licence in every on-arm
renderer (R2 E4), so its terms record must exist even though bootstrap stages nothing.

File names (``files[].path``) reach ``hf download`` as literal argv elements and are never shell
evaluated, so whitespace and shell metacharacters are permitted (CPU-09: "must arrive literally as
one argv element ... rather than being blanket-rejected"). Reasoned deviation, agent technical
default (D-018): a component starting with ``-`` is rejected (it would be parsed as an ``hf`` option;
emitting ``--`` would reorder the R2-anchored argv), as are control characters, NUL, empty, dot and
parent components. Unknown sizes and hashes are the explicit string
``"unverified"``; they are never treated as zero. The module never reads the network, the
filesystem (apart from hashing its own source for producer identity), a provider, a credential or
a clock, and never decides a scientific or licence question.

Guest runtime (F3; completeness-audit CPU-09 "guest runtime"). The PhysLoop runner needs
Python >= 3.11 (``tomllib`` users, ``datetime.UTC``) plus PyYAML and protobuf, while the Ubuntu
22.04 guest's ``python3`` is 3.10. The inventory therefore carries ``guest_runtime``: the pinned
uv release (``UV_VERSION``) with the SHA-256 and size of its published
``x86_64-unknown-linux-gnu`` archive, the uv-managed CPython request (``PYTHON_REQUEST``) and the
identity of the sealed PhysLoop source bundle (``scripts/s2_prepare_bundle.py``). Pins, all
agent technical defaults (D-018), L2 (public release metadata, never a downloaded binary):

- uv ``0.12.24`` (published 2026-10-08, the last 0.12 patch, 32 days old when pinned on
  2026-11-09; satisfies R2 A6's ``>= 0.9.17``). Archive SHA-256 ``b4dfaef4...3da6`` and size
  17,249,236 B were read on 2026-11-09 from three public sources that agree: the release's
  ``uv-x86_64-unknown-linux-gnu.tar.gz.sha256`` asset (whose own sha256 is the GitHub asset
  digest ``ffef1b08...0bb0``), its ``sha256.sum`` and the GitHub release-asset digest; the
  ``dist-manifest.json`` checksum (asset digest ``e74ba5c5...4ac0``) agrees too.
- CPython ``3.12.11`` (``cpython-3.12.11-linux-x86_64-gnu``): the interpreter the 2026-11-08
  clean frozen rebuild and the CPU suite ran on (``evidence/s2-clean-rebuild-20261108-v1``).
  uv 0.12.24's ``crates/uv-python-managed/download-metadata.json`` lists that key with archive
  sha256 ``PYTHON_ARCHIVE_SHA256`` (build 20251007), which uv itself verifies on install; newer
  3.12 patches (to 3.12.15) exist there and are the documented alternative.
- Locked runtime versions (``LOCKED_RUNTIME``) are the ``uv.lock`` versions of the main
  dependency group (``tests/test_s2_runtime.py`` re-reads the lock to keep them aligned).

An unverified uv archive hash or size, or an unset bundle identity, is a blocking issue, so a
plan that installs the runtime refuses to compile until it is supplied.
"""

import hashlib
import re
from pathlib import Path, PurePosixPath

from gate.s2_identity import identity_hash

ALPASIM_SHA = "e121e59141dc391f6c295513d6dadc840a419149"
ALPASIM_REPOSITORY = "https://github.com/NVlabs/alpasim"
ALPAMAYO15_SHA = "36aeb4c5938cbc2eb2aed33b22434773da4ab639"
ALPAMAYO15_REPOSITORY = "https://github.com/NVlabs/alpamayo1.5.git"
TEACHER_REVISION = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
PLATFORM = "linux/amd64"
DISK_REQUEST_FLOOR_BYTES = 300_000_000_000
DISK_REQUEST_FLOOR_BASIS = (
    "agent technical default (D-018): R2 sizing estimates 120-170 GiB before scenes "
    "(teacher 20.64 GiB, NRE ~40-50 GiB, alpasim-base ~20-30 GiB, host env ~10-15 GiB, "
    "Harmonizer 1.35 GiB; image/env sizes INFERRED); 300 GB adds room for unverified scenes, "
    ">= 11 units of raw outputs and one temporary export copy. Request floor, not a fit proof."
)
UNVERIFIED = "unverified"
SCOPES = ("synthetic", "public_source_fixture", "actual")
HOST_PATHS = (
    "physloop_root", "checkout", "data_root", "teacher_snapshot", "hf_home", "harmonizer_dir",
    "download_scratch", "evidence_dir", "docker_data_root", "home",
)
HARMONIZER_SOURCES = ("pinned_mount", "upstream_default_download")
# role -> (required kind(s), gated, terms label)
IMAGE_ROLES = {
    "renderer": (("registry_digest",), True, "NGC NRE image terms (Huy accepts)"),
    "alpasim_base": (("registry_digest", "local_build"), False, "public build inputs"),
    "public_cuda_probe": (("registry_digest",), False, "public image"),
}
# role -> (repository, repo_type, gated, terms label)
ASSET_ROLES = {
    "teacher": ("nvidia/Alpamayo-1.5-10B", "model", True, "OpenMDW-1.1"),
    "vlm_config": ("nvidia/Cosmos-Reason2-8B", "model", True, "gated HF access (Huy accepts)"),
    "processor": ("Qwen/Qwen3-VL-2B-Instruct", "model", False, "public"),
    "nurec_scenes": ("nvidia/PhysicalAI-Autonomous-Vehicles-NuRec", "dataset", True,
                     "NuRec dataset licence (Huy accepts)"),
    "harmonizer": ("nvidia/Harmonizer", "model", True, "NVIDIA Open Model License"),
}
# F3 guest runtime pins (agent technical defaults, D-018; derivation in the module docstring).
UV_VERSION = "0.12.24"
UV_TARGET = "x86_64-unknown-linux-gnu"
UV_ARCHIVE_URL = (f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/"
                  f"uv-{UV_TARGET}.tar.gz")
UV_ARCHIVE_SHA256 = "b4dfaef47d491a7296981f8374a4595f55dbf84e8937c8ecd2983574d8bb3da6"
UV_ARCHIVE_SIZE_BYTES = 17_249_236
UV_MINIMUM = "0.9.17"  # R2 A6 (AlpaSim relative exclude-newer, managed CPython 3.12)
PYTHON_VERSION = "3.12.11"
PYTHON_REQUEST = f"cpython-{PYTHON_VERSION}-linux-x86_64-gnu"
PYTHON_ARCHIVE_SHA256 = "f98121eb1fb2b05a25c1f3d2fe7cf08c3a2468c350785df3d84c2516e7280d3f"
# (distribution, import name, locked version) for the S2 guest's main dependency group.
LOCKED_RUNTIME = (
    ("pyyaml", "yaml", "6.0.3"),
    ("pyarrow", "pyarrow", "25.0.1"),
    ("protobuf", "google.protobuf", "7.36.2"),
)
SOURCE_MANIFEST_KIND = "s2_source_manifest"
SOURCE_MANIFEST_NAME = "physloop-source-manifest.json"
# Directories the guest itself creates inside the unpacked tree that the source-tree file walk
# skips: the frozen environment (top level only) and interpreter bytecode caches (anywhere). The
# caches are not trusted blindly: gate.s2_bootstrap._shadowing_bytecode refuses any entry CPython
# would load in place of a sealed source (F3-2 review).
SOURCE_UNTRACKED_TOP_DIRS = (".venv",)
SOURCE_UNTRACKED_DIRS = ("__pycache__",)
PILOT_SCENE_COUNT = 5
PLACEHOLDERS = {"tbd", "todo", "later", "____", "placeholder", "xxx", "n/a", "?"}
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"nvapi-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST_REFERENCE = re.compile(r"[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ABS = re.compile(r"/[A-Za-z0-9._/-]+")
_TAG = re.compile(r"[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9._-]+")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_SCENE_ID = re.compile(r"[A-Za-z0-9._-]+")
_TOOLCHAIN = re.compile(r"\d+\.\d+\.\d+")
_SOURCE_COMPONENT = re.compile(r"[A-Za-z0-9._][A-Za-z0-9._-]*")

_TOP = {
    "artifact_kind", "schema_version", "scope", "sources", "platform", "host_paths", "toolchain",
    "harmonizer_source", "images", "assets", "disk", "guest_runtime",
}
_GUEST_FIELDS = {"uv", "python", "source_bundle"}
_UV_FIELDS = {"version", "target", "archive_url", "archive_sha256", "archive_size_bytes"}
_BUNDLE_FIELDS = {"bundle_sha256", "archive_sha256", "archive_size_bytes"}
_MANIFEST_FIELDS = {
    "artifact_kind", "schema_version", "scope", "producer", "inputs", "units", "files", "issues",
}
_REGISTRY_FIELDS = {
    "role", "kind", "reference", "platform", "size_bytes", "compressed_bytes", "authorization_ref",
}
_LOCAL_FIELDS = {
    "role", "kind", "build", "image_id", "mutable_inputs", "platform", "size_bytes",
    "authorization_ref",
}
_BUILD_FIELDS = {"context", "dockerfile", "tag", "source_commit"}
_ASSET_FIELDS = {"role", "repository", "repo_type", "revision", "authorization_ref", "files"}
_FILE_FIELDS = {"path", "size_bytes", "sha256"}
_SCENE_FIELDS = {"scene_id", "uuid", "path", "revision", "size_bytes", "sha256"}


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


def is_portable_absolute(value):
    """Absolute POSIX path of ASCII letters, digits, dot, underscore, hyphen and slash only."""
    return (
        type(value) is str
        and _ABS.fullmatch(value) is not None
        and value != "/"
        and all(part not in ("", ".", "..") for part in value.split("/")[1:])
    )


def _clean_component(part):
    return part not in ("", ".", "..") and part.isprintable() and "\x00" not in part


def is_safe_relative(value):
    """Relative POSIX path of printable components (spaces and metacharacters allowed, since it
    is only ever a literal argv element); no empty, dot, parent or ``-``-leading components (a
    relative name is a positional argument, so ``-x`` would parse as an option)."""
    return (
        type(value) is str
        and not value.startswith("/")
        and all(_clean_component(part) and not part.startswith("-")
                for part in value.split("/"))
    )


def is_clean_absolute(value):
    """Absolute POSIX path of printable components without empty, dot or parent components
    (rule paths: a portable root plus an inventory file name). A ``-``-leading component is
    harmless here because the path starts with ``/``."""
    return (
        type(value) is str
        and value.startswith("/")
        and value != "/"
        and all(_clean_component(part) for part in value.split("/")[1:])
    )


def paths_overlap(first, second):
    first, second = PurePosixPath(first), PurePosixPath(second)
    return first == second or first in second.parents or second in first.parents


def secret_like(value):
    """True when a string resembles a credential (HF/NGC/GitHub token or private key)."""
    return type(value) is str and any(p.search(value) for p in SECRET_PATTERNS)


def is_placeholder(value):
    return type(value) is str and value.strip().lower() in PLACEHOLDERS


def is_source_path(value):
    """Relative bundle path of portable components (``[A-Za-z0-9._-]``, no ``-``-leading, empty,
    dot or parent component), so the same bytes land at the same path on every guest."""
    return (
        type(value) is str
        and not value.startswith("/")
        and all(part not in (".", "..") and _SOURCE_COMPONENT.fullmatch(part) is not None
                for part in value.split("/"))
    )


def source_bundle_archive_name(bundle_sha256):
    """File name of a sealed source bundle archive, bound to its manifest identity (L1)."""
    if type(bundle_sha256) is not str or _SHA256.fullmatch(bundle_sha256) is None:
        raise ValueError("bundle_sha256 is 64 lowercase hex")
    return f"physloop-source-{bundle_sha256}.tar"


def _manifest_issue(code, path, message):
    return {"code": code, "path": path, "severity": "error", "evidence_refs": [],
            "message": message}


def validate_source_manifest(manifest):
    """Issues for an ``s2_source_manifest`` (L1; never raises); empty means usable.

    ``files`` lists every bundled regular file once, sorted by path, as ``{path, size_bytes,
    sha256}``; the manifest never lists itself and never lists a path inside a directory the
    guest creates (``SOURCE_UNTRACKED_*``). A manifest that recorded issues was not sealed.
    """
    where = "manifest"
    if type(manifest) is not dict or set(manifest) != _MANIFEST_FIELDS:
        return [_manifest_issue("invalid_source_manifest", where,
                                f"Manifest fields are exactly {sorted(_MANIFEST_FIELDS)}.")]
    issues = []
    if (manifest["artifact_kind"] != SOURCE_MANIFEST_KIND
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1):
        issues.append(_manifest_issue("invalid_source_manifest", f"{where}.schema_version",
                                      f"Only {SOURCE_MANIFEST_KIND} schema_version 1 is known."))
    if manifest["scope"] not in SCOPES:
        issues.append(_manifest_issue("invalid_source_manifest", f"{where}.scope",
                                      f"scope is one of {list(SCOPES)}."))
    if manifest["units"] != {"size_bytes": "bytes"} or manifest["issues"] != []:
        issues.append(_manifest_issue("invalid_source_manifest", f"{where}.units",
                                      "units are {size_bytes: bytes}; issues are empty."))
    if type(manifest["producer"]) is not dict or type(manifest["inputs"]) is not dict:
        issues.append(_manifest_issue("invalid_source_manifest", f"{where}.producer",
                                      "producer and inputs are objects."))
    files = manifest["files"]
    if type(files) is not list or not files:
        issues.append(_manifest_issue("invalid_source_manifest", f"{where}.files",
                                      "files is a non-empty list."))
        return issues
    seen = set()
    previous = None
    for index, row in enumerate(files):
        rpath = f"{where}.files[{index}]"
        if type(row) is not dict or set(row) != {"path", "size_bytes", "sha256"}:
            issues.append(_manifest_issue("invalid_source_manifest", rpath,
                                          "File rows are exactly {path, size_bytes, sha256}."))
            continue
        name = row["path"]
        parts = name.split("/") if type(name) is str else []
        if (not is_source_path(name) or name == SOURCE_MANIFEST_NAME
                or parts[0] in SOURCE_UNTRACKED_TOP_DIRS
                or any(part in SOURCE_UNTRACKED_DIRS for part in parts)):
            issues.append(_manifest_issue("unsafe_source_path", f"{rpath}.path",
                                          "Portable relative path outside guest-made dirs."))
            continue
        folded = name.casefold()
        if folded in seen or (previous is not None and name <= previous):
            issues.append(_manifest_issue("invalid_source_manifest", f"{rpath}.path",
                                          "Paths are unique (case-insensitively) and sorted."))
        seen.add(folded)
        previous = name
        if type(row["size_bytes"]) is not int or row["size_bytes"] < 0:
            issues.append(_manifest_issue("invalid_size", f"{rpath}.size_bytes",
                                          "size_bytes is the file's non-negative byte count."))
        if type(row["sha256"]) is not str or _SHA256.fullmatch(row["sha256"]) is None:
            issues.append(_manifest_issue("invalid_sha256", f"{rpath}.sha256",
                                          "sha256 is 64 lowercase hex."))
    return issues


def _strings(value, path="inventory"):
    if type(value) is str:
        yield path, value
    elif type(value) is dict:
        for key in sorted(value, key=str):
            yield from _strings(value[key], f"{path}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def _size_ok(value):
    return value == UNVERIFIED or (type(value) is int and value > 0)


def _sha_ok(value):
    return value == UNVERIFIED or (type(value) is str and _SHA256.fullmatch(value) is not None)


class _Issues:
    def __init__(self):
        self.items = []

    def add(self, code, path, severity, message, evidence_refs=()):
        self.items.append({
            "code": code, "path": path, "severity": severity,
            "evidence_refs": list(evidence_refs), "message": message,
        })

    def error(self, code, path, message):
        self.add(code, path, "error", message)


def _check_auth(issues, path, ref, needed):
    """Authorization refs name recorded decisions; Huy's acceptance is never inferred."""
    if ref is None:
        if needed:
            issues.add("authorization_missing", path, "blocking",
                       "Gated item needs a recorded authorization reference before acquisition.")
        return "missing" if needed else "not_applicable"
    if not needed:
        issues.error("unexpected_authorization_ref", path,
                     "Public or not-acquired item must not carry an authorization reference.")
        return "invalid"
    if type(ref) is not str or not ref.strip() or is_placeholder(ref):
        issues.error("invalid_authorization_ref", path,
                     "Authorization reference must be a non-placeholder recorded reference.")
        return "invalid"
    return "recorded"


def _validate_images(images, issues, demand):
    summary = []
    if type(images) is not list:
        issues.error("invalid_inventory", "images", "images must be a list.")
        return summary
    seen = set()
    for index, item in enumerate(images):
        path = f"images[{index}]"
        if (type(item) is not dict or type(item.get("role")) is not str
                or item["role"] not in IMAGE_ROLES):
            issues.error("invalid_image", path, "Unknown or missing image role.")
            continue
        role = item["role"]
        if role in seen:
            issues.error("duplicate_role", f"{path}.role", f"Image role {role} repeats.")
            continue
        seen.add(role)
        kinds, gated, terms = IMAGE_ROLES[role]
        kind = item.get("kind")
        status = "passed"
        if kind not in ("registry_digest", "local_build"):
            issues.error("invalid_image", f"{path}.kind", "kind is registry_digest or local_build.")
            continue
        fields = _REGISTRY_FIELDS if kind == "registry_digest" else _LOCAL_FIELDS
        if set(item) != fields:
            issues.error("invalid_image", path, f"{kind} image fields are exactly {sorted(fields)}.")
            continue
        if kind not in kinds:
            issues.error("digest_required", f"{path}.kind",
                         f"{role} must be a pullable repository@sha256 reference.")
            status = "error"
        if item["platform"] != PLATFORM:
            issues.error("invalid_platform", f"{path}.platform", f"Platform must be {PLATFORM}.")
            status = "error"
        if not _size_ok(item["size_bytes"]):
            issues.error("invalid_size", f"{path}.size_bytes",
                         "On-disk size is a positive integer or 'unverified', never 0/null.")
            status = "error"
        else:
            demand.append((f"image:{role}", item["size_bytes"]))
        if kind == "registry_digest":
            reference = item["reference"]
            if reference is None:
                issues.add("image_reference_unset", f"{path}.reference", "blocking",
                           "Immutable repository@sha256 reference not yet resolved.")
                status = "blocked"
            elif type(reference) is not str or _DIGEST_REFERENCE.fullmatch(reference) is None:
                issues.error("mutable_image_reference", f"{path}.reference",
                             "Require repository@sha256:<64 lowercase hex>; tags are mutable.")
                status = "error"
            if not _size_ok(item["compressed_bytes"]):
                issues.error("invalid_size", f"{path}.compressed_bytes",
                             "Compressed size is a positive integer or 'unverified'.")
                status = "error"
        else:
            build = item["build"]
            if type(build) is not dict or set(build) != _BUILD_FIELDS:
                issues.error("invalid_build_inputs", f"{path}.build",
                             f"Build inputs are exactly {sorted(_BUILD_FIELDS)}.")
                status = "error"
            else:
                if build["source_commit"] != ALPASIM_SHA:
                    issues.error("pin_mismatch", f"{path}.build.source_commit",
                                 "Local base image must build from the reviewed AlpaSim pin.")
                    status = "error"
                if (build["context"] != "checkout" or build["dockerfile"] != "Dockerfile"
                        or type(build["tag"]) is not str or _TAG.fullmatch(build["tag"]) is None):
                    issues.error("invalid_build_inputs", f"{path}.build",
                                 "Build the checkout's Dockerfile under a local name:tag.")
                    status = "error"
            mutable = item["mutable_inputs"]
            if (type(mutable) is not list or not mutable
                    or not all(type(m) is str and m.strip() for m in mutable)):
                issues.error("mutable_inputs_unrecorded", f"{path}.mutable_inputs",
                             "Record every mutable build input (R2 D3); a local build is not "
                             "reproducible from its tag.")
                status = "error"
            image_id = item["image_id"]
            if image_id is None:
                issues.add("local_build_image_id_pending", f"{path}.image_id", "info",
                           "Built image ID is recorded by the bootstrap build stage.")
                status = "not_run" if status == "passed" else status
            elif type(image_id) is not str or _IMAGE_ID.fullmatch(image_id) is None:
                issues.error("invalid_local_image_id", f"{path}.image_id",
                             "Local image ID is a bare sha256:<64 lowercase hex>.")
                status = "error"
        auth = _check_auth(issues, f"{path}.authorization_ref", item["authorization_ref"], gated)
        if auth == "invalid":
            status = "error"
        elif auth == "missing" and status == "passed":
            status = "blocked"
        summary.append({"role": role, "kind": kind, "gated": gated, "terms": terms,
                        "authorization": auth, "status": status})
    for role in sorted(set(IMAGE_ROLES) - seen):
        issues.error("missing_role", "images", f"Image role {role} is required.")
    return summary


def _validate_files(asset, path, issues, demand, role):
    files = asset["files"]
    if type(files) is not list:
        issues.error("invalid_asset", f"{path}.files", "files must be a list.")
        return
    scenes = role == "nurec_scenes"
    if scenes and not files:
        issues.add("owner_input_missing", f"{path}.files", "blocking",
                   "Five registered scene rows (Huy) are required; none are chosen here.")
        return
    if not files:
        issues.error("invalid_asset", f"{path}.files", "List every file to acquire.")
        return
    if scenes and len(files) != PILOT_SCENE_COUNT:
        issues.error("pilot_scene_count", f"{path}.files",
                     "The pilot stages exactly five scenes, never a preset or a 441-scene set.")
    seen_paths, seen_scene = set(), set()
    for index, row in enumerate(files):
        rpath = f"{path}.files[{index}]"
        fields = _SCENE_FIELDS if scenes else _FILE_FIELDS
        if type(row) is not dict or set(row) != fields:
            issues.error("invalid_file", rpath, f"File rows are exactly {sorted(fields)}.")
            continue
        name = row["path"]
        if not is_safe_relative(name):
            issues.error("unsafe_relative_path", f"{rpath}.path",
                         "Relative portable path without traversal, empty or dot components.")
        elif name in seen_paths:
            issues.error("duplicate_file", f"{rpath}.path", "File path repeats.")
        else:
            seen_paths.add(name)
        if not _size_ok(row["size_bytes"]):
            issues.error("invalid_size", f"{rpath}.size_bytes",
                         "Size is a positive integer or 'unverified', never 0/null.")
        else:
            demand.append((f"asset:{role}:{name}", row["size_bytes"]))
        if not _sha_ok(row["sha256"]):
            issues.error("invalid_sha256", f"{rpath}.sha256",
                         "sha256 is 64 lowercase hex or 'unverified'.")
        if scenes:
            if type(row["scene_id"]) is not str or _SCENE_ID.fullmatch(row["scene_id"]) is None:
                issues.error("invalid_scene_id", f"{rpath}.scene_id", "Portable scene label.")
            if type(row["uuid"]) is not str or _UUID.fullmatch(row["uuid"]) is None:
                issues.error("invalid_uuid", f"{rpath}.uuid", "Lowercase artifact UUID.")
            if type(name) is str and not name.endswith(".usdz"):
                issues.error("invalid_file", f"{rpath}.path", "Scene artifact is a .usdz file.")
            for key in ("scene_id", "uuid"):
                marker = (key, row[key]) if type(row[key]) is str else None
                if marker in seen_scene:
                    issues.error("duplicate_scene", f"{rpath}.{key}", f"Scene {key} repeats.")
                seen_scene.add(marker)
            _check_revision(issues, f"{rpath}.revision", row["revision"])


def _check_revision(issues, path, revision):
    if revision is None:
        issues.add("revision_unresolved", path, "blocking",
                   "Resolve the moving reference to an immutable commit before acquisition.")
    elif type(revision) is not str or _COMMIT.fullmatch(revision) is None:
        issues.error("mutable_revision", path,
                     "Revision must be a 40-hex commit; tags/branches/short hashes move.")


def _validate_assets(assets, issues, demand):
    summary = []
    if type(assets) is not list:
        issues.error("invalid_inventory", "assets", "assets must be a list.")
        return summary
    seen = set()
    for index, item in enumerate(assets):
        path = f"assets[{index}]"
        if (type(item) is not dict or set(item) != _ASSET_FIELDS or type(item["role"]) is not str
                or item["role"] not in ASSET_ROLES):
            issues.error("invalid_asset", path, f"Asset fields are exactly {sorted(_ASSET_FIELDS)} "
                         "with a known role.")
            continue
        role = item["role"]
        if role in seen:
            issues.error("duplicate_role", f"{path}.role", f"Asset role {role} repeats.")
            continue
        seen.add(role)
        repository, repo_type, gated, terms = ASSET_ROLES[role]
        before = len(issues.items)
        if item["repository"] != repository or item["repo_type"] != repo_type:
            issues.error("repository_mismatch", f"{path}.repository",
                         f"{role} is {repo_type} {repository} (R2 E1-E4).")
        if role == "nurec_scenes":
            if item["revision"] is not None:
                issues.error("invalid_asset", f"{path}.revision",
                             "Scene revisions are per scene row (26.01/26.04 resolve separately).")
        elif role == "teacher":
            if item["revision"] != TEACHER_REVISION:
                issues.error("pin_mismatch", f"{path}.revision",
                             f"Teacher revision is pinned to {TEACHER_REVISION}.")
        else:
            _check_revision(issues, f"{path}.revision", item["revision"])
        # Gated items need their terms record whatever the source (review A2-F12).
        auth = _check_auth(issues, f"{path}.authorization_ref", item["authorization_ref"], gated)
        _validate_files(item, path, issues, demand, role)
        new = issues.items[before:]
        status = ("error" if any(i["severity"] == "error" for i in new)
                  else "blocked" if any(i["severity"] == "blocking" for i in new) else "passed")
        summary.append({"role": role, "gated": gated, "terms": terms, "authorization": auth,
                        "status": status})
    for role in sorted(set(ASSET_ROLES) - seen):
        issues.error("missing_role", "assets", f"Asset role {role} is required.")
    return summary


def _disk(inventory_disk, issues, demand):
    report = {
        "known_bytes": 0, "unknown_components": [], "total_bytes": None, "status": "unverified",
        "capacity_bytes": None, "request_floor_bytes": DISK_REQUEST_FLOOR_BYTES,
        "request_floor_basis": DISK_REQUEST_FLOOR_BASIS,
    }
    if type(inventory_disk) is not dict or set(inventory_disk) != {"capacity_bytes", "components"}:
        issues.error("invalid_disk", "disk", "disk fields are exactly capacity_bytes, components.")
        return report
    components = inventory_disk["components"]
    if type(components) is not list:
        issues.error("invalid_disk", "disk.components", "components must be a list.")
        return report
    names = set()
    for index, row in enumerate(components):
        path = f"disk.components[{index}]"
        if (type(row) is not dict or set(row) != {"name", "size_bytes"}
                or type(row["name"]) is not str or _SCENE_ID.fullmatch(row["name"]) is None
                or row["name"] in names):
            issues.error("invalid_disk", path, "Component rows are unique {name, size_bytes}.")
            continue
        names.add(row["name"])
        if not _size_ok(row["size_bytes"]):
            issues.error("invalid_size", f"{path}.size_bytes",
                         "Size is a positive integer or 'unverified', never 0/null.")
            continue
        demand.append((f"component:{row['name']}", row["size_bytes"]))
    known = sum(size for _, size in demand if type(size) is int)
    unknown = sorted(name for name, size in demand if size == UNVERIFIED)
    report.update(known_bytes=known, unknown_components=unknown)
    if not unknown:
        report.update(total_bytes=known, status="passed")
    capacity = inventory_disk["capacity_bytes"]
    if capacity is None:
        issues.add("disk_capacity_unset", "disk.capacity_bytes", "blocking",
                   "Record the rented disk size from the action record or host probe.")
    elif type(capacity) is not int or capacity <= 0:
        issues.error("invalid_disk", "disk.capacity_bytes", "Capacity is a positive integer.")
    else:
        report["capacity_bytes"] = capacity
        if capacity < DISK_REQUEST_FLOOR_BYTES:
            issues.add("disk_below_request_floor", "disk.capacity_bytes", "blocking",
                       "Capacity is below the 300 GB agent technical default (D-018).")
        if capacity < known:
            issues.add("disk_insufficient_for_known_demand", "disk.capacity_bytes", "blocking",
                       "Capacity is below the sum of known component sizes.")
    return report


def _validate_guest_runtime(runtime, issues):
    """F3 guest runtime: pinned uv archive, managed CPython request, sealed bundle identity."""
    if type(runtime) is not dict or set(runtime) != _GUEST_FIELDS:
        issues.error("invalid_inventory", "guest_runtime",
                     f"guest_runtime fields are exactly {sorted(_GUEST_FIELDS)}.")
        return
    uv = runtime["uv"]
    if type(uv) is not dict or set(uv) != _UV_FIELDS:
        issues.error("invalid_inventory", "guest_runtime.uv",
                     f"guest_runtime.uv fields are exactly {sorted(_UV_FIELDS)}.")
    else:
        for field, pinned in (("version", UV_VERSION), ("target", UV_TARGET),
                              ("archive_url", UV_ARCHIVE_URL)):
            if uv[field] != pinned:
                issues.error("pin_mismatch", f"guest_runtime.uv.{field}",
                             f"uv {field} is pinned to {pinned}.")
        for field, pinned, ok in (
                ("archive_sha256", UV_ARCHIVE_SHA256,
                 lambda v: type(v) is str and _SHA256.fullmatch(v) is not None),
                ("archive_size_bytes", UV_ARCHIVE_SIZE_BYTES,
                 lambda v: type(v) is int and v > 0)):
            value = uv[field]
            if value == UNVERIFIED or pinned is None:
                issues.add("uv_archive_unverified", f"guest_runtime.uv.{field}", "blocking",
                           "The uv archive's published checksum and size must be recorded "
                           "before a plan installs it.")
            elif not ok(value):
                issues.error("invalid_guest_runtime", f"guest_runtime.uv.{field}",
                             "64 lowercase hex sha256 / positive integer size, or 'unverified'.")
            elif value != pinned:
                issues.error("pin_mismatch", f"guest_runtime.uv.{field}",
                             f"uv {UV_VERSION} archive {field} is {pinned}.")
    python = runtime["python"]
    if type(python) is not dict or set(python) != {"request"}:
        issues.error("invalid_inventory", "guest_runtime.python",
                     "guest_runtime.python is exactly {request}.")
    elif python["request"] != PYTHON_REQUEST:
        issues.error("pin_mismatch", "guest_runtime.python.request",
                     f"The guest interpreter is pinned to {PYTHON_REQUEST}.")
    bundle = runtime["source_bundle"]
    if type(bundle) is not dict or set(bundle) != _BUNDLE_FIELDS:
        issues.error("invalid_inventory", "guest_runtime.source_bundle",
                     f"source_bundle fields are exactly {sorted(_BUNDLE_FIELDS)}.")
        return
    if any(bundle[field] is None for field in _BUNDLE_FIELDS):
        issues.add("source_bundle_unset", "guest_runtime.source_bundle", "blocking",
                   "Record the sealed bundle (scripts/s2_prepare_bundle.py) before a plan "
                   "unpacks it on the guest.")
    for field in ("bundle_sha256", "archive_sha256"):
        value = bundle[field]
        if value is not None and (type(value) is not str or _SHA256.fullmatch(value) is None):
            issues.error("invalid_guest_runtime", f"guest_runtime.source_bundle.{field}",
                         "64 lowercase hex or null.")
    size = bundle["archive_size_bytes"]
    if size is not None and (type(size) is not int or size <= 0):
        issues.error("invalid_guest_runtime", "guest_runtime.source_bundle.archive_size_bytes",
                     "Positive integer or null.")


def validate_runtime_inventory(inventory):
    """Validate a runtime inventory and return a deterministic report (L1; facts are L2).

    ``valid`` is False when any structural/pin/path ``error`` exists (a plan must not compile).
    ``status`` is ``passed`` only with no ``error`` or ``blocking`` issue; ``info`` notes do not
    block. A passing report is not host, input or rental qualification.
    """
    issues = _Issues()
    try:
        input_sha256 = identity_hash(inventory)
    except (TypeError, ValueError, UnicodeError):
        input_sha256 = None
    report = {
        "artifact_kind": "s2_runtime_inventory_report",
        "schema_version": 1,
        "scope": None,
        "producer": {"module": "gate.s2_runtime", "source_sha256": _source_sha256()},
        "input_sha256": input_sha256,
        "units": {"size_bytes": "bytes"},
        "level": "L1",
        "execution_status": "qualify_on_host",
        "valid": False,
        "status": "blocked",
        "issues": issues.items,
        "images": [],
        "assets": [],
        "disk_demand": None,
    }
    if input_sha256 is None or type(inventory) is not dict or set(inventory) != _TOP:
        issues.error("invalid_inventory", "inventory",
                     f"Inventory fields are exactly {sorted(_TOP)} (strict JSON).")
        return report
    try:
        _validate_inventory(inventory, issues, report)
    except (TypeError, AttributeError, KeyError, ValueError) as exc:
        # Backstop (review A2-F7): an unexpected JSON shape is an issue, never a crash.
        issues.error("invalid_inventory", "inventory",
                     f"Unexpected value type in the inventory ({type(exc).__name__}).")
    issues.items.sort(key=lambda i: (i["path"], i["code"]))
    severities = {i["severity"] for i in issues.items}
    report["valid"] = "error" not in severities
    report["status"] = "blocked" if severities & {"error", "blocking"} else "passed"
    return report


def _validate_inventory(inventory, issues, report):
    if inventory["artifact_kind"] != "s2_runtime_inventory":
        issues.error("invalid_inventory", "artifact_kind", "artifact_kind s2_runtime_inventory.")
    if type(inventory["schema_version"]) is not int or inventory["schema_version"] != 1:
        issues.error("unsupported_schema", "schema_version", "Only schema_version 1 is known.")
    if inventory["scope"] in SCOPES:
        report["scope"] = inventory["scope"]
    else:
        issues.error("invalid_scope", "scope", f"scope is one of {list(SCOPES)}.")
    for path, value in _strings(inventory):
        if secret_like(value):
            issues.error("secret_like_value", path, "Credential-like value; never embed secrets.")
    sources = inventory["sources"]
    expected_sources = {
        "alpasim": (ALPASIM_REPOSITORY, ALPASIM_SHA),
        "alpamayo1_5": (ALPAMAYO15_REPOSITORY, ALPAMAYO15_SHA),
    }
    if type(sources) is not dict or set(sources) != set(expected_sources):
        issues.error("invalid_inventory", "sources", "sources are exactly alpasim, alpamayo1_5.")
    else:
        for name, (repository, commit) in expected_sources.items():
            entry = sources[name]
            if type(entry) is not dict or set(entry) != {"repository", "commit"}:
                issues.error("invalid_inventory", f"sources.{name}", "{repository, commit}.")
            elif entry["repository"] != repository or entry["commit"] != commit:
                issues.error("pin_mismatch", f"sources.{name}",
                             f"{name} is pinned to {repository} @ {commit}.")
    if inventory["platform"] != PLATFORM:
        issues.error("invalid_platform", "platform", f"Platform must be {PLATFORM}.")
    host = inventory["host_paths"]
    if type(host) is not dict or set(host) != set(HOST_PATHS):
        issues.error("invalid_inventory", "host_paths", f"host_paths are exactly {HOST_PATHS}.")
    else:
        for name in HOST_PATHS:
            value = host[name]
            if value is None:
                issues.add("host_path_unset", f"host_paths.{name}", "blocking",
                           "Host path is chosen with the host; not defaulted.")
            elif not is_portable_absolute(value):
                issues.error("invalid_host_path", f"host_paths.{name}",
                             "Absolute portable ASCII path; no empty/dot/parent components.")
        checkout = host["checkout"]
        if is_portable_absolute(checkout):
            for name in HOST_PATHS:
                if name != "checkout" and is_portable_absolute(host[name]) and paths_overlap(
                        checkout, host[name]) and name != "home":
                    issues.error("host_path_overlap", f"host_paths.{name}",
                                 "Keep outputs outside the upstream checkout (git status stays "
                                 "clean, evidence never mixes with source).")
        # F3: the unpacked PhysLoop tree is verified file-for-file against its manifest, so
        # nothing else may be written inside it (and it never sits inside another root).
        source = host["physloop_root"]
        if is_portable_absolute(source):
            for name in HOST_PATHS:
                if name not in ("physloop_root", "checkout", "home") and is_portable_absolute(
                        host[name]) and paths_overlap(source, host[name]):
                    issues.error("host_path_overlap", f"host_paths.{name}",
                                 "Keep every other root outside the sealed PhysLoop source "
                                 "tree (its manifest check refuses unlisted files).")
    toolchain = inventory["toolchain"]
    if type(toolchain) is not dict or set(toolchain) != {"rust_toolchain"}:
        issues.error("invalid_inventory", "toolchain", "toolchain is exactly {rust_toolchain}.")
    elif toolchain["rust_toolchain"] is None:
        issues.add("toolchain_unset", "toolchain.rust_toolchain", "blocking",
                   "R2 A7 requires a pinned --default-toolchain; none is chosen here.")
    elif type(toolchain["rust_toolchain"]) is not str or _TOOLCHAIN.fullmatch(
            toolchain["rust_toolchain"]) is None:
        issues.error("invalid_toolchain", "toolchain.rust_toolchain", "Exact x.y.z toolchain.")
    harmonizer_source = inventory["harmonizer_source"]
    if harmonizer_source is None:
        issues.add("owner_input_missing", "harmonizer_source", "blocking",
                   "Harmonizer source (pinned_mount or upstream_default_download) is Huy's "
                   "decision (R2 L8).")
    elif harmonizer_source not in HARMONIZER_SOURCES:
        issues.error("invalid_harmonizer_source", "harmonizer_source",
                     f"harmonizer_source is one of {list(HARMONIZER_SOURCES)}.")
    _validate_guest_runtime(inventory["guest_runtime"], issues)
    demand = []
    report["images"] = _validate_images(inventory["images"], issues, demand)
    report["assets"] = _validate_assets(inventory["assets"], issues, demand)
    report["disk_demand"] = _disk(inventory["disk"], issues, demand)
