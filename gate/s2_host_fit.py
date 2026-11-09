"""Delivered-host fit: host-probe facts against an action record's required resources.

Tickets: completeness-audit CPU-09 guest runtime (F3: "compare delivered host facts from the
host probe ... to the action record's required resources, returning a blocking issue for any
shortfall (no automatic replacement host)"), CPU-01 host inventory, R2 A1/H1
(``docs/research/2026-11-08-codebase-build/lane-r2-alpasim-runtime.md``). Agent-authored
evaluator engineering under D-018/D-024; serves S2.1/S2.2/S2.7 [C] (G2, G7). Not a
scientific registration, rental, admission or host qualification.

Inputs. ``texts`` maps the probe's output file names (``PROBE_FILES``) to their text, or
``None`` when the file is absent; the files are the ones
``scripts/vast_s2_host_probe.sh`` writes into its per-attempt output directory: ``checks.tsv``
(check name, category, exit code) and one ``<check>.log`` per check. A fact is used only when
its check row in ``checks.tsv`` exists exactly once with exit code 0 and its log parses
strictly; anything else is ``host_fact_unverified`` (blocking), never a pass and never zero.
``requirements`` is the action record's ``required_resources`` object (``REQUIREMENT_FIELDS``);
every value is a recorded rental input with no default here.

Facts and their probe sources (literal shapes in ``tests/test_s2_host_fit.py`` are the
2026-11-01 Vast host's output, ``evidence/s2-vast-host-20261101-v1/probe-v2``):

- GPUs: ``gpu_identity.log`` = ``nvidia-smi --query-gpu=index,name,uuid,driver_version,
  memory.total,pci.bus_id --format=csv`` (memory in MiB, converted to bytes x 1,048,576).
- CUDA compute capability: ``gpu_compute_cap.log`` = ``nvidia-smi --query-gpu=index,compute_cap
  --format=csv``. The probe does not yet run this check (requested shared change), so until it
  does every real host reports ``host_fact_unverified`` for it: fail closed.
- vCPU: ``cpu_count.log`` (``nproc``); guest RAM: ``memory_bytes.log`` (``free -b``, ``Mem:``
  total); free bytes: ``docker_root_free.log`` and ``data_root_free.log`` (``df -B1 --``, the
  ``Available`` column; a wrapped long device name is accepted).

Comparisons (each shortfall is its own blocking code): at least ``gpu_count`` GPUs; every GPU's
name equals ``gpu_name`` exactly (the Workstation/Server edition is part of the name); every
GPU's memory >= ``min_vram_bytes``; every GPU's driver major equals ``driver_branch`` (a
driver family, e.g. ``"580"``); every GPU's compute capability >= ``min_cuda_capability``;
RAM, vCPU and both free-space values >= their minimum. A shortfall blocks; the module never
proposes, searches for or selects a replacement host (``automatic_replacement: false``).

Verification level per function: ``validate_requirements``, ``parse_probe_facts``,
``assess_host_fit`` and ``host_fit_report`` are **L1** (pure, literal known answers and
negative controls). The probe output they read is **L3**: only an admitted host produces it,
and a pass here is a fit check of recorded text, not ``HOST_QUALIFIED``.

Never: runs a process, reads a file, opens a network connection, reads a credential, chooses a
rental resource or replaces a host. Python 3.10-compatible standard library only, because the
guest runner imports it under the system interpreter before the managed runtime exists.
"""

import hashlib
import re
from pathlib import Path

from gate.s2_identity import identity_hash
from gate.s2_runtime import is_placeholder, secret_like

SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
MIB = 1_048_576
CHECKS_FILE = "checks.tsv"
# fact -> (probe check name, log file)
FACT_SOURCES = {
    "gpus": ("gpu_identity", "gpu_identity.log"),
    "compute_capability": ("gpu_compute_cap", "gpu_compute_cap.log"),
    "vcpu": ("cpu_count", "cpu_count.log"),
    "guest_ram_bytes": ("memory_bytes", "memory_bytes.log"),
    "docker_data_root_available_bytes": ("docker_root_free", "docker_root_free.log"),
    "data_root_available_bytes": ("data_root_free", "data_root_free.log"),
}
PROBE_FILES = (CHECKS_FILE, *sorted({log for _, log in FACT_SOURCES.values()}))
REQUIREMENT_FIELDS = (
    "gpu_name", "gpu_count", "min_vram_bytes", "driver_branch", "min_cuda_capability",
    "min_guest_ram_bytes", "min_vcpu", "min_docker_data_root_free_bytes",
    "min_data_root_free_bytes",
)
UNITS = {
    "min_vram_bytes": "bytes", "memory_total_bytes": "bytes", "min_guest_ram_bytes": "bytes",
    "guest_ram_bytes": "bytes", "min_docker_data_root_free_bytes": "bytes",
    "docker_data_root_available_bytes": "bytes", "min_data_root_free_bytes": "bytes",
    "data_root_available_bytes": "bytes", "vcpu": "count", "gpu_count": "count",
}
GPU_HEADER = "index, name, uuid, driver_version, memory.total [MiB], pci.bus_id"
CAP_HEADER = "index, compute_cap"
_CHECKS_HEADER = "check\tcategory\texit_code"
_CAPABILITY = re.compile(r"(\d{1,3})\.(\d{1,3})")
_DRIVER = re.compile(r"(\d{2,4})\.\d{1,4}(?:\.\d{1,4})?")
_BRANCH = re.compile(r"\d{2,4}")
_INT = re.compile(r"\d{1,20}")
_MIB = re.compile(r"(\d{1,9}) MiB")
_PERCENT = re.compile(r"\d{1,3}%|-")


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


def _issue(code, path, message, severity="blocking"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _positive_int(value):
    return type(value) is int and value > 0


def validate_requirements(requirements):
    """Issues for a ``required_resources`` object; empty means structurally usable (L1).

    Every field is required and typed; there are no defaults. Never raises.
    """
    where = "required_resources"
    if type(requirements) is not dict or set(requirements) != set(REQUIREMENT_FIELDS):
        return [_issue("invalid_requirements", where,
                       f"required_resources fields are exactly {list(REQUIREMENT_FIELDS)}.",
                       "error")]
    issues = []
    name = requirements["gpu_name"]
    if (type(name) is not str or not name.strip() or name != name.strip()
            or is_placeholder(name) or secret_like(name) or not name.isprintable()):
        issues.append(_issue("invalid_requirements", f"{where}.gpu_name",
                             "gpu_name is the exact nvidia-smi name, edition included.",
                             "error"))
    for field in ("gpu_count", "min_vram_bytes", "min_guest_ram_bytes", "min_vcpu",
                  "min_docker_data_root_free_bytes", "min_data_root_free_bytes"):
        if not _positive_int(requirements[field]):
            issues.append(_issue("invalid_requirements", f"{where}.{field}",
                                 f"{field} is a positive integer.", "error"))
    branch = requirements["driver_branch"]
    if type(branch) is not str or _BRANCH.fullmatch(branch) is None:
        issues.append(_issue("invalid_requirements", f"{where}.driver_branch",
                             "driver_branch is the driver major version, e.g. \"580\".",
                             "error"))
    capability = requirements["min_cuda_capability"]
    if type(capability) is not str or _CAPABILITY.fullmatch(capability) is None:
        issues.append(_issue("invalid_requirements", f"{where}.min_cuda_capability",
                             "min_cuda_capability is <major>.<minor>, e.g. \"12.0\".",
                             "error"))
    return issues


# --- probe parsing ----------------------------------------------------------------------


class _Unverified(ValueError):
    pass


def _lines(text):
    if type(text) is not str:
        raise _Unverified("probe file absent")
    return [line.rstrip("\r") for line in text.split("\n") if line.strip()]


def _check_status(checks_text):
    """check name -> exit code, for names that appear exactly once; duplicates are dropped."""
    rows = _lines(checks_text)
    if not rows or rows[0] != _CHECKS_HEADER:
        raise _Unverified("checks.tsv header mismatch")
    seen, repeated = {}, set()
    for row in rows[1:]:
        parts = row.split("\t")
        if len(parts) != 3 or _INT.fullmatch(parts[2]) is None:
            raise _Unverified("checks.tsv row malformed")
        if parts[0] in seen:
            repeated.add(parts[0])
        seen[parts[0]] = int(parts[2])
    return {name: code for name, code in seen.items() if name not in repeated}


def _parse_gpus(text):
    rows = _lines(text)
    if not rows or rows[0] != GPU_HEADER:
        raise _Unverified("gpu_identity header mismatch")
    gpus = []
    for row in rows[1:]:
        parts = row.split(", ")
        if len(parts) != 6 or _INT.fullmatch(parts[0]) is None:
            raise _Unverified("gpu_identity row malformed")
        index, name, _uuid, driver, memory, _bus = parts
        size = _MIB.fullmatch(memory)
        if not name or _DRIVER.fullmatch(driver) is None or size is None:
            raise _Unverified("gpu_identity values malformed")
        gpus.append({"index": int(index), "name": name, "driver_version": driver,
                     "memory_total_bytes": int(size.group(1)) * MIB,
                     "compute_capability": None})
    if not gpus or len({g["index"] for g in gpus}) != len(gpus):
        raise _Unverified("gpu_identity lists no GPU or repeats an index")
    return gpus


def _parse_capabilities(text):
    rows = _lines(text)
    if not rows or rows[0] != CAP_HEADER:
        raise _Unverified("gpu_compute_cap header mismatch")
    caps = {}
    for row in rows[1:]:
        parts = row.split(", ")
        if (len(parts) != 2 or _INT.fullmatch(parts[0]) is None
                or _CAPABILITY.fullmatch(parts[1]) is None or int(parts[0]) in caps):
            raise _Unverified("gpu_compute_cap row malformed")
        caps[int(parts[0])] = parts[1]
    return caps


def _parse_nproc(text):
    rows = _lines(text)
    if len(rows) != 1 or _INT.fullmatch(rows[0].strip()) is None or int(rows[0]) < 1:
        raise _Unverified("nproc output malformed")
    return int(rows[0])


def _parse_free(text):
    rows = _lines(text)
    if not rows or rows[0].split()[:1] != ["total"]:
        raise _Unverified("free -b header mismatch")
    memory = [row.split() for row in rows[1:] if row.split()[:1] == ["Mem:"]]
    if len(memory) != 1 or len(memory[0]) < 2 or _INT.fullmatch(memory[0][1]) is None:
        raise _Unverified("free -b Mem: row malformed")
    return int(memory[0][1])


def _parse_df(text):
    """``df -B1 -- <path>``: one record; tokens are joined so a wrapped device name parses."""
    rows = _lines(text)
    header = rows[0].split() if rows else []
    if header[:2] != ["Filesystem", "1B-blocks"] or header[2:5] != ["Used", "Available",
                                                                      "Use%"]:
        raise _Unverified("df -B1 header mismatch")
    tokens = " ".join(rows[1:]).split()
    if (len(tokens) < 6 or not all(_INT.fullmatch(t) for t in tokens[1:4])
            or _PERCENT.fullmatch(tokens[4]) is None
            or any(_PERCENT.fullmatch(t) for t in tokens[5:])):
        raise _Unverified("df -B1 record malformed or more than one record")
    return int(tokens[3])


_PARSERS = {
    "gpus": _parse_gpus, "compute_capability": _parse_capabilities, "vcpu": _parse_nproc,
    "guest_ram_bytes": _parse_free, "docker_data_root_available_bytes": _parse_df,
    "data_root_available_bytes": _parse_df,
}


def parse_probe_facts(texts):
    """``(facts, issues)`` from probe file texts (L1; never raises).

    A fact whose check is absent, repeated or non-zero in ``checks.tsv``, or whose log is
    absent or does not parse strictly, is ``None`` with a blocking ``host_fact_unverified``.
    """
    facts = {"gpus": None, "vcpu": None, "guest_ram_bytes": None,
             "docker_data_root_available_bytes": None, "data_root_available_bytes": None}
    issues = []
    if type(texts) is not dict:
        texts = {}
    try:
        status = _check_status(texts.get(CHECKS_FILE))
    except _Unverified as exc:
        status = None
        issues.append(_issue("host_fact_unverified", f"probe.{CHECKS_FILE}", str(exc)))
    capabilities = None
    for fact in sorted(FACT_SOURCES):
        check, log = FACT_SOURCES[fact]
        if status is None:
            issues.append(_issue("host_fact_unverified", f"facts.{fact}",
                                 f"{CHECKS_FILE} is unusable, so {log} is not trusted."))
            continue
        if status.get(check) != 0:
            issues.append(_issue("host_fact_unverified", f"facts.{fact}",
                                 f"Probe check {check} is absent, repeated or failed."))
            continue
        try:
            value = _PARSERS[fact](texts.get(log))
        except _Unverified as exc:
            issues.append(_issue("host_fact_unverified", f"facts.{fact}", f"{log}: {exc}."))
            continue
        if fact == "compute_capability":
            capabilities = value
        else:
            facts[fact] = value
    if facts["gpus"] is not None:
        for gpu in facts["gpus"]:
            cap = (capabilities or {}).get(gpu["index"])
            gpu["compute_capability"] = cap
            if cap is None and capabilities is not None:
                issues.append(_issue("host_fact_unverified",
                                     f"facts.gpus[{gpu['index']}].compute_capability",
                                     "No compute capability row for this GPU index."))
        if capabilities is not None and set(capabilities) - {g["index"] for g in facts["gpus"]}:
            issues.append(_issue("host_fact_unverified", "facts.compute_capability",
                                 "Compute capability rows name a GPU index nvidia-smi did "
                                 "not list."))
    return facts, issues


def _capability_tuple(text):
    match = _CAPABILITY.fullmatch(text)
    return int(match.group(1)), int(match.group(2))


def _compare(facts, req):
    issues = []
    gpus = facts["gpus"]
    if gpus is not None:
        if len(gpus) < req["gpu_count"]:
            issues.append(_issue("gpu_count_short", "facts.gpus",
                                 f"{len(gpus)} GPU(s) delivered; {req['gpu_count']} required."))
        for gpu in gpus:
            where = f"facts.gpus[{gpu['index']}]"
            if gpu["name"] != req["gpu_name"]:
                issues.append(_issue("gpu_edition_mismatch", f"{where}.name",
                                     f"Delivered {gpu['name']!r}; required "
                                     f"{req['gpu_name']!r}."))
            if gpu["memory_total_bytes"] < req["min_vram_bytes"]:
                issues.append(_issue("gpu_vram_short", f"{where}.memory_total_bytes",
                                     f"{gpu['memory_total_bytes']} B < "
                                     f"{req['min_vram_bytes']} B."))
            branch = _DRIVER.fullmatch(gpu["driver_version"]).group(1)
            if branch != req["driver_branch"]:
                issues.append(_issue("driver_branch_mismatch", f"{where}.driver_version",
                                     f"Driver {gpu['driver_version']} is not the "
                                     f"{req['driver_branch']} family."))
            cap = gpu["compute_capability"]
            if cap is not None and _capability_tuple(cap) < _capability_tuple(
                    req["min_cuda_capability"]):
                issues.append(_issue("cuda_capability_short", f"{where}.compute_capability",
                                     f"{cap} < {req['min_cuda_capability']}."))
    for fact, field, code in (
            ("guest_ram_bytes", "min_guest_ram_bytes", "guest_ram_short"),
            ("vcpu", "min_vcpu", "vcpu_short"),
            ("docker_data_root_available_bytes", "min_docker_data_root_free_bytes",
             "docker_data_root_free_short"),
            ("data_root_available_bytes", "min_data_root_free_bytes", "data_root_free_short")):
        if facts[fact] is not None and facts[fact] < req[field]:
            issues.append(_issue(code, f"facts.{fact}", f"{facts[fact]} < {req[field]}."))
    return issues


def assess_host_fit(texts, requirements):
    """``{status, facts, issues, automatic_replacement}`` for probe texts (L1; never raises).

    ``status`` is ``passed`` only with valid requirements, every fact verified and no
    shortfall; otherwise ``blocked``. Issues are sorted by ``(path, code)``.
    """
    try:
        issues = validate_requirements(requirements)
        facts, fact_issues = parse_probe_facts(texts)
        issues.extend(fact_issues)
        if not any(i["severity"] == "error" for i in issues):
            issues.extend(_compare(facts, requirements))
    except Exception as exc:  # noqa: BLE001 - "never raises": an unexpected shape fails closed
        facts = None
        issues = [_issue("invalid_input", "inputs",
                         f"Unexpected value in probe texts or requirements "
                         f"({type(exc).__name__}).", "error")]
    issues.sort(key=lambda i: (i["path"], i["code"]))
    return {"status": "passed" if not issues else "blocked", "facts": facts, "issues": issues,
            "automatic_replacement": False}


def _hash_or_none(value):
    try:
        return identity_hash(value)
    except (TypeError, ValueError, UnicodeError):
        return None


def host_fit_report(texts, requirements, *, scope):
    """The ``s2_host_fit_report`` artifact for probe texts and requirements (L1)."""
    result = assess_host_fit(texts, requirements)
    issues = list(result["issues"])
    if scope not in SCOPES:
        issues.append(_issue("invalid_scope", "scope", f"scope is one of {list(SCOPES)}.",
                             "error"))
    return {
        "artifact_kind": "s2_host_fit_report", "schema_version": SCHEMA_VERSION,
        "scope": scope if scope in SCOPES else None,
        "producer": {"module": "gate.s2_host_fit", "source_sha256": _source_sha256()},
        "inputs": {"probe_texts_sha256": _hash_or_none(texts),
                   "requirements_sha256": _hash_or_none(requirements)},
        "units": dict(UNITS), "level": "L1", "execution_status": "qualify_on_host",
        "status": "passed" if not issues else "blocked", "requirements": requirements,
        "facts": result["facts"], "issues": issues, "automatic_replacement": False,
    }
